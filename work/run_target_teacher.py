"""Target-as-teacher, acceptance-aware tuning of a NanoQuant 1-bit FlexDraft drafter.

Same ADMM init / eval prompts / unfused path as run_tune_fact.py (pass B/C), so
tau numbers are directly comparable.  Differences are ONLY the tuning objective:

  tune_fact (pass C):  per-block MSE vs the FP *drafter's* hidden states
  ce        (here)  :  cross-entropy vs the *target's* greedy tokens at mask positions
  acc       (here)  :  -sum_k prod_{j<=k} p_j(y*_j)  -- smooth surrogate of expected
                       acceptance length.  Its gradient is exactly a CE whose position
                       weights are w_j = sum_{k>=j} prod_{i<=k} p_i (early positions
                       count more; a near-certain miss zeroes what follows).

All 40 projections' binary latents + scales are tuned jointly; anchor_bias_mlp,
mask_embedding, q/k norms and the whole target stay frozen.
"""
import argparse, json, math, os, sys, time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))

from fd_common import load_models, build_prompts, measure_tau
from flexdraft.draft_model import FlexDraftModel
from nanoquant.modules.linear import NanoQuantLinear          # before core.* (circular import)
from nanoquant.core.compress_block import factorize_and_replace, get_param_group_config
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.optimi import AdamW
from nanoquant.utils.utils import find_layers, set_seed
from run_tune_fact import force_unfused, rank_for, PROJ_ORDER

BS = 16  # block size


# ----------------------------------------------------------------- data
@torch.no_grad()
def build_windows(target, draft, tok, n_windows, seqlen, seed):
    from datasets import load_dataset
    from flexdraft.utils import _MATH_FINAL_ANSWER_INSTRUCTION
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    stream, i, need = [], 0, n_windows * seqlen + seqlen
    while len(stream) < need and i < len(ds):
        ex = ds[i]; i += 1
        msgs = [{"role": "user", "content": ex["question"] + _MATH_FINAL_ANSWER_INSTRUCTION}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False) + ex["answer"] + tok.eos_token
        stream.extend(tok.encode(txt))
    toks = torch.tensor(stream[:n_windows * seqlen], dtype=torch.long).view(n_windows, seqlen)

    dev = target.device; dtype = target.model.embed_tokens.weight.dtype
    split = draft.target_layer_ids[0]
    from transformers import DynamicCache
    Q = seqlen + BS
    mask_emb = draft.mask_embedding.to(dtype).view(1, 1, -1).expand(1, BS, -1)
    hid = torch.empty(n_windows, Q, target.config.hidden_size, dtype=dtype, device=dev)
    for j in range(n_windows):
        ids = toks[j:j + 1].to(dev)
        h = target.model.embed_tokens(ids)
        pe = target.model.rotary_emb(h, torch.arange(seqlen, device=dev).unsqueeze(0))
        cache = DynamicCache()
        for lyr in target.model.layers[:split]:
            h = lyr(hidden_states=h, attention_mask=None, position_embeddings=pe,
                    past_key_value=cache, use_cache=True)
        hid[j] = torch.cat([h, mask_emb], dim=1)[0]

    # target greedy labels: y*_n (anchor) and y*_{n+1..n+15}
    labels = torch.empty(n_windows, BS, dtype=torch.long, device=dev)
    for s in range(0, n_windows, 32):
        ids = toks[s:s + 32].to(dev)
        out = target.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                              max_new_tokens=BS, min_new_tokens=BS,
                              pad_token_id=tok.eos_token_id)
        labels[s:s + 32] = out[:, seqlen:seqlen + BS]

    min_val = torch.finfo(dtype).min
    m = torch.zeros(1, 1, Q, Q, dtype=dtype, device=dev)
    m[:, :, :seqlen, :seqlen] = torch.triu(torch.full((seqlen, seqlen), min_val, dtype=dtype, device=dev), 1)
    m[:, :, :seqlen, seqlen:] = min_val
    pe = target.model.rotary_emb(torch.empty(1, 1, dtype=dtype, device=dev),
                                 torch.arange(Q, device=dev).unsqueeze(0))
    return hid, labels, m, pe


# ----------------------------------------------------------------- model fn
def draft_logits(target, draft, hid, anchor_ids, mask, pe, seqlen):
    """Prefill-layout draft forward -> logits of mask tokens 1..15 (what fills Lc[1:])."""
    split = draft.target_layer_ids[0]
    h = hid
    for i, dl in enumerate(draft.layers):
        h = FlexDraftModel._dual_attn_layer_forward(
            draft, target_layer=target.model.layers[split + i], draft_layer=dl,
            hidden_states=h, candidate_len=seqlen, attention_mask=mask,
            position_embeddings=pe, past_key_value=None, fused_qkv=None)
    hm = target.model.norm(h[:, seqlen + 1: seqlen + BS, :])          # mask tokens 1..15
    logits = target.lm_head(hm)
    anchor_e = target.model.embed_tokens(anchor_ids).to(hm.dtype).unsqueeze(1).expand(-1, BS - 1, -1)
    return logits + draft.get_anchor_bias_logits(hm, anchor_e).to(logits.dtype)


def loss_fn(logits, tgt, mode):
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B,15)
    if mode == "ce":
        return -logp.mean()
    if mode == "acc":
        return -torch.exp(logp.cumsum(-1)).sum(-1).mean()       # -E[accepted length]
    raise ValueError(mode)


@torch.no_grad()
def prefill_tau(target, draft, hid, labels, mask, pe, seqlen, bs=8):
    """Exact greedy acceptance in the prefill layout: 1 + #leading mask positions
    whose argmax equals the target's greedy token.  Also returns E[tau] surrogate."""
    accs, surr = [], []
    for s in range(0, hid.shape[0], bs):
        lg = draft_logits(target, draft, hid[s:s + bs], labels[s:s + bs, 0], mask, pe, seqlen)
        tgt = labels[s:s + bs, 1:]
        match = (lg.argmax(-1) == tgt).long()
        accs.append(1 + match.cumprod(-1).sum(-1))
        logp = F.log_softmax(lg.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        surr.append(1 + torch.exp(logp.cumsum(-1)).sum(-1))
    a = torch.cat(accs).float(); s_ = torch.cat(surr)
    return {"prefill_tau": a.mean().item(), "prefill_tau_sem": (a.std() / math.sqrt(len(a))).item(),
            "surrogate": s_.mean().item()}


# ----------------------------------------------------------------- quantize
def admm_all(draft, cfg, min_rank):
    dev = next(draft.parameters()).device
    for i, dl in enumerate(draft.layers):
        for name in PROJ_ORDER:
            lin = find_layers(dl)[name]
            r = rank_for(lin.in_features, lin.out_features, cfg["bits"]); assert r > min_rank
            lin.register_buffer("i_norm", torch.ones(lin.in_features, device=dev), persistent=False)
            lin.register_buffer("o_norm", torch.ones(lin.out_features, device=dev), persistent=False)
            factorize_and_replace(dl, name, r, cfg)       # do_train=True -> latents stay trainable
        print(f"  ADMM layer {i} done", flush=True)


# ----------------------------------------------------------------- train
def train(target, draft, params, groups, hid, labels, mask, pe, seqlen, mode, epochs, bsz, seed):
    set_seed(seed)
    opt = AdamW(groups, weight_decay=0)
    n = hid.shape[0]; steps = epochs * math.ceil(n / bsz)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4 * groups[0]["lr"])
    hist = []
    with torch.enable_grad():
        for ep in range(epochs):
            perm = torch.randperm(n); tot = 0.0
            for s in range(0, n, bsz):
                idx = perm[s:s + bsz]
                lg = draft_logits(target, draft, hid[idx], labels[idx, 0], mask, pe, seqlen)
                loss = loss_fn(lg, labels[idx, 1:], mode)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
                tot += loss.item() * len(idx)
            hist.append(tot / n)
            print(f"    [{mode}] epoch {ep+1:02d}/{epochs} loss {tot/n:.4f}", flush=True)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--eval-samples", type=int, default=40); ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--train-windows", type=int, default=128); ap.add_argument("--heldout-windows", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=16); ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lrs", type=str, default="1e-5,1e-4,1e-3")
    ap.add_argument("--modes", type=str, default="ce,acc")
    ap.add_argument("--gen-eval", type=str, default="best,1e-5",
                    help="which configs get full generation-tau: 'best' per mode and/or explicit lrs")
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lrs = [float(x) for x in a.lrs.split(",")]; modes = a.modes.split(",")
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = {"args": vars(a)}

    cfg = NanoQuantConfig(bits=1.0, seed=a.seed, admm_type="nanoquant", admm_outer_iters=a.admm_outer_iters,
                          admm_inner_iters=5, admm_penalty_scheduler="linear",
                          tune_nonfact=False, tune_fact=True, tune_model=False)
    target, draft, tok = load_models(a.target, a.draft); force_unfused(draft); dev = target.device
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    ev = dict(block_size=BS, max_new_tokens=a.max_new_tokens, temperature=0.0, thr=0.01, strategy="cumulative_product")
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev)

    print("===== data =====", flush=True)
    nW = a.train_windows + a.heldout_windows
    hid, labels, mask, pe = build_windows(target, draft, tok, nW, a.seqlen, a.seed)
    tr = slice(0, a.train_windows); ho = slice(a.train_windows, nW)
    print(f"  windows {tuple(hid.shape)}  labels {tuple(labels.shape)}", flush=True)

    out["proxy_fp"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    print(f"  proxy FP drafter        : {out['proxy_fp']}", flush=True)

    print("===== ADMM (all 40, do_train=True) =====", flush=True)
    admm_all(draft, cfg, a.min_rank)
    out["proxy_admm"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    print(f"  proxy ADMM-only         : {out['proxy_admm']}", flush=True)

    # trainable set = NanoQuantLinear latents + scales only
    groups0 = get_param_group_config(draft, 1.0, 1.0, 1.0)
    params = [p for g in groups0 for p in g["params"]]
    init = [p.detach().clone() for p in params]
    n_lat = sum(p.numel() for g in groups0 for p in g["params"] if getattr(p, "optim_group", "") == "binary")
    print(f"  trainable: {len(params)} tensors, {sum(p.numel() for p in params)/1e6:.1f}M params "
          f"({n_lat/1e6:.1f}M binary latents)", flush=True)

    def restore():
        with torch.no_grad():
            for p, q in zip(params, init): p.copy_(q)

    def flip_frac():
        with torch.no_grad():
            f, t = 0, 0
            for p, q in zip(params, init):
                if getattr(p, "optim_group", "") == "binary":
                    f += (torch.sign(p) != torch.sign(q)).sum().item(); t += p.numel()
            return f / max(t, 1)

    print("===== sweep =====", flush=True)
    runs = {}
    for mode in modes:
        for lr in lrs:
            key = f"{mode}@{lr:g}"; restore()
            groups = get_param_group_config(draft, binary_lr=lr, scale_lr=lr, bias_lr=lr)
            t0 = time.time()
            hist = train(target, draft, params, groups, hid[tr], labels[tr], mask, pe, a.seqlen,
                         mode, a.epochs, a.batch, a.seed)
            r = {"mode": mode, "lr": lr, "train_loss": hist, "train_sec": time.time() - t0,
                 "flip_frac": flip_frac(),
                 **prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen),
                 "trained": [p.detach().clone().cpu() for p in params]}
            runs[key] = r
            print(f"  >>> {key:12s} proxy_tau={r['prefill_tau']:.3f}±{r['prefill_tau_sem']:.3f} "
                  f"surr={r['surrogate']:.3f} flips={100*r['flip_frac']:.2f}%  ({r['train_sec']:.0f}s)", flush=True)
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "trained"} for k, v in runs.items()},
                      open(a.out + ".sweep.json", "w"), indent=2)

    # pick configs for generation tau
    chosen = set()
    for mode in modes:
        cands = [k for k in runs if runs[k]["mode"] == mode]
        for spec in a.gen_eval.split(","):
            if spec == "best":
                chosen.add(max(cands, key=lambda k: runs[k]["prefill_tau"]))
            else:
                k = f"{mode}@{float(spec):g}"
                if k in runs: chosen.add(k)
    print(f"===== generation tau for {sorted(chosen)} =====", flush=True)
    out["runs"] = {}
    for key in sorted(chosen):
        with torch.no_grad():
            for p, q in zip(params, runs[key]["trained"]): p.copy_(q.to(p.device))
        g = measure_tau(target, draft, tok, eval_prompts, tag=key, **ev)
        out["runs"][key] = {**{k: v for k, v in runs[key].items() if k != "trained"}, "gen": g}
        print(f"  >>> {key}: generation tau = {g['tau']:.3f} ± {g['tau_sem']:.3f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=2)
    out["sweep"] = {k: {kk: vv for kk, vv in v.items() if kk != "trained"} for k, v in runs.items()}
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
