"""Round 2 of target-as-teacher tuning for the 1-bit FlexDraft drafter.

Changes vs run_target_teacher.py (all aimed at the data-starved / scale-only regime found there):
  * ON-POLICY calibration windows: context = the target's own greedy continuation of a GSM8K-train
    prompt (what the drafter sees at deployment, since FlexDraft is lossless), 8x more of them.
    Window = first `seqlen` tokens of prompt+continuation; labels = the next 16 tokens, which are
    by construction the target's greedy continuation of that exact prefix (no truncation tricks).
  * Calibrated i_norm ADMM init (activation second moments from real drafting), not ones.
  * DECOUPLED learning rates: scales fixed at 1e-5, binary latents swept -> does flipping bits
    add anything beyond re-scaling?
  * Selection on generation-tau over 8 dev prompts disjoint from the 40 eval prompts.
Post calibrator (anchor_bias_mlp), mask_embedding, q/k norms, target: frozen.
"""
import argparse, json, math, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))

from fd_common import load_models, build_prompts, measure_tau
from nanoquant.modules.linear import NanoQuantLinear          # before core.* (circular import)
from nanoquant.core.compress_block import factorize_and_replace, get_param_group_config
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils.utils import find_layers
from run_target_teacher import BS, draft_logits, prefill_tau, train, force_unfused, rank_for, PROJ_ORDER
from run_all import install_calibration, collect_i_norm
from flexdraft.utils import _MATH_FINAL_ANSWER_INSTRUCTION


def chat_ids(tok, question):
    m = [{"role": "user", "content": question + _MATH_FINAL_ANSWER_INSTRUCTION}]
    return tok.encode(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False))


@torch.no_grad()
def onpolicy_windows(target, draft, tok, questions, n_windows, seqlen, gen_bs=64, max_new=320):
    """Greedy-generate with the target, keep sequences with >= seqlen+16 tokens (up to EOS)."""
    dev = target.device
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = []
    for s in range(0, len(questions), gen_bs):
        batch = [chat_ids(tok, q) for q in questions[s:s + gen_bs]]
        L = max(len(b) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, b in enumerate(batch):
            ids[i, L - len(b):] = torch.tensor(b); am[i, L - len(b):] = 1
        out = target.generate(ids.to(dev), attention_mask=am.to(dev), do_sample=False,
                              max_new_tokens=max_new, pad_token_id=pad_id)
        for i, b in enumerate(batch):
            gen = out[i, L:].tolist()
            if tok.eos_token_id in gen:
                gen = gen[:gen.index(tok.eos_token_id) + 1]
            full = b + gen
            if len(full) >= seqlen + BS:
                seqs.append(full[:seqlen + BS])
        print(f"    generated {min(s+gen_bs, len(questions))}/{len(questions)} prompts -> {len(seqs)} usable windows", flush=True)
        if len(seqs) >= n_windows:
            break
    assert len(seqs) >= n_windows, f"only {len(seqs)} windows"
    toks = torch.tensor(seqs[:n_windows], dtype=torch.long)
    labels = toks[:, seqlen:seqlen + BS].to(dev)          # y*_n (anchor) .. y*_{n+15}

    dtype = target.model.embed_tokens.weight.dtype
    split = draft.target_layer_ids[0]
    from transformers import DynamicCache
    Q = seqlen + BS
    mask_emb = draft.mask_embedding.to(dtype).view(1, 1, -1).expand(1, BS, -1)
    hid = torch.empty(n_windows, Q, target.config.hidden_size, dtype=dtype, device=dev)
    for j in range(n_windows):
        ids = toks[j:j + 1, :seqlen].to(dev)
        h = target.model.embed_tokens(ids)
        pe = target.model.rotary_emb(h, torch.arange(seqlen, device=dev).unsqueeze(0))
        cache = DynamicCache()
        for lyr in target.model.layers[:split]:
            h = lyr(hidden_states=h, attention_mask=None, position_embeddings=pe,
                    past_key_value=cache, use_cache=True)
        hid[j] = torch.cat([h, mask_emb], dim=1)[0]
    min_val = torch.finfo(dtype).min
    m = torch.zeros(1, 1, Q, Q, dtype=dtype, device=dev)
    m[:, :, :seqlen, :seqlen] = torch.triu(torch.full((seqlen, seqlen), min_val, dtype=dtype, device=dev), 1)
    m[:, :, :seqlen, seqlen:] = min_val
    pe = target.model.rotary_emb(torch.empty(1, 1, dtype=dtype, device=dev),
                                 torch.arange(Q, device=dev).unsqueeze(0))
    return hid, labels, m, pe


def admm_all_calibrated(draft, stats, cfg, min_rank):
    dev = next(draft.parameters()).device
    for i, dl in enumerate(draft.layers):
        for name in PROJ_ORDER:
            lin = find_layers(dl)[name]
            r = rank_for(lin.in_features, lin.out_features, cfg["bits"]); assert r > min_rank
            key = (i, "o" if name.endswith("o_proj") else "qkv")
            lin.register_buffer("i_norm", stats[key].to(dev).float(), persistent=False)
            lin.register_buffer("o_norm", torch.ones(lin.out_features, device=dev), persistent=False)
            factorize_and_replace(dl, name, r, cfg)
        print(f"  ADMM layer {i} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--eval-samples", type=int, default=40); ap.add_argument("--dev-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--train-windows", type=int, default=1024); ap.add_argument("--heldout-windows", type=int, default=128)
    ap.add_argument("--gen-prompts", type=int, default=1800); ap.add_argument("--seqlen", type=int, default=192)
    ap.add_argument("--inorm-prompts", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--scale-lr", type=float, default=1e-5)
    ap.add_argument("--latent-lrs", type=str, default="1e-5,1e-4,3e-4,1e-3")
    ap.add_argument("--modes", type=str, default="acc,ce")
    ap.add_argument("--pairs", type=str, default="", help="explicit scale:latent lr pairs, overrides --scale-lr/--latent-lrs")
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    modes = a.modes.split(",")
    pairs = ([tuple(float(v) for v in pr.split(":")) for pr in a.pairs.split(",")] if a.pairs
             else [(a.scale_lr, float(x)) for x in a.latent_lrs.split(",")])
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = {"args": vars(a), "runs": {}}
    def dump(): json.dump(out, open(a.out, "w"), indent=2)

    cfg = NanoQuantConfig(bits=1.0, seed=a.seed, admm_type="nanoquant", admm_outer_iters=a.admm_outer_iters,
                          admm_inner_iters=5, admm_penalty_scheduler="linear",
                          tune_nonfact=False, tune_fact=True, tune_model=False)
    target, draft, tok = load_models(a.target, a.draft); force_unfused(draft); dev = target.device
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    ev = dict(block_size=BS, max_new_tokens=a.max_new_tokens, temperature=0.0, thr=0.01, strategy="cumulative_product")
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev)                       # test[0:40]
    dev_prompts = build_prompts(tok, "gsm8k", a.dev_samples, dev, split_offset=a.eval_samples)  # test[40:48]

    print("===== i_norm calibration (real drafting, FP drafter) =====", flush=True)
    from datasets import load_dataset
    train_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
    install_calibration(draft)
    cal = [torch.tensor([chat_ids(tok, train_ds[i]["question"])], device=dev) for i in range(a.inorm_prompts)]
    stats = collect_i_norm(draft, target, tok, cal, BS, 128, 0.01, "cumulative_product", 0.4)

    print("===== on-policy windows =====", flush=True)
    t0 = time.time()
    qs = [train_ds[i]["question"] for i in range(a.inorm_prompts, a.inorm_prompts + a.gen_prompts)]
    nW = a.train_windows + a.heldout_windows
    hid, labels, mask, pe = onpolicy_windows(target, draft, tok, qs, nW, a.seqlen)
    tr = slice(0, a.train_windows); ho = slice(a.train_windows, nW)
    print(f"  windows {tuple(hid.shape)} in {time.time()-t0:.0f}s", flush=True)
    out["proxy_fp"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    print(f"  proxy FP: {out['proxy_fp']}", flush=True)

    print("===== ADMM, calibrated i_norm (do_train=True) =====", flush=True)
    admm_all_calibrated(draft, stats, cfg, a.min_rank)
    out["proxy_admm"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    out["admm_gen"] = measure_tau(target, draft, tok, eval_prompts, tag="admm", **ev)
    out["admm_dev"] = measure_tau(target, draft, tok, dev_prompts, tag="admm-dev", **ev)
    print(f"  ADMM-only: gen tau {out['admm_gen']['tau']:.3f}  dev {out['admm_dev']['tau']:.3f}  proxy {out['proxy_admm']}", flush=True)
    dump()

    groups0 = get_param_group_config(draft, 1.0, 1.0, 1.0)
    params = [p for g in groups0 for p in g["params"]]
    init = [p.detach().clone() for p in params]
    is_bin = [getattr(p, "optim_group", "") == "binary" for p in params]

    def restore():
        with torch.no_grad():
            for p, q in zip(params, init): p.copy_(q)

    def flip_frac():
        with torch.no_grad():
            f = sum((torch.sign(p) != torch.sign(q)).sum().item() for p, q, b in zip(params, init, is_bin) if b)
            t = sum(p.numel() for p, b in zip(params, is_bin) if b)
            return f / t

    print("===== sweep =====", flush=True)
    for mode in modes:
        for slr, llr in pairs:
            key = f"{mode}@lat{llr:g}_sc{slr:g}"; restore()
            groups = get_param_group_config(draft, binary_lr=llr, scale_lr=slr, bias_lr=slr)
            t0 = time.time()
            hist = train(target, draft, params, groups, hid[tr], labels[tr], mask, pe, a.seqlen,
                         mode, a.epochs, a.batch, a.seed)
            r = {"mode": mode, "latent_lr": llr, "scale_lr": slr, "train_loss": hist,
                 "train_sec": time.time() - t0, "flip_frac": flip_frac(),
                 "proxy": prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)}
            r["dev"] = measure_tau(target, draft, tok, dev_prompts, tag=key + "-dev", **ev)
            r["gen"] = measure_tau(target, draft, tok, eval_prompts, tag=key, **ev)
            out["runs"][key] = r; dump()
            print(f"  >>> {key:24s} dev={r['dev']['tau']:.3f}  GEN={r['gen']['tau']:.3f}±{r['gen']['tau_sem']:.3f}  "
                  f"proxy={r['proxy']['prefill_tau']:.3f}  flips={100*r['flip_frac']:.3f}%  ({r['train_sec']:.0f}s)", flush=True)

    best = max(out["runs"], key=lambda k: out["runs"][k]["dev"]["tau"])
    out["dev_selected"] = best
    print(f"\n########## dev-selected: {best}  gen tau = {out['runs'][best]['gen']['tau']:.3f}  "
          f"(ADMM-only {out['admm_gen']['tau']:.3f}) ##########")
    dump(); print("wrote", a.out)


if __name__ == "__main__":
    main()
