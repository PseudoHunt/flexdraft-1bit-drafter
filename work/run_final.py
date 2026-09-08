"""Mixture-of-rank / residual binary base on top of the round-2 pipeline.

  W ~= W1 + W2,   W1 = diag(s1_post) S_A1 S_B1 diag(s1_pre)   (rank r1, NanoQuant ADMM as before)
                  W2 = diag(s2_post) S_A2 S_B2 diag(s2_pre)   (rank r2, NanoQuant ADMM on the residual W - W1)
  bits = (r1 + r2)(in+out) + 32(in+out)   ->  ~1.13 bpw at r2 = 256 (q/o) / 96 (k/v)
Then target-teacher tuning (ce, latents 1e-4, scales 1e-5) of BOTH paths.  Post calibrator frozen.
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))

from fd_common import load_models, build_prompts, measure_tau, load_admm_nq
from nanoquant.modules.linear import NanoQuantLinear          # before core.* (circular import)
from nanoquant.core.compress_block import factorize_and_replace, get_param_group_config
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils.utils import find_layers, set_seed
from run_target_teacher import BS, prefill_tau, train, force_unfused, rank_for, PROJ_ORDER
from run_onpolicy import chat_ids
import run_onpolicy as _rop
import torch.nn.functional as F


@torch.no_grad()
def onpolicy_windows_toks(target, draft, tok, questions, n_windows, seqlen, gen_bs=64, max_new=320):
    """run_onpolicy.onpolicy_windows, plus the window token ids (needed for soft labels)."""
    dev = target.device
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = []
    for s in range(0, len(questions), gen_bs):
        batch = [chat_ids(tok, q) for q in questions[s:s + gen_bs]]
        L = max(len(b) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long); am = torch.zeros_like(ids)
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
    labels = toks[:, seqlen:seqlen + BS].to(dev)
    dtype = target.model.embed_tokens.weight.dtype; split = draft.target_layer_ids[0]
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
            h = lyr(hidden_states=h, attention_mask=None, position_embeddings=pe, past_key_value=cache, use_cache=True)
        hid[j] = torch.cat([h, mask_emb], dim=1)[0]
    min_val = torch.finfo(dtype).min
    m = torch.zeros(1, 1, Q, Q, dtype=dtype, device=dev)
    m[:, :, :seqlen, :seqlen] = torch.triu(torch.full((seqlen, seqlen), min_val, dtype=dtype, device=dev), 1)
    m[:, :, :seqlen, seqlen:] = min_val
    pe = target.model.rotary_emb(torch.empty(1, 1, dtype=dtype, device=dev), torch.arange(Q, device=dev).unsqueeze(0))
    return hid, labels, m, pe, toks


@torch.no_grad()
def build_soft_labels(target, toks, seqlen, topk, bs=16):
    """Target's top-k next-token logits at the 15 positions mask tokens 1..15 must match:
    logits index n+k-1 predicts token n+k (teacher-forced on the target's own greedy continuation)."""
    dev = target.device; N = toks.shape[0]
    vals = torch.empty(N, BS - 1, topk, dtype=torch.float16); idx = torch.empty(N, BS - 1, topk, dtype=torch.int32)
    for s in range(0, N, bs):
        ids = toks[s:s + bs].to(dev)
        lg = target(ids, logits_to_keep=BS).logits            # last 16 positions: indices n .. n+15
        assert lg.shape[1] == BS
        lg = lg[:, :BS - 1, :].float()                          # indices n..n+14 predict tokens n+1..n+15
        v, i = lg.topk(topk, dim=-1)
        vals[s:s + bs] = v.half().cpu(); idx[s:s + bs] = i.int().cpu()
    return vals, idx


def loss_kd(student_logits, tv, ti):
    logp = F.log_softmax(student_logits.float(), dim=-1).gather(-1, ti.long())      # (B,15,K)
    pt = F.softmax(tv.float(), dim=-1)                                              # renormalised top-k teacher
    return -(pt * logp).sum(-1).mean()


def train_es(target, draft, params, groups, hid, labels, soft, mask, pe, seqlen, mode, epochs, bsz, seed,
             ho_eval, log):
    """Round-2 training loop + per-epoch held-out proxy with best-epoch checkpointing."""
    from run_target_teacher import draft_logits, loss_fn
    set_seed(seed)
    from nanoquant.optimi import AdamW
    import math
    opt = AdamW(groups, weight_decay=0)
    n = hid.shape[0]; steps = epochs * math.ceil(n / bsz)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4 * groups[0]["lr"])
    hist, curve, best = [], [], {"proxy": -1, "epoch": 0, "params": None}
    with torch.enable_grad():
        for ep in range(epochs):
            perm = torch.randperm(n); tot = 0.0
            for s in range(0, n, bsz):
                idx = perm[s:s + bsz]
                lg = draft_logits(target, draft, hid[idx], labels[idx, 0], mask, pe, seqlen)
                if mode == "kd":
                    tv, ti = soft; loss = loss_kd(lg, tv[idx].to(lg.device), ti[idx].to(lg.device))
                else:
                    loss = loss_fn(lg, labels[idx, 1:], mode)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
                tot += loss.item() * len(idx)
            hist.append(tot / n)
            pr = ho_eval()
            curve.append(pr)
            log(f"    [{mode}] epoch {ep+1:02d}/{epochs} loss {tot/n:.4f}  held-out proxy {pr['ho']:.3f}  train proxy {pr['train']:.3f}")
            if pr["ho"] > best["proxy"]:
                best = {"proxy": pr["ho"], "epoch": ep + 1, "params": [p.detach().clone() for p in params]}
    return hist, curve, best
from run_all import install_calibration, collect_i_norm
nq = load_admm_nq()

R2 = {"q_proj": 256, "k_proj": 96, "v_proj": 96, "o_proj": 256}

# ---------------------------------------------------------------- two-path forward
_orig_forward = NanoQuantLinear.forward

def _forward_residual(self, x):
    rot = getattr(self, "rot", None)
    if rot is not None:
        x = x @ rot.to(x.dtype)
    y = _orig_forward(self, x)
    if hasattr(self, "U2_latent"):
        y = y + self._compute_forward(x, self.V2_latent, self.U2_latent, self.scale2_pre, None, self.scale2_post)
    return y

NanoQuantLinear.forward = _forward_residual


def randomized_hadamard(n, seed, device):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator().manual_seed(seed)
    sgn = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return (H * sgn.view(1, -1) / n ** 0.5).to(device)          # orthogonal, entries +-2^-6 exact in bf16


import run_all as _ra
_COV = {}
def _acc_with_cov(key, x):
    v = x.detach().float().reshape(-1, x.shape[-1])
    c = v.T @ v / v.shape[0]
    _COV[key] = c if key not in _COV else _COV[key] + c
    _ra._CAL["n"][key] += 0  # counted below via _CAL n from the original accumulator
    _orig_acc(key, x)
_orig_acc = _ra._acc


def rotated_i_norm(cov_by_key, n_by_key, R, shrinkage):
    """diag(R^T Sigma R) with the same shrinkage collect_i_norm applies."""
    out = {}
    for k, c in cov_by_key.items():
        d = torch.diag(R.T.float() @ (c / max(n_by_key[k], 1)) @ R.float()).clone()
        if 0.0 < shrinkage < 1.0:
            d.mul_(1.0 - shrinkage).add_(d.mean() * shrinkage)
        out[k] = d
    return out


@torch.no_grad()
def deployed_w1(lin):
    U = torch.sign(lin.U_latent.float()); V = torch.sign(lin.V_latent.float())
    return (U @ V) * lin.scale_pre.float() * lin.scale_post.float().view(-1, 1)


@torch.no_grad()
def add_residual_base(lin, W, i_norm, r2, cfg):
    """Second NanoQuant ADMM factorization of the residual; params registered as trainable path 2."""
    W1 = deployed_w1(lin)
    R = (W.float() - W1).to(W.dtype)
    out_f, in_f = W.shape
    set_seed(cfg["seed"])
    res = nq.factorize_admm_nanoquant(
        R, i_norm.float(), torch.ones(out_f, device=W.device), mid_rank=r2,
        outer_iters=cfg["admm_outer_iters"], inner_iters=cfg["admm_inner_iters"], reg=cfg["admm_reg"],
        is_transpose=(out_f < in_f), rho_scheduler=cfg["admm_penalty_scheduler"])
    def P(t, grp):
        p = nn.Parameter(t.to(lin.dtype).contiguous(), requires_grad=True); p.optim_group = grp; return p
    lin.U2_latent = P(res["A_latent"].mT, "binary")     # (out, r2)
    lin.V2_latent = P(res["B_latent"], "binary")        # (r2, in)
    lin.scale2_pre = P(res["scale_pre"], "scale")       # (1, in)
    lin.scale2_post = P(res["scale_post"], "scale")     # (1, out)
    lin.rank2 = r2
    W2 = (torch.sign(lin.U2_latent.float()) @ torch.sign(lin.V2_latent.float())) * lin.scale2_pre.float() * lin.scale2_post.float().view(-1, 1)
    n = W.float().square().sum()
    return ((W1 - W.float()).square().sum() / n).item(), ((W1 + W2 - W.float()).square().sum() / n).item()


def admm_all_residual(draft, stats, cfg, min_rank, R=None):
    dev = next(draft.parameters()).device; e1, e2 = [], []
    for i, dl in enumerate(draft.layers):
        for name in PROJ_ORDER:
            lin = find_layers(dl)[name]; pname = name.split(".")[1]
            r = rank_for(lin.in_features, lin.out_features, cfg["bits"]); assert r > min_rank
            i_norm = stats[(i, "o" if pname == "o_proj" else "qkv")].to(dev).float()
            lin.register_buffer("i_norm", i_norm, persistent=False)
            lin.register_buffer("o_norm", torch.ones(lin.out_features, device=dev), persistent=False)
            if R is not None:
                lin.weight.data.copy_((lin.weight.data.float() @ R.float()).to(lin.weight.dtype))   # W_R = W R
            W = lin.weight.data.clone()
            nano, _ = factorize_and_replace(dl, name, r, cfg)
            if R is not None:
                nano.register_buffer("rot", R.to(nano.dtype), persistent=False)
            a, b = add_residual_base(nano, W, i_norm, R2[pname], cfg)
            e1.append(a); e2.append(b)
        print(f"  layer {i}: rel_err base {sum(e1[-4:])/4:.4f} -> +residual {sum(e2[-4:])/4:.4f}", flush=True)
    return sum(e1) / len(e1), sum(e2) / len(e2)


def bpw_report(draft):
    bits = wts = 0
    for dl in draft.layers:
        for name in PROJ_ORDER:
            lin = getattr(dl.self_attn, name.split(".")[1]); i_, o_ = lin.in_features, lin.out_features
            bits += (lin.rank + lin.rank2) * (i_ + o_) + 16 * 2 * (i_ + o_); wts += i_ * o_
    return bits / wts, bits / 8 / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--eval-samples", type=int, default=40); ap.add_argument("--dev-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--train-windows", type=int, default=1024); ap.add_argument("--heldout-windows", type=int, default=128)
    ap.add_argument("--gen-prompts", type=int, default=1800); ap.add_argument("--seqlen", type=int, default=192)
    ap.add_argument("--inorm-prompts", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--scale-lr", type=float, default=1e-5); ap.add_argument("--latent-lr", type=float, default=1e-4)
    ap.add_argument("--mode", default="ce", choices=["ce", "acc", "kd"])
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--cache", type=str, default="")
    ap.add_argument("--tune-calibrator", action="store_true", help="also fine-tune anchor_bias_mlp in FP (not quantized)")
    ap.add_argument("--calib-lr", type=float, default=1e-5)
    ap.add_argument("--rotate", choices=["none", "hadamard"], default="none")
    ap.add_argument("--rot-seed", type=int, default=0)
    ap.add_argument("--skip-admm-eval", action="store_true")
    ap.add_argument("--save-params", type=str, default="")
    ap.add_argument("--kd-topk", type=int, default=256)
    ap.add_argument("--early-stop", action="store_true", help="track held-out proxy per epoch, also eval best epoch")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = {"args": vars(a), "r2": R2}
    def dump(): json.dump(out, open(a.out, "w"), indent=2)

    cfg = NanoQuantConfig(bits=1.0, seed=a.seed, admm_type="nanoquant", admm_outer_iters=a.admm_outer_iters,
                          admm_inner_iters=5, admm_penalty_scheduler="linear",
                          tune_nonfact=False, tune_fact=True, tune_model=False)
    target, draft, tok = load_models(a.target, a.draft); force_unfused(draft); dev = target.device
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    ev = dict(block_size=BS, max_new_tokens=a.max_new_tokens, temperature=0.0, thr=0.01, strategy="cumulative_product")
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev)
    dev_prompts = build_prompts(tok, "gsm8k", a.dev_samples, dev, split_offset=a.eval_samples)

    nW = a.train_windows + a.heldout_windows
    if a.cache and os.path.exists(a.cache):
        print(f"===== cached data {a.cache} =====", flush=True)
        c = torch.load(a.cache, weights_only=False)
        stats = {k: v.to(dev) for k, v in c["stats"].items()}
        hid, labels, mask, pe = c["hid"].to(dev), c["labels"].to(dev), c["mask"].to(dev), tuple(t.to(dev) for t in c["pe"])
        toks = c.get("toks", None)
        assert hid.shape[0] == nW and hid.shape[1] == a.seqlen + BS
        if a.mode == "kd" and toks is None:
            raise SystemExit("cache has no token ids; use a new --cache path so windows are regenerated with toks")
    else:
        print("===== i_norm calibration =====", flush=True)
        from datasets import load_dataset
        train_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
        install_calibration(draft)
        cal = [torch.tensor([chat_ids(tok, train_ds[i]["question"])], device=dev) for i in range(a.inorm_prompts)]
        stats = collect_i_norm(draft, target, tok, cal, BS, 128, 0.01, "cumulative_product", 0.4)
        print("===== on-policy windows =====", flush=True)
        qs = [train_ds[i]["question"] for i in range(a.inorm_prompts, a.inorm_prompts + a.gen_prompts)]
        hid, labels, mask, pe, toks = onpolicy_windows_toks(target, draft, tok, qs, nW, a.seqlen)
        if a.cache:
            torch.save({"stats": {k: v.cpu() for k, v in stats.items()}, "hid": hid.cpu(), "labels": labels.cpu(),
                        "mask": mask.cpu(), "pe": tuple(t.cpu() for t in pe), "toks": toks.cpu()}, a.cache)
    soft = None
    if a.mode == "kd":
        print(f"===== target top-{a.kd_topk} soft labels =====", flush=True)
        tv, ti = build_soft_labels(target, toks, a.seqlen, a.kd_topk)
        soft = (tv.to(dev), ti.to(dev))
        agree = (ti[:, :, 0].to(labels.device) == labels[:, 1:].cpu().int().to(labels.device)).float().mean().item()
        print(f"  soft labels {tuple(tv.shape)}; top-1 agrees with cached greedy labels {100*agree:.1f}%", flush=True)
    tr = slice(0, a.train_windows); ho = slice(a.train_windows, nW)
    out["proxy_fp"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)

    R = None
    if a.rotate == "hadamard":
        R = randomized_hadamard(target.config.hidden_size, a.rot_seed, dev)
        print("===== re-collecting input covariance for rotated i_norm (FP drafter) =====", flush=True)
        from datasets import load_dataset
        train_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
        install_calibration(draft); _ra._acc = _acc_with_cov
        cal = [torch.tensor([chat_ids(tok, train_ds[i]["question"])], device=dev) for i in range(a.inorm_prompts)]
        _ = collect_i_norm(draft, target, tok, cal, BS, 128, 0.01, "cumulative_product", 0.4)
        stats = rotated_i_norm(_COV, dict(_ra._CAL["n"]), R, 0.4)
        _ra._acc = _orig_acc; _COV.clear()
        print(f"  rotated i_norm ready ({len(stats)} tensors); R^T R - I max = "
              f"{(R.T @ R - torch.eye(R.shape[0], device=dev)).abs().max().item():.1e}", flush=True)
    out["rotate"] = a.rotate; out["tune_calibrator"] = a.tune_calibrator

    print("===== ADMM base + ADMM residual base =====", flush=True)
    out["rel_err_base"], out["rel_err_residual"] = admm_all_residual(draft, stats, cfg, a.min_rank, R=R)
    out["bpw"], out["MiB"] = bpw_report(draft)
    print(f"  mean weight rel_err {out['rel_err_base']:.4f} -> {out['rel_err_residual']:.4f};  {out['bpw']:.3f} bpw, {out['MiB']:.1f} MiB", flush=True)
    out["proxy_admm"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    out["proxy_admm_train"] = prefill_tau(target, draft, hid[:128], labels[:128], mask, pe, a.seqlen)
    if not a.skip_admm_eval:
        out["admm_gen"] = measure_tau(target, draft, tok, eval_prompts, tag="admm+res", **ev)
        out["admm_dev"] = measure_tau(target, draft, tok, dev_prompts, tag="admm+res-dev", **ev)
        print(f"  ADMM+residual only: gen tau {out['admm_gen']['tau']:.3f}  dev {out['admm_dev']['tau']:.3f}", flush=True)
    dump()

    groups = get_param_group_config(draft, binary_lr=a.latent_lr, scale_lr=a.scale_lr, bias_lr=a.scale_lr)
    if a.tune_calibrator:
        cal_params = list(draft.anchor_bias_mlp.parameters())
        for p in cal_params:
            p.requires_grad_(True)
        groups.append({"params": cal_params, "lr": a.calib_lr})
        print(f"  + tuning post-calibrator in FP: {sum(p.numel() for p in cal_params)/1e6:.1f}M params @ lr {a.calib_lr:g}", flush=True)
    params = [p for g in groups for p in g["params"]]
    init = [p.detach().clone() for p in params]
    is_bin = [getattr(p, "optim_group", "") == "binary" for p in params]
    print(f"  trainable: {sum(p.numel() for p in params)/1e6:.1f}M params in {len(params)} tensors", flush=True)

    print(f"===== tune ({a.mode}, latent {a.latent_lr:g}, scale {a.scale_lr:g}) =====", flush=True)
    t0 = time.time()
    def _ho_eval():
        return {"ho": prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)["prefill_tau"],
                "train": prefill_tau(target, draft, hid[:128], labels[:128], mask, pe, a.seqlen)["prefill_tau"]}
    soft_tr = None if soft is None else (soft[0][:a.train_windows], soft[1][:a.train_windows])
    hist, curve, best = train_es(target, draft, params, groups, hid[tr], labels[tr], soft_tr, mask, pe, a.seqlen,
                                 a.mode, a.epochs, a.batch, a.seed, _ho_eval, lambda m: print(m, flush=True))
    out["epoch_curve"] = curve; out["best_epoch"] = best["epoch"]
    with torch.no_grad():
        flips = sum((torch.sign(p) != torch.sign(q)).sum().item() for p, q, b in zip(params, init, is_bin) if b) / \
                sum(p.numel() for p, b in zip(params, is_bin) if b)
    r = {"train_loss": hist, "train_sec": time.time() - t0, "flip_frac": flips,
         "proxy": prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen),
         "proxy_train": prefill_tau(target, draft, hid[:128], labels[:128], mask, pe, a.seqlen)}
    if a.save_params:
        sd = {n: p.detach().cpu() for n, p in draft.named_parameters() if p.requires_grad}
        sd["__rot__"] = None if R is None else R.cpu(); sd["__r2__"] = R2
        torch.save(sd, a.save_params); print(f"  saved {len(sd)} tensors -> {a.save_params}", flush=True)
    r["dev"] = measure_tau(target, draft, tok, dev_prompts, tag="tuned-dev", **ev)
    r["gen"] = measure_tau(target, draft, tok, eval_prompts, tag="tuned", **ev)
    out["tuned"] = r; dump()
    if a.early_stop and best["epoch"] != a.epochs and best["params"] is not None:
        with torch.no_grad():
            for p, q in zip(params, best["params"]): p.copy_(q)
        rb = {"epoch": best["epoch"], "proxy_ho": best["proxy"]}
        rb["dev"] = measure_tau(target, draft, tok, dev_prompts, tag=f"best-ep{best['epoch']}-dev", **ev)
        rb["gen"] = measure_tau(target, draft, tok, eval_prompts, tag=f"best-ep{best['epoch']}", **ev)
        out["tuned_best_epoch"] = rb; dump()
        print(f"  early-stopped (epoch {best['epoch']}): gen tau {rb['gen']['tau']:.3f} ± {rb['gen']['tau_sem']:.3f}", flush=True)
        if a.save_params:
            sd = {n: p.detach().cpu() for n, p in draft.named_parameters() if p.requires_grad}
            torch.save(sd, a.save_params.replace(".pt", f"_best_ep{best['epoch']}.pt"))
    print(f"\n########## [{a.rotate}{'+calib' if a.tune_calibrator else ''}] ADMM+residual {out.get('admm_gen',{}).get('tau',float('nan')):.3f}  ->  tuned {r['gen']['tau']:.3f} ± {r['gen']['tau_sem']:.3f}   "
          f"({out['bpw']:.3f} bpw, flips {100*flips:.3f}%) ##########")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
