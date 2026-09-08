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
from run_onpolicy import chat_ids, onpolicy_windows
from run_all import install_calibration, collect_i_norm
nq = load_admm_nq()

R2 = {"q_proj": 256, "k_proj": 96, "v_proj": 96, "o_proj": 256}

# ---------------------------------------------------------------- two-path forward
_orig_forward = NanoQuantLinear.forward

def _forward_residual(self, x):
    y = _orig_forward(self, x)
    if hasattr(self, "U2_latent"):
        y = y + self._compute_forward(x, self.V2_latent, self.U2_latent, self.scale2_pre, None, self.scale2_post)
    return y

NanoQuantLinear.forward = _forward_residual


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


def admm_all_residual(draft, stats, cfg, min_rank):
    dev = next(draft.parameters()).device; e1, e2 = [], []
    for i, dl in enumerate(draft.layers):
        for name in PROJ_ORDER:
            lin = find_layers(dl)[name]; pname = name.split(".")[1]
            r = rank_for(lin.in_features, lin.out_features, cfg["bits"]); assert r > min_rank
            i_norm = stats[(i, "o" if pname == "o_proj" else "qkv")].to(dev).float()
            lin.register_buffer("i_norm", i_norm, persistent=False)
            lin.register_buffer("o_norm", torch.ones(lin.out_features, device=dev), persistent=False)
            W = lin.weight.data.clone()
            nano, _ = factorize_and_replace(dl, name, r, cfg)
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
    ap.add_argument("--mode", default="ce")
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--cache", type=str, default="")
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
        assert hid.shape[0] == nW and hid.shape[1] == a.seqlen + BS
    else:
        print("===== i_norm calibration =====", flush=True)
        from datasets import load_dataset
        train_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
        install_calibration(draft)
        cal = [torch.tensor([chat_ids(tok, train_ds[i]["question"])], device=dev) for i in range(a.inorm_prompts)]
        stats = collect_i_norm(draft, target, tok, cal, BS, 128, 0.01, "cumulative_product", 0.4)
        print("===== on-policy windows =====", flush=True)
        qs = [train_ds[i]["question"] for i in range(a.inorm_prompts, a.inorm_prompts + a.gen_prompts)]
        hid, labels, mask, pe = onpolicy_windows(target, draft, tok, qs, nW, a.seqlen)
        if a.cache:
            torch.save({"stats": {k: v.cpu() for k, v in stats.items()}, "hid": hid.cpu(), "labels": labels.cpu(),
                        "mask": mask.cpu(), "pe": tuple(t.cpu() for t in pe)}, a.cache)
    tr = slice(0, a.train_windows); ho = slice(a.train_windows, nW)
    out["proxy_fp"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)

    print("===== ADMM base + ADMM residual base =====", flush=True)
    out["rel_err_base"], out["rel_err_residual"] = admm_all_residual(draft, stats, cfg, a.min_rank)
    out["bpw"], out["MiB"] = bpw_report(draft)
    print(f"  mean weight rel_err {out['rel_err_base']:.4f} -> {out['rel_err_residual']:.4f};  {out['bpw']:.3f} bpw, {out['MiB']:.1f} MiB", flush=True)
    out["proxy_admm"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    out["admm_gen"] = measure_tau(target, draft, tok, eval_prompts, tag="admm+res", **ev)
    out["admm_dev"] = measure_tau(target, draft, tok, dev_prompts, tag="admm+res-dev", **ev)
    print(f"  ADMM+residual only: gen tau {out['admm_gen']['tau']:.3f}  dev {out['admm_dev']['tau']:.3f}", flush=True)
    dump()

    groups = get_param_group_config(draft, binary_lr=a.latent_lr, scale_lr=a.scale_lr, bias_lr=a.scale_lr)
    params = [p for g in groups for p in g["params"]]
    init = [p.detach().clone() for p in params]
    is_bin = [getattr(p, "optim_group", "") == "binary" for p in params]
    print(f"  trainable: {sum(p.numel() for p in params)/1e6:.1f}M params in {len(params)} tensors", flush=True)

    print(f"===== tune ({a.mode}, latent {a.latent_lr:g}, scale {a.scale_lr:g}) =====", flush=True)
    t0 = time.time()
    hist = train(target, draft, params, groups, hid[tr], labels[tr], mask, pe, a.seqlen, a.mode, a.epochs, a.batch, a.seed)
    with torch.no_grad():
        flips = sum((torch.sign(p) != torch.sign(q)).sum().item() for p, q, b in zip(params, init, is_bin) if b) / \
                sum(p.numel() for p, b in zip(params, is_bin) if b)
    r = {"train_loss": hist, "train_sec": time.time() - t0, "flip_frac": flips,
         "proxy": prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)}
    r["dev"] = measure_tau(target, draft, tok, dev_prompts, tag="tuned-dev", **ev)
    r["gen"] = measure_tau(target, draft, tok, eval_prompts, tag="tuned", **ev)
    out["tuned"] = r; dump()
    print(f"\n########## ADMM+residual {out['admm_gen']['tau']:.3f}  ->  tuned {r['gen']['tau']:.3f} ± {r['gen']['tau_sem']:.3f}   "
          f"({out['bpw']:.3f} bpw, flips {100*flips:.3f}%) ##########")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
