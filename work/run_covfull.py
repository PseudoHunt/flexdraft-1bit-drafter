"""Full pipeline on the covariance objective: cov-ADMM base -> cov-ADMM residual base -> target-teacher tuning.
Controls: diag pipeline res.json (5.950 -> 6.360); cov base-only covgate2.json (6.328)."""
import argparse, json, os, sys, time
from argparse import Namespace
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))
from fd_common import load_models, build_prompts, measure_tau
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.core.compress_block import get_param_group_config
from run_target_teacher import BS, prefill_tau, train, force_unfused, rank_for
from run_residual import _forward_residual, deployed_w1          # installs the two-path forward
from admm_cov import factorize_admm_cov, CovSide
from nanoquant_seed import set_seed
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]; R2 = {"q_proj": 256, "k_proj": 96, "v_proj": 96, "o_proj": 256}


def load_sides(path, dev):
    cs = torch.load(path); sides = {}
    for k, d in cs.items():
        o = CovSide.__new__(CovSide); o.i_norm = d["i_norm"].to(dev); o.lam = d["lam"].to(dev); o.Q = d["Q"].to(dev)
        o.Lr = o.Q * o.lam.sqrt().view(1, -1); o.is_identity = False; sides[k] = o
    return sides


@torch.no_grad()
def cov_quantize(draft, sides, iters, seed, residual=True):
    dev = next(draft.parameters()).device; e1, e2 = [], []
    for i, dl in enumerate(draft.layers):
        for p in PROJ:
            lin = getattr(dl.self_attn, p); W = lin.weight.data.clone(); out_f, in_f = W.shape
            r = rank_for(in_f, out_f, 1.0); side = sides[(i, "o" if p == "o_proj" else "qkv")]; o = torch.ones(out_f, device=dev)
            set_seed(seed); res = factorize_admm_cov(W, side, o, mid_rank=r, outer_iters=iters, rho_scheduler="linear")
            lin.__class__ = NanoQuantLinear; lin.__quant_convert__(do_train=True, rank=r, factor_results=Namespace(**res))
            W1 = deployed_w1(lin); n = W.float().square().sum(); e1.append(((W1 - W.float()).square().sum() / n).item())
            if residual:
                set_seed(seed); r2 = factorize_admm_cov((W.float() - W1).to(W.dtype), side, o, mid_rank=R2[p], outer_iters=iters, rho_scheduler="linear")
                def P(t, grp):
                    q = nn.Parameter(t.to(lin.dtype).contiguous(), requires_grad=True); q.optim_group = grp; return q
                lin.U2_latent = P(r2["A_latent"].mT, "binary"); lin.V2_latent = P(r2["B_latent"], "binary")
                lin.scale2_pre = P(r2["scale_pre"], "scale"); lin.scale2_post = P(r2["scale_post"], "scale"); lin.rank2 = R2[p]
                W2 = (torch.sign(lin.U2_latent.float()) @ torch.sign(lin.V2_latent.float())) * lin.scale2_pre.float() * lin.scale2_post.float().view(-1, 1)
                e2.append(((W1 + W2 - W.float()).square().sum() / n).item())
        print(f"  layer {i}: weight rel_err base {sum(e1[-4:])/4:.4f}" + (f" -> +residual {sum(e2[-4:])/4:.4f}" if residual else ""), flush=True)
    return sum(e1) / 40, (sum(e2) / 40 if residual else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--load-cov", required=True)
    ap.add_argument("--eval-samples", type=int, default=40); ap.add_argument("--dev-samples", type=int, default=8); ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--train-windows", type=int, default=1024); ap.add_argument("--heldout-windows", type=int, default=128); ap.add_argument("--seqlen", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--scale-lr", type=float, default=1e-5); ap.add_argument("--latent-lr", type=float, default=1e-4); ap.add_argument("--mode", default="ce")
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--save-params", type=str, default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = {"args": vars(a), "r2": R2}
    def dump(): json.dump(out, open(a.out, "w"), indent=2)
    target, draft, tok = load_models(a.target, a.draft); force_unfused(draft); dev = target.device
    for p in list(target.parameters()) + list(draft.parameters()): p.requires_grad_(False)
    ev = dict(block_size=BS, max_new_tokens=a.max_new_tokens, temperature=0.0, thr=0.01, strategy="cumulative_product")
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev); dev_prompts = build_prompts(tok, "gsm8k", a.dev_samples, dev, split_offset=a.eval_samples)
    c = torch.load(a.cache, weights_only=False)
    hid, labels, mask, pe = c["hid"].to(dev), c["labels"].to(dev), c["mask"].to(dev), tuple(t.to(dev) for t in c["pe"])
    nW = a.train_windows + a.heldout_windows; assert hid.shape[0] == nW; tr = slice(0, a.train_windows); ho = slice(a.train_windows, nW)
    sides = load_sides(a.load_cov, dev)

    print("===== cov-ADMM base" + ("" if a.no_residual else " + cov-ADMM residual") + " =====", flush=True); t0 = time.time()
    out["rel_err_base"], out["rel_err_residual"] = cov_quantize(draft, sides, a.admm_outer_iters, a.seed, residual=not a.no_residual)
    out["admm_sec"] = time.time() - t0
    out["proxy_admm"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    out["admm_gen"] = measure_tau(target, draft, tok, eval_prompts, tag="cov-admm", **ev); out["admm_dev"] = measure_tau(target, draft, tok, dev_prompts, tag="cov-admm-dev", **ev)
    print(f"  ADMM-only: gen {out['admm_gen']['tau']:.3f} ± {out['admm_gen']['tau_sem']:.3f}  dev {out['admm_dev']['tau']:.3f}  proxy {out['proxy_admm']['prefill_tau']:.2f}  ({out['admm_sec']:.0f}s)", flush=True); dump()

    groups = get_param_group_config(draft, binary_lr=a.latent_lr, scale_lr=a.scale_lr, bias_lr=a.scale_lr)
    params = [p for g in groups for p in g["params"]]; init = [p.detach().clone() for p in params]; is_bin = [getattr(p, "optim_group", "") == "binary" for p in params]
    print(f"===== tune ({a.mode}, lat {a.latent_lr:g}, sc {a.scale_lr:g}, {sum(p.numel() for p in params)/1e6:.0f}M params) =====", flush=True); t0 = time.time()
    hist = train(target, draft, params, groups, hid[tr], labels[tr], mask, pe, a.seqlen, a.mode, a.epochs, a.batch, a.seed)
    with torch.no_grad():
        flips = sum((torch.sign(p) != torch.sign(q)).sum().item() for p, q, b in zip(params, init, is_bin) if b) / sum(p.numel() for p, b in zip(params, is_bin) if b)
    r = {"train_loss": hist, "train_sec": time.time() - t0, "flip_frac": flips,
         "proxy": prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen), "proxy_train": prefill_tau(target, draft, hid[:128], labels[:128], mask, pe, a.seqlen)}
    r["dev"] = measure_tau(target, draft, tok, dev_prompts, tag="tuned-dev", **ev); r["gen"] = measure_tau(target, draft, tok, eval_prompts, tag="tuned", **ev)
    out["tuned"] = r; dump()
    if a.save_params:
        torch.save({n: p.detach().cpu() for n, p in draft.named_parameters() if p.requires_grad}, a.save_params)
    print(f"\n########## cov pipeline: ADMM {out['admm_gen']['tau']:.3f} -> tuned {r['gen']['tau']:.3f} ± {r['gen']['tau_sem']:.3f}  ({100*r['gen']['tau']/7.312:.1f}% of FP; flips {100*flips:.3f}%) ##########")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
