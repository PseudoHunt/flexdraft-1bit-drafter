"""Option B gate: real input covariance -> spectrum report -> cov-ADMM (base only) vs NanoQuant diag-ADMM
on held-out OUTPUT error, then ADMM-only tau (no residual, no tuning; control = 5.530 in op.json)."""
import argparse, json, os, sys, time
from collections import defaultdict
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))
from fd_common import load_models, build_prompts, measure_tau, load_admm_nq
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.modules.quant_config import NanoQuantConfig
from run_target_teacher import BS, prefill_tau, force_unfused, rank_for
from run_onpolicy import chat_ids
from run_onorm import capture_inputs
import run_all as _ra
from run_final import _acc_with_cov, _COV, _orig_acc
from admm_cov import factorize_admm_cov, CovSide, shrink_cov
from nanoquant_seed import set_seed
nq = load_admm_nq()
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--eval-samples", type=int, default=40); ap.add_argument("--dev-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256); ap.add_argument("--train-windows", type=int, default=1024)
    ap.add_argument("--heldout-windows", type=int, default=128); ap.add_argument("--seqlen", type=int, default=192)
    ap.add_argument("--inorm-prompts", type=int, default=128); ap.add_argument("--shrinkage", type=float, default=0.4)
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--save-cov", type=str, default=""); ap.add_argument("--load-cov", type=str, default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = {"args": vars(a)}
    def dump(): json.dump(out, open(a.out, "w"), indent=2)
    cfg = NanoQuantConfig(bits=1.0, seed=a.seed, admm_type="nanoquant", admm_outer_iters=a.admm_outer_iters, admm_inner_iters=5,
                          admm_penalty_scheduler="linear", tune_nonfact=False, tune_fact=False, tune_model=False)
    target, draft, tok = load_models(a.target, a.draft); force_unfused(draft); dev = target.device
    for p in list(target.parameters()) + list(draft.parameters()): p.requires_grad_(False)
    ev = dict(block_size=BS, max_new_tokens=a.max_new_tokens, temperature=0.0, thr=0.01, strategy="cumulative_product")
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev); dev_prompts = build_prompts(tok, "gsm8k", a.dev_samples, dev, split_offset=a.eval_samples)
    c = torch.load(a.cache, weights_only=False)
    stats = {k: v.to(dev) for k, v in c["stats"].items()}
    hid, labels, mask, pe = c["hid"].to(dev), c["labels"].to(dev), c["mask"].to(dev), tuple(t.to(dev) for t in c["pe"])
    nW = a.train_windows + a.heldout_windows; ho = slice(a.train_windows, nW)

    # ---- 1) real input covariance from FP drafting
    if a.load_cov:
        print(f"===== loading covariance sides from {a.load_cov} =====", flush=True)
        cs = torch.load(a.load_cov); sides = {}
        for k, d in cs.items():
            o = CovSide.__new__(CovSide); o.i_norm = d["i_norm"].to(dev); o.lam = d["lam"].to(dev); o.Q = d["Q"].to(dev)
            o.Lr = o.Q * o.lam.sqrt().view(1, -1); o.is_identity = False; sides[k] = o
        out["spectrum"] = "see covgate.json"
    else:
      print("===== input covariance (FP drafter, real drafting) =====", flush=True)
    if not a.load_cov:
      from datasets import load_dataset
      ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
      _ra.install_calibration(draft); _ra._acc = _acc_with_cov
      cal = [torch.tensor([chat_ids(tok, ds[i]["question"])], device=dev) for i in range(a.inorm_prompts)]
      t0 = time.time(); _ = _ra.collect_i_norm(draft, target, tok, cal, BS, 128, 0.01, "cumulative_product", a.shrinkage)
      n_calls = dict(_ra._CAL["n"]); _ra._acc = _orig_acc
      sides, spec = {}, {}
      for key, Ssum in _COV.items():
          Sig = shrink_cov(Ssum / max(n_calls[key], 1), a.shrinkage); side = CovSide(Sig); sides[key] = side
          lam = side.lam; C = (side.Q * lam.view(1, -1)) @ side.Q.mT; n = C.shape[0]
          off = (C - torch.eye(n, device=dev)).abs(); 
          lam_raw = torch.linalg.eigvalsh((Ssum / n_calls[key]).double() / (Ssum / n_calls[key]).double().diagonal().sqrt().view(-1, 1) / (Ssum / n_calls[key]).double().diagonal().sqrt().view(1, -1)).clamp_min(1e-12)
          spec[f"L{key[0]}.{key[1]}"] = {"cond_shrunk": (lam.max() / lam.min()).item(), "cond_raw": (lam_raw.max() / lam_raw.min()).item(),
                                          "eff_rank": (lam.sum() ** 2 / (lam ** 2).sum()).item(), "mean_abs_offdiag_corr": (off.sum() / (n * (n - 1))).item(),
                                          "top1_eig_share": (lam.max() / lam.sum()).item(),
                                          "diag_vs_cached_inorm_maxrel": ((side.i_norm - stats[key]).abs() / stats[key]).max().item()}
      _COV.clear()
      out["spectrum"] = spec
      agg = lambda f: {k: sum(spec[s][f] for s in spec if s.endswith(k)) / 10 for k in ["qkv", "o"]}
      print(f"  collected in {time.time()-t0:.0f}s. mean over layers -- cond(shrunk C): {agg('cond_shrunk')}  cond(raw C): {agg('cond_raw')}  "
            f"eff.rank/4096: { {k: v/4096 for k, v in agg('eff_rank').items()} }  mean|offdiag corr|: {agg('mean_abs_offdiag_corr')}  top-eig share: {agg('top1_eig_share')}  "
            f"diag vs cached i_norm max rel: {max(spec[s]['diag_vs_cached_inorm_maxrel'] for s in spec):.2e}", flush=True); dump()
      if a.save_cov:
          torch.save({k: {"i_norm": v.i_norm.cpu(), "lam": v.lam.cpu(), "Q": v.Q.cpu()} for k, v in sides.items()}, a.save_cov)

    # ---- 2) held-out inputs, then both factorizations (function level) with paired output errors
    print("===== base-only ADMM: NanoQuant diag vs cov, held-out output error =====", flush=True)
    X = capture_inputs(draft, target, hid[ho], labels[ho], mask, pe, a.seqlen)
    per = {"diag": defaultdict(list), "cov": defaultdict(list)}; results_cov = {}; t_d = t_c = 0.0
    for i, dl in enumerate(draft.layers):
        for p in PROJ:
            lin = getattr(dl.self_attn, p); W = lin.weight.data; out_f, in_f = W.shape; r = rank_for(in_f, out_f, 1.0); T = out_f < in_f
            side = sides[(i, "o" if p == "o_proj" else "qkv")]; o = torch.ones(out_f, device=dev)
            x = X[(i, p)].float(); Y = x @ W.float().T; yn = Y.square().sum(); L = side.i_norm.sqrt().view(-1, 1) * side.Lr
            set_seed(cfg["seed"]); t0 = time.time()
            fd = nq.factorize_admm_nanoquant(W, side.i_norm, o, mid_rank=r, outer_iters=a.admm_outer_iters, is_transpose=T, rho_scheduler="linear"); t_d += time.time() - t0
            set_seed(cfg["seed"]); t0 = time.time()
            fc = factorize_admm_cov(W, side, o, mid_rank=r, outer_iters=a.admm_outer_iters, is_transpose=T, rho_scheduler="linear"); t_c += time.time() - t0
            results_cov[(i, p)] = fc
            for name, f in [("diag", fd), ("cov", fc)]:
                Wf = f["W_final"].float()
                per[name][p].append({"w": ((Wf - W.float()).square().sum() / W.float().square().sum()).item(),
                                     "y": ((x @ Wf.T - Y).square().sum() / yn).item(),
                                     "s": (((Wf - W.float()) @ L).square().sum() / (W.float() @ L).square().sum()).item()})
        print(f"  layer {i}  diag: out_err {sum(d['y'] for p in PROJ for d in per['diag'][p][-1:])/4:.4f}   cov: {sum(d['y'] for p in PROJ for d in per['cov'][p][-1:])/4:.4f}", flush=True)
    del X; torch.cuda.empty_cache()
    summ = {}
    for name in per:
        summ[name] = {p: {m: sum(d[k] for d in per[name][p]) / len(per[name][p]) for k, m in [("w", "weight_err"), ("y", "output_err"), ("s", "sigma_err")]} for p in PROJ}
        summ[name]["all"] = {m: sum(d[k] for p in PROJ for d in per[name][p]) / 40 for k, m in [("w", "weight_err"), ("y", "output_err"), ("s", "sigma_err")]}
    out["errors"] = summ; out["admm_sec"] = {"diag": t_d, "cov": t_c}
    for name in ["diag", "cov"]:
        d = summ[name]["all"]; print(f"  {name:4s}: weight_err {d['weight_err']:.4f}  OUTPUT_err {d['output_err']:.4f}  Sigma_err {d['sigma_err']:.4f}   ({out['admm_sec'][name]:.0f}s)", flush=True)
    print("  per-proj output err diag->cov: " + ", ".join(f"{p} {summ['diag'][p]['output_err']:.4f}->{summ['cov'][p]['output_err']:.4f}" for p in PROJ), flush=True); dump()

    # ---- 3) install cov factors as binarized NanoQuantLinear (exactly factorize_and_replace's conversion), ADMM-only tau
    print("===== ADMM-only tau, cov objective =====", flush=True)
    import argparse as _ap
    for i, dl in enumerate(draft.layers):
        for p in PROJ:
            lin = getattr(dl.self_attn, p); r = rank_for(lin.in_features, lin.out_features, 1.0)
            lin.__class__ = NanoQuantLinear; lin.__quant_convert__(do_train=False, rank=r, factor_results=_ap.Namespace(**results_cov[(i, p)]))
    out["proxy_admm_cov"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    out["admm_gen"] = measure_tau(target, draft, tok, eval_prompts, tag="admm-cov", **ev)
    out["admm_dev"] = measure_tau(target, draft, tok, dev_prompts, tag="admm-cov-dev", **ev)
    print(f"########## cov-ADMM only: gen tau {out['admm_gen']['tau']:.3f} ± {out['admm_gen']['tau_sem']:.3f}  dev {out['admm_dev']['tau']:.3f}  "
          f"proxy {out['proxy_admm_cov']['prefill_tau']:.2f}   (diag control: 5.530 / dev 5.754 / proxy 7.67) ##########")
    dump(); print("wrote", a.out)


if __name__ == "__main__":
    main()
