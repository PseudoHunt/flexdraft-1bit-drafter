"""Option A: NanoQuant's o_norm (per-output-channel gradient second moment) computed from the
TARGET-TEACHER loss, instead of o_norm = 1.

  ADMM objective  ||diag(sqrt o_norm) (W - AB) diag(sqrt i_norm)||_F
  o_norm[c] = E_t[ (dL/dy_c)^2 ] over mask-token positions, L = CE(drafter logits, target greedy tokens),
  gradients taken through the FP drafter on on-policy windows, robust-clipped and shrunk exactly like
  nanoquant.core.importance (online strategy, 99.9th-pct clipping, shrinkage 0.4).
Everything else identical to run_residual.py (residual base r2=256/96, ce tuning lat 1e-4 / sc 1e-5).
Also reports held-out OUTPUT error ||(W - W_hat) X||^2 / ||W X||^2 on real mask inputs for o_norm=1 vs learned.
"""
import argparse, json, os, sys, time
from collections import defaultdict
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))

from fd_common import load_models, build_prompts, measure_tau, load_admm_nq
from nanoquant.modules.linear import NanoQuantLinear          # before core.* (circular import)
from nanoquant.core.compress_block import factorize_and_replace, get_param_group_config
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils.utils import find_layers, set_seed
from run_target_teacher import BS, draft_logits, loss_fn, prefill_tau, train, force_unfused, rank_for, PROJ_ORDER
from run_residual import _forward_residual, add_residual_base   # installs the two-path forward
nq = load_admm_nq()
R2 = {"q_proj": 256, "k_proj": 96, "v_proj": 96, "o_proj": 256}
GRAD_SCALE = 1e6; PCT = 0.999


def collect_o_norm(target, draft, hid, labels, mask, pe, seqlen, bsz, shrinkage):
    """Mirror of nanoquant.core.importance._online_clipping_hook (backward branch) driven by our loss."""
    mods = {(i, p): getattr(dl.self_attn, p) for i, dl in enumerate(draft.layers) for p in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    acc = {k: torch.zeros(m.out_features, device=hid.device) for k, m in mods.items()}
    n = defaultdict(int); gmax = {}
    def mk(key):
        def hook(mod, grad_in, grad_out):
            g = grad_out[0].detach().float().reshape(-1, grad_out[0].shape[-1]) * GRAD_SCALE
            norms = g.norm(dim=1, keepdim=True)
            k = max(1, int(norms.numel() * (1 - PCT))); tau = torch.topk(norms.reshape(-1), k).values[-1]
            if key not in gmax: gmax[key] = tau
            elif tau > gmax[key]:
                acc[key].mul_((tau / (gmax[key] + 1e-8)) ** 2); gmax[key] = tau
            clip = torch.clamp(gmax[key] / (norms + 1e-8), max=1.0)
            acc[key] += (g * clip).square().mean(0) / GRAD_SCALE; n[key] += 1
        return hook
    hs = [m.register_full_backward_hook(mk(k)) for k, m in mods.items()]
    ws = [m.weight for m in mods.values()]
    for w in ws: w.requires_grad_(True)               # so grad_output exists at every module
    with torch.enable_grad():
        for s in range(0, hid.shape[0], bsz):
            lg = draft_logits(target, draft, hid[s:s + bsz], labels[s:s + bsz, 0], mask, pe, seqlen)
            loss_fn(lg, labels[s:s + bsz, 1:], "ce").backward()
            for w in ws: w.grad = None
    for w in ws: w.requires_grad_(False)
    for h in hs: h.remove()
    out = {}
    for k in mods:
        t = acc[k] / max(n[k], 1)
        if 0 < shrinkage < 1: t = t * (1 - shrinkage) + t.mean() * shrinkage
        out[k] = t
    return out


@torch.no_grad()
def capture_inputs(draft, target, hid, labels, mask, pe, seqlen, bsz=8):
    mods = {(i, p): getattr(dl.self_attn, p) for i, dl in enumerate(draft.layers) for p in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    store = defaultdict(list)
    hs = [m.register_forward_pre_hook(lambda mod, inp, k=k: store[k].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).clone())) for k, m in mods.items()]
    for s in range(0, hid.shape[0], bsz):
        draft_logits(target, draft, hid[s:s + bsz], labels[s:s + bsz, 0], mask, pe, seqlen)
    for h in hs: h.remove()
    return {k: torch.cat(v) for k, v in store.items()}


@torch.no_grad()
def output_error_diag(draft, stats, onorm, X, cfg, min_rank):
    """Base-only ADMM (no residual) with o_norm=1 vs learned; weight- and output-space rel. error per proj type."""
    dev = next(draft.parameters()).device; res = {"ones": defaultdict(list), "learned": defaultdict(list)}
    for i, dl in enumerate(draft.layers):
        for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            lin = getattr(dl.self_attn, p); W = lin.weight.data; out_f, in_f = W.shape
            r = rank_for(in_f, out_f, cfg["bits"]); i_norm = stats[(i, "o" if p == "o_proj" else "qkv")].to(dev).float()
            x = X[(i, p)].float(); Y = x @ W.float().T; yn = Y.square().sum()
            sw = onorm[(i, p)].float().sqrt().view(1, -1); Yw = Y * sw; ywn = Yw.square().sum()   # loss-weighted output
            for name, on in [("ones", torch.ones(out_f, device=dev)), ("learned", onorm[(i, p)])]:
                set_seed(cfg["seed"])
                f = nq.factorize_admm_nanoquant(W, i_norm, on, mid_rank=r, outer_iters=cfg["admm_outer_iters"],
                        inner_iters=cfg["admm_inner_iters"], reg=cfg["admm_reg"], is_transpose=(out_f < in_f),
                        rho_scheduler=cfg["admm_penalty_scheduler"])["W_final"].float()
                E = x @ f.T - Y
                res[name][p].append({"w": ((f - W.float()).square().sum() / W.float().square().sum()).item(),
                                     "y": (E.square().sum() / yn).item(), "yw": ((E * sw).square().sum() / ywn).item()})
        print(f"  diag layer {i}", flush=True)
    summ = {}
    for name in res:
        summ[name] = {p: {"weight_err": sum(d["w"] for d in v) / len(v), "output_err": sum(d["y"] for d in v) / len(v),
                          "weighted_output_err": sum(d["yw"] for d in v) / len(v)} for p, v in res[name].items()}
        summ[name]["all"] = {k2: sum(d[k1] for v in res[name].values() for d in v) / 40 for k1, k2 in
                             [("w", "weight_err"), ("y", "output_err"), ("yw", "weighted_output_err")]}
    return summ


def admm_all(draft, stats, onorm, cfg, min_rank):
    dev = next(draft.parameters()).device; e1, e2 = [], []
    for i, dl in enumerate(draft.layers):
        for name in PROJ_ORDER:
            lin = find_layers(dl)[name]; p = name.split(".")[1]
            r = rank_for(lin.in_features, lin.out_features, cfg["bits"]); assert r > min_rank
            i_norm = stats[(i, "o" if p == "o_proj" else "qkv")].to(dev).float(); on = onorm[(i, p)].to(dev).float()
            lin.register_buffer("i_norm", i_norm, persistent=False); lin.register_buffer("o_norm", on, persistent=False)
            W = lin.weight.data.clone()
            nano, _ = factorize_and_replace(dl, name, r, cfg)
            a, b = add_residual_base_onorm(nano, W, i_norm, on, R2[p], cfg); e1.append(a); e2.append(b)
        print(f"  ADMM layer {i}: rel_err {sum(e1[-4:])/4:.4f} -> {sum(e2[-4:])/4:.4f}", flush=True)
    return sum(e1) / 40, sum(e2) / 40


@torch.no_grad()
def add_residual_base_onorm(lin, W, i_norm, o_norm, r2, cfg):
    from run_residual import deployed_w1
    W1 = deployed_w1(lin); R = (W.float() - W1).to(W.dtype); out_f, in_f = W.shape
    set_seed(cfg["seed"])
    res = nq.factorize_admm_nanoquant(R, i_norm.float(), o_norm.float(), mid_rank=r2, outer_iters=cfg["admm_outer_iters"],
            inner_iters=cfg["admm_inner_iters"], reg=cfg["admm_reg"], is_transpose=(out_f < in_f), rho_scheduler=cfg["admm_penalty_scheduler"])
    def P(t, grp):
        p = nn.Parameter(t.to(lin.dtype).contiguous(), requires_grad=True); p.optim_group = grp; return p
    lin.U2_latent = P(res["A_latent"].mT, "binary"); lin.V2_latent = P(res["B_latent"], "binary")
    lin.scale2_pre = P(res["scale_pre"], "scale"); lin.scale2_post = P(res["scale_post"], "scale"); lin.rank2 = r2
    W2 = (torch.sign(lin.U2_latent.float()) @ torch.sign(lin.V2_latent.float())) * lin.scale2_pre.float() * lin.scale2_post.float().view(-1, 1)
    n = W.float().square().sum()
    return ((W1 - W.float()).square().sum() / n).item(), ((W1 + W2 - W.float()).square().sum() / n).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--eval-samples", type=int, default=40); ap.add_argument("--dev-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256); ap.add_argument("--train-windows", type=int, default=1024)
    ap.add_argument("--heldout-windows", type=int, default=128); ap.add_argument("--seqlen", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--scale-lr", type=float, default=1e-5); ap.add_argument("--latent-lr", type=float, default=1e-4)
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--shrinkage", type=float, default=0.4); ap.add_argument("--skip-diag", action="store_true")
    ap.add_argument("--save-params", type=str, default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = {"args": vars(a), "r2": R2}
    def dump(): json.dump(out, open(a.out, "w"), indent=2)
    cfg = NanoQuantConfig(bits=1.0, seed=a.seed, admm_type="nanoquant", admm_outer_iters=a.admm_outer_iters, admm_inner_iters=5,
                          admm_penalty_scheduler="linear", tune_nonfact=False, tune_fact=True, tune_model=False)
    target, draft, tok = load_models(a.target, a.draft); force_unfused(draft); dev = target.device
    for p in list(target.parameters()) + list(draft.parameters()): p.requires_grad_(False)
    ev = dict(block_size=BS, max_new_tokens=a.max_new_tokens, temperature=0.0, thr=0.01, strategy="cumulative_product")
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev); dev_prompts = build_prompts(tok, "gsm8k", a.dev_samples, dev, split_offset=a.eval_samples)

    c = torch.load(a.cache, weights_only=False)
    stats = {k: v.to(dev) for k, v in c["stats"].items()}
    hid, labels, mask, pe = c["hid"].to(dev), c["labels"].to(dev), c["mask"].to(dev), tuple(t.to(dev) for t in c["pe"])
    nW = a.train_windows + a.heldout_windows; assert hid.shape[0] == nW; tr = slice(0, a.train_windows); ho = slice(a.train_windows, nW)

    print("===== o_norm from target-teacher CE gradients (FP drafter, train windows) =====", flush=True)
    t0 = time.time(); onorm = collect_o_norm(target, draft, hid[tr], labels[tr], mask, pe, a.seqlen, a.batch, a.shrinkage)
    spread = {p: float(torch.stack([onorm[(i, p)] for i in range(10)]).mean(0).max() / torch.stack([onorm[(i, p)] for i in range(10)]).mean(0).min()) for p in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    out["onorm_maxmin_ratio_after_shrink"] = spread
    print(f"  done in {time.time()-t0:.0f}s; per-proj max/min channel weight after shrinkage: {spread}", flush=True)

    if not a.skip_diag:
        print("===== diagnostic: base-only ADMM, o_norm=1 vs learned, held-out output error =====", flush=True)
        X = capture_inputs(draft, target, hid[ho], labels[ho], mask, pe, a.seqlen)
        out["diag"] = output_error_diag(draft, stats, onorm, X, cfg, a.min_rank); del X; torch.cuda.empty_cache()
        for name in ["ones", "learned"]:
            d = out["diag"][name]["all"]; print(f"  {name:8s} weight_err {d['weight_err']:.4f}  output_err {d['output_err']:.4f}  loss-weighted output_err {d['weighted_output_err']:.4f}", flush=True)
        dump()

    print("===== ADMM (i_norm + learned o_norm) + residual =====", flush=True)
    out["rel_err_base"], out["rel_err_residual"] = admm_all(draft, stats, onorm, cfg, a.min_rank)
    out["proxy_admm"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    out["admm_gen"] = measure_tau(target, draft, tok, eval_prompts, tag="admm+onorm", **ev)
    out["admm_dev"] = measure_tau(target, draft, tok, dev_prompts, tag="admm+onorm-dev", **ev)
    print(f"  ADMM-only: gen {out['admm_gen']['tau']:.3f}  dev {out['admm_dev']['tau']:.3f}  proxy {out['proxy_admm']['prefill_tau']:.2f}", flush=True); dump()

    groups = get_param_group_config(draft, binary_lr=a.latent_lr, scale_lr=a.scale_lr, bias_lr=a.scale_lr)
    params = [p for g in groups for p in g["params"]]; init = [p.detach().clone() for p in params]
    is_bin = [getattr(p, "optim_group", "") == "binary" for p in params]
    print("===== tune (ce) =====", flush=True); t0 = time.time()
    hist = train(target, draft, params, groups, hid[tr], labels[tr], mask, pe, a.seqlen, "ce", a.epochs, a.batch, a.seed)
    with torch.no_grad():
        flips = sum((torch.sign(p) != torch.sign(q)).sum().item() for p, q, b in zip(params, init, is_bin) if b) / sum(p.numel() for p, b in zip(params, is_bin) if b)
    r = {"train_loss": hist, "train_sec": time.time() - t0, "flip_frac": flips,
         "proxy": prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen), "proxy_train": prefill_tau(target, draft, hid[:128], labels[:128], mask, pe, a.seqlen)}
    r["dev"] = measure_tau(target, draft, tok, dev_prompts, tag="tuned-dev", **ev); r["gen"] = measure_tau(target, draft, tok, eval_prompts, tag="tuned", **ev)
    out["tuned"] = r; dump()
    if a.save_params:
        torch.save({n: p.detach().cpu() for n, p in draft.named_parameters() if p.requires_grad}, a.save_params)
    print(f"\n########## [o_norm from target loss] ADMM+residual {out['admm_gen']['tau']:.3f}  ->  tuned {r['gen']['tau']:.3f} ± {r['gen']['tau_sem']:.3f}  (flips {100*flips:.3f}%) ##########")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
