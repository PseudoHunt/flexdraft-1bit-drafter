"""Round 2 repeated with GROUP-128 scales.

Format:  W_ij ~= G[i, g(j)] * s_pre[j] * (S_A S_B)_ij ,  G in R^{out x in/128}
  - strict superset of NanoQuant's diag(s_post) S_A S_B diag(s_pre): G[i,:] = s_post[i] recovers it exactly
  - bits = r(in+out) + 16(in + out*in/128)  ->  ~1.11 bpw at the same ranks (was 0.99)
  - G is initialised least-squares-optimal per (row, group) under the SAME i_norm-weighted
    objective ADMM minimised, so "ADMM-only + g128" is a proper baseline, not just a reparam.
Everything else identical to run_onpolicy.py round 2 (same seeds/prompts/data/grid).
"""
import argparse, json, os, sys, time
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))

from fd_common import load_models, build_prompts, measure_tau
from nanoquant.modules.linear import NanoQuantLinear          # before core.* (circular import)
from nanoquant.core.compress_block import factorize_and_replace, get_param_group_config
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils.utils import find_layers
from run_target_teacher import BS, prefill_tau, train, force_unfused, rank_for, PROJ_ORDER
from run_onpolicy import chat_ids, onpolicy_windows
from run_all import install_calibration, collect_i_norm

GROUP = 128

# ---------------------------------------------------------------- grouped forward
_orig_forward = NanoQuantLinear.forward

def _forward_g128(self, x):
    G = getattr(self, "scale_grp", None)
    if G is None:
        return _orig_forward(self, x)
    V = self.V_latent if self.do_train and hasattr(self, "V_latent") else self.V
    U = self.U_latent if self.do_train and hasattr(self, "U_latent") else self.U
    Vq, Uq = self.quantize(V), self.quantize(U)
    B, T, In = x.shape; ng = G.shape[1]; gs = In // ng
    xs = (x * self.scale_pre).view(B, T, ng, gs)
    z = torch.einsum("btgk,rgk->btgr", xs, Vq.view(-1, ng, gs))      # per-group rank-r codes
    sm = getattr(self, "scale_mid", None)
    if sm is not None:
        z = z * sm.view(-1).to(z.dtype)
    o = torch.einsum("btgr,or->btgo", z, Uq)                          # per-group outputs
    y = torch.einsum("btgo,og->bto", o, G.to(o.dtype))                # group scales, summed
    if self.bias is not None:
        y = y + self.bias
    return y

NanoQuantLinear.forward = _forward_g128


@torch.no_grad()
def to_g128(lin, W, i_norm, group=GROUP):
    """Replace scale_post by LS-optimal per-(row,group) scales under i_norm weighting."""
    out_f, in_f = W.shape; ng = in_f // group
    P = torch.sign(lin.U_latent.float()) @ torch.sign(lin.V_latent.float())          # (out,in) binary product
    Q = P * lin.scale_pre.float()                                                     # incl. per-input scale
    w = i_norm.float().view(1, ng, group)
    Wg, Qg = W.float().view(out_f, ng, group), Q.view(out_f, ng, group)
    num = (w * Wg * Qg).sum(-1); den = (w * Qg * Qg).sum(-1).clamp_min(1e-12)
    G = (num / den)
    err_pc = ((Q * lin.scale_post.float().view(-1, 1) - W.float()).square().sum() / W.float().square().sum()).item()
    W_g = (Qg * G.unsqueeze(-1)).view(out_f, in_f)
    err_g = ((W_g - W.float()).square().sum() / W.float().square().sum()).item()
    p = nn.Parameter(G.to(lin.dtype), requires_grad=True); p.optim_group = "scale"
    lin.scale_grp = p
    lin.scale_post.requires_grad_(False)          # unused by grouped forward, kept for reference
    return err_pc, err_g


def add_scale_mid(lin):
    p = nn.Parameter(torch.ones(1, lin.rank, device=lin.scale_pre.device, dtype=lin.dtype), requires_grad=True)
    p.optim_group = "scale"; lin.scale_mid = p


def admm_all_g128(draft, stats, cfg, min_rank, group=GROUP, scale_mid=False):
    dev = next(draft.parameters()).device; errs = {"per_channel": [], "g128": []}
    for i, dl in enumerate(draft.layers):
        for name in PROJ_ORDER:
            lin = find_layers(dl)[name]
            r = rank_for(lin.in_features, lin.out_features, cfg["bits"]); assert r > min_rank
            key = (i, "o" if name.endswith("o_proj") else "qkv")
            i_norm = stats[key].to(dev).float()
            lin.register_buffer("i_norm", i_norm, persistent=False)
            lin.register_buffer("o_norm", torch.ones(lin.out_features, device=dev), persistent=False)
            W = lin.weight.data.clone()
            nano, _ = factorize_and_replace(dl, name, r, cfg)
            if group:
                e_pc, e_g = to_g128(nano, W, i_norm, group)
                errs["per_channel"].append(e_pc); errs["g128"].append(e_g)
            if scale_mid:
                add_scale_mid(nano)
        print(f"  ADMM+g128 layer {i} done", flush=True)
    return errs


def bpw_report(draft):
    bits = wts = 0
    for dl in draft.layers:
        for name in PROJ_ORDER:
            lin = getattr(dl.self_attn, name.split(".")[1]); r = lin.rank; i_, o_ = lin.in_features, lin.out_features
            G = getattr(lin, "scale_grp", None)
            n_scales = i_ + (G.numel() if G is not None else o_) + (r if getattr(lin, "scale_mid", None) is not None else 0)
            bits += r * (i_ + o_) + 16 * n_scales; wts += i_ * o_
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
    ap.add_argument("--scale-lr", type=float, default=1e-5)
    ap.add_argument("--latent-lrs", type=str, default="1e-5,1e-4,3e-4,1e-3")
    ap.add_argument("--modes", type=str, default="acc,ce")
    ap.add_argument("--admm-outer-iters", type=int, default=400); ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--group", type=int, default=GROUP, help="0 = per-channel (NanoQuant default)")
    ap.add_argument("--scale-mid", action="store_true")
    ap.add_argument("--cache", type=str, default="", help="path to cache stats+windows across invocations")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lat_lrs = [float(x) for x in a.latent_lrs.split(",")]; modes = a.modes.split(",")
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
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev)
    dev_prompts = build_prompts(tok, "gsm8k", a.dev_samples, dev, split_offset=a.eval_samples)

    nW = a.train_windows + a.heldout_windows
    if a.cache and os.path.exists(a.cache):
        print(f"===== loading cached stats/windows from {a.cache} =====", flush=True)
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

    tag = f"g{a.group}" if a.group else "perch"; tag += "+mid" if a.scale_mid else ""
    print(f"===== ADMM (calibrated i_norm), scales: {tag} =====", flush=True)
    errs = admm_all_g128(draft, stats, cfg, a.min_rank, group=a.group, scale_mid=a.scale_mid)
    out["recon_err"] = {k: sum(v) / len(v) for k, v in errs.items() if v}
    out["bpw"], out["MiB"] = bpw_report(draft); out["scale_format"] = tag
    print(f"  weight rel_err {out['recon_err']};  {out['bpw']:.3f} bpw, {out['MiB']:.1f} MiB", flush=True)
    lin = getattr(draft.layers[0].self_attn, "q_proj")
    if a.group:
      with torch.no_grad():
          x = torch.randn(1, 4, lin.in_features, device=dev, dtype=torch.bfloat16)
          Gsave = lin.scale_grp.data.clone(); lin.scale_grp.data.copy_(lin.scale_post.data.view(-1, 1).expand_as(Gsave))
          y_g = _forward_g128(lin, x).float(); y_pc = _orig_forward(lin, x).float()
          lin.scale_grp.data.copy_(Gsave)
          print(f"  sanity: grouped fwd with G=s_post vs per-channel fwd, max|Δ|/max|y| = "
                f"{(y_g - y_pc).abs().max().item() / y_pc.abs().max().item():.2e}", flush=True)

    out["proxy_admm"] = prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)
    out["admm_gen"] = measure_tau(target, draft, tok, eval_prompts, tag="admm-"+tag, **ev)
    out["admm_dev"] = measure_tau(target, draft, tok, dev_prompts, tag="admm-"+tag+"-dev", **ev)
    print(f"  ADMM+g128 only: gen tau {out['admm_gen']['tau']:.3f}  dev {out['admm_dev']['tau']:.3f}", flush=True)
    dump()

    groups0 = get_param_group_config(draft, 1.0, 1.0, 1.0)
    params = [p for g in groups0 for p in g["params"]]
    init = [p.detach().clone() for p in params]
    is_bin = [getattr(p, "optim_group", "") == "binary" for p in params]
    n_scale = sum(p.numel() for p, b in zip(params, is_bin) if not b)
    print(f"  trainable: {sum(p.numel() for p in params)/1e6:.1f}M ({n_scale/1e6:.2f}M scale params incl. G)", flush=True)

    def restore():
        with torch.no_grad():
            for p, q in zip(params, init): p.copy_(q)
    def flip_frac():
        with torch.no_grad():
            f = sum((torch.sign(p) != torch.sign(q)).sum().item() for p, q, b in zip(params, init, is_bin) if b)
            return f / sum(p.numel() for p, b in zip(params, is_bin) if b)

    print("===== sweep =====", flush=True)
    for mode in modes:
        for llr in lat_lrs:
            key = f"{mode}@lat{llr:g}_sc{a.scale_lr:g}"; restore()
            groups = get_param_group_config(draft, binary_lr=llr, scale_lr=a.scale_lr, bias_lr=a.scale_lr)
            t0 = time.time()
            hist = train(target, draft, params, groups, hid[tr], labels[tr], mask, pe, a.seqlen, mode, a.epochs, a.batch, a.seed)
            r = {"mode": mode, "latent_lr": llr, "scale_lr": a.scale_lr, "train_loss": hist,
                 "train_sec": time.time() - t0, "flip_frac": flip_frac(),
                 "proxy": prefill_tau(target, draft, hid[ho], labels[ho], mask, pe, a.seqlen)}
            r["dev"] = measure_tau(target, draft, tok, dev_prompts, tag=key + "-dev", **ev)
            r["gen"] = measure_tau(target, draft, tok, eval_prompts, tag=key, **ev)
            out["runs"][key] = r; dump()
            print(f"  >>> {key:24s} dev={r['dev']['tau']:.3f}  GEN={r['gen']['tau']:.3f}±{r['gen']['tau_sem']:.3f}  "
                  f"proxy={r['proxy']['prefill_tau']:.3f}  flips={100*r['flip_frac']:.3f}%  ({r['train_sec']:.0f}s)", flush=True)
    best = max(out["runs"], key=lambda k: out["runs"][k]["dev"]["tau"]); out["dev_selected"] = best
    print(f"\n########## dev-selected: {best}  gen tau = {out['runs'][best]['gen']['tau']:.3f}  (ADMM-only {out['admm_gen']['tau']:.3f}, {out['scale_format']}) ##########")
    dump(); print("wrote", a.out)


if __name__ == "__main__":
    main()
