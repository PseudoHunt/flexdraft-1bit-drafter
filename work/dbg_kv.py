import os, sys, torch, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open
from admm_cov import nq, factorize_admm_cov, CovSide, _sylvester_step
from nanoquant_seed import set_seed
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); dev = "cuda"
f = safe_open(os.path.join(S, "ckpt/flexdraft-qwen3-8b/model.safetensors"), "pt")
cs = torch.load(os.path.join(S, "work/cov_sides.pt"))
def mk(k):
    o = CovSide.__new__(CovSide); d = cs[k]; o.i_norm = d["i_norm"].to(dev); o.lam = d["lam"].to(dev); o.Q = d["Q"].to(dev)
    o.Lr = o.Q * o.lam.sqrt().view(1, -1); o.is_identity = False; return o
def rel(X, Y): return ((X.float() - Y.float()).square().sum() / Y.float().square().sum()).item()
for key, kind, T_default in [("layers.5.self_attn.k_proj.weight", "qkv", True), ("layers.5.self_attn.q_proj.weight", "qkv", False)]:
    W = f.get_tensor(key).to(dev); out_f, in_f = W.shape; r = 800 if out_f < in_f else 2016
    side = mk((5, kind)); L = side.i_norm.sqrt().view(-1, 1) * side.Lr; o = torch.ones(out_f, device=dev)
    print(f"\n== {key.split('.',1)[1]} {out_f}x{in_f} r={r}  C: cond {side.lam.max()/side.lam.min():.0f} lam[{side.lam.min():.2e},{side.lam.max():.1f}] ==")
    ce = lambda Wf: rel(Wf.float() @ L, W.float() @ L); de = lambda Wf: rel(Wf.float() * side.i_norm.sqrt().view(1, -1), W.float() * side.i_norm.sqrt().view(1, -1))
    set_seed(0); ref = nq.factorize_admm_nanoquant(W, side.i_norm, o, mid_rank=r, outer_iters=400, is_transpose=T_default, rho_scheduler="linear")
    print(f"  NanoQuant diag (T={T_default}) : diag-err {de(ref['W_final']):.4f}  Sigma-err {ce(ref['W_final']):.4f}")
    for T in ([True, False] if out_f < in_f else [False]):
        set_seed(0); t0 = time.time(); m = factorize_admm_cov(W, side, o, mid_rank=r, outer_iters=400, is_transpose=T, rho_scheduler="linear")
        print(f"  cov-ADMM   (transpose={T!s:5s}): diag-err {de(m['W_final']):.4f}  Sigma-err {ce(m['W_final']):.4f}   ({time.time()-t0:.0f}s)")
