import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open
from admm_cov import nq, factorize_admm_cov, CovSide, shrink_cov, _sylvester_step
from nanoquant_seed import set_seed
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); dev = "cuda"
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
f = safe_open(os.path.join(S, "ckpt/flexdraft-qwen3-8b/model.safetensors"), "pt")
def rank_for(i, o, b=1.0):
    r = (i * o * b) / (i + o) - 16; r = (int(r) // 32) * 32; return min(max(r, 32), min(i, o))
def rel(X, Y): return ((X.float() - Y.float()).square().sum() / Y.float().square().sum()).item()

# ---------- [0] eigh timing (the per-iteration cost of the Sylvester step)
for n in (800, 2016):
    M = torch.randn(n, n, device=dev); M = M @ M.mT + torch.eye(n, device=dev)
    torch.linalg.eigh(M); torch.cuda.synchronize(); t = time.time()
    for _ in range(5): torch.linalg.eigh(M)
    torch.cuda.synchronize(); print(f"[0] eigh({n}) fp32: {(time.time()-t)/5*1000:.0f} ms")

# ---------- [1] Sylvester solver: satisfies its normal equations; equals Cholesky solve when C = I
g = torch.Generator(device=dev).manual_seed(3)
n_in, mid, n_out = 512, 128, 384
X = torch.randn(n_out, mid, generator=g, device=dev); Y = torch.randn(n_out, n_in, generator=g, device=dev)
Z = torch.randn(mid, n_in, generator=g, device=dev); U = 0.1 * torch.randn(mid, n_in, generator=g, device=dev)
A_ = torch.randn(n_in, n_in, generator=g, device=dev); Sigma = A_ @ A_.mT / n_in + 0.05 * torch.eye(n_in, device=dev)
side = CovSide(shrink_cov(Sigma, 0.4)); rho, reg = 0.7, 3e-2
B = _sylvester_step(X, Y, Z, U, rho, reg, side).float()
M = X.mT @ X; dm = M.diagonal().mean(); C = (side.Q * side.lam.view(1, -1)) @ side.Q.mT
lhs = M @ B @ C + (rho * dm + reg) * B; rhs = (X.mT @ Y) @ C + rho * (Z - U)
print(f"[1] Sylvester normal-equation residual: {rel(lhs, rhs):.2e}  (fp32, expect ~1e-10)")
sideI = CovSide.identity(n_in, side.i_norm, dev)
print(f"[1] C=I: Sylvester vs NanoQuant Cholesky solve rel diff: {rel(_sylvester_step(X, Y, Z, U, rho, reg, CovSide(torch.eye(n_in, device=dev).double()*side.i_norm.double().view(-1,1).sqrt()*side.i_norm.double().view(1,-1).sqrt(), ) if False else sideI), nq._admm_solve_step(X, Y, Z, U, rho, reg)):.2e}")

# ---------- [2] full ADMM on real drafter weights
for key in ["layers.5.self_attn.q_proj.weight", "layers.5.self_attn.k_proj.weight"]:
    W = f.get_tensor(key).to(dev); out_f, in_f = W.shape; r = rank_for(in_f, out_f); T = out_f < in_f
    o_norm = torch.ones(out_f, device=dev)
    i_norm = torch.rand(in_f, generator=g, device=dev) * 3 + 0.2
    print(f"\n== {key.split('.',1)[1]} {out_f}x{in_f} r={r} transpose={T} ({ITERS} iters) ==")
    # identity path must be NanoQuant bit-for-bit (same ops, same RNG)
    set_seed(0); ref = nq.factorize_admm_nanoquant(W, i_norm, o_norm, mid_rank=r, outer_iters=ITERS, is_transpose=T, rho_scheduler="linear")
    set_seed(0); mine = factorize_admm_cov(W, CovSide.identity(in_f, i_norm, dev), o_norm, mid_rank=r, outer_iters=ITERS, is_transpose=T, rho_scheduler="linear")
    print(f"[2a] C=I path vs NanoQuant: W_final rel diff {rel(mine['W_final'], ref['W_final']):.1e}  (expect 0)   unweighted err {rel(ref['W_final'], W):.4f}")
    # correlated inputs: Sigma-weighted error must drop
    Qm, _ = torch.linalg.qr(torch.randn(in_f, in_f, generator=g, device=dev))
    Cc = Qm @ torch.diag(torch.logspace(0, -3, in_f, device=dev)) @ Qm.mT; Cc = Cc / Cc.diagonal().mean()
    d = i_norm.sqrt(); Sig = shrink_cov(d.view(-1, 1) * Cc * d.view(1, -1), 0.4); side = CovSide(Sig)
    L = side.i_norm.sqrt().view(-1, 1) * side.Lr
    assert torch.allclose(L @ L.mT, Sig.float(), rtol=1e-3, atol=1e-4)
    print(f"[2b] correlated C: cond {side.lam.max().item()/side.lam.min().item():.0f}")
    set_seed(0); t0 = time.time(); ref = nq.factorize_admm_nanoquant(W, side.i_norm, o_norm, mid_rank=r, outer_iters=ITERS, is_transpose=T, rho_scheduler="linear"); t_ref = time.time() - t0
    set_seed(0); t0 = time.time(); mine = factorize_admm_cov(W, side, o_norm, mid_rank=r, outer_iters=ITERS, is_transpose=T, rho_scheduler="linear"); t_cov = time.time() - t0
    wv = side.i_norm.sqrt().view(1, -1)
    de = lambda Wf: rel(Wf.float() * wv, W.float() * wv)          # ||(W-Wf)D||^2 / ||W D||^2
    ce = lambda Wf: rel(Wf.float() @ L, W.float() @ L)            # ||(W-Wf)L||^2 / ||W L||^2
    print(f"     NanoQuant diag : diag-err {de(ref['W_final']):.4f}  Sigma-err {ce(ref['W_final']):.4f}   ({t_ref:.1f}s)")
    print(f"     cov-ADMM       : diag-err {de(mine['W_final']):.4f}  Sigma-err {ce(mine['W_final']):.4f}   ({t_cov:.1f}s)   <- Sigma-err must be lower")
