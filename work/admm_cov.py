"""Full-covariance ("Hessian-aware") variant of NanoQuant's factorize_admm_nanoquant.

NanoQuant minimises  || diag(no) (W - A B) diag(ni) ||_F   (ni = sqrt i_norm, no = sqrt o_norm).
Here we minimise      || diag(no) (W - A B) L ||_F,  L L^T = Sigma  (the input covariance), i.e. the
true output-error proxy E||(W - AB) x||^2 rather than its diagonal approximation.

Write Sigma = D C D with D = diag(ni) (NanoQuant's weights) and C the input CORRELATION matrix, and work
in NanoQuant's normalised space (W_norm = diag(no) W D, B'' = B D).  The objective is ||(W_norm - A B'') Lr||^2
with Lr Lr^T = C.  ADMM with the binary constraint on B'' (a diagonal D preserves the rank-1-sign structure,
a full Lr would not):

  A-step   min_A  ||(W_norm - A B_z) Lr||^2 + rho||A - V_A||^2   ->  ordinary (mid x mid) solve with
           X_A = (B_z Lr)^T normalised, target W_norm Lr        (NanoQuant's solve, unchanged form)
  B-step   min_B  ||(W_norm - X_B B) Lr||^2 + rho||B - V_B||^2   ->  SYLVESTER:  M B C + rho' B = R
           (the penalty must stay in the ORIGINAL metric -- whitening it too makes C cancel and the step
           collapses back to NanoQuant's).  Solved exactly with eigendecompositions of M (per iteration)
           and C (once per input side).
  Z/U      unchanged.   Export unchanged.   With C = I this is NanoQuant's algorithm, code path included.
Transposed problems (out < in, k/v) mirror this: the covariance side becomes the A-step's Sylvester.
"""
import os, importlib.util
import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nq_admm_nq", os.path.join(os.path.dirname(_here), "NanoQuant/src/nanoquant/core/admm_nq.py"))
nq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(nq)
rank1_approx, _admm_solve_step, RHO = nq.rank1_approx, nq._admm_solve_step, nq.RHO_SCHEDULER_REGISTRY


@torch.no_grad()
def shrink_cov(Sigma, shrinkage):
    """(1-s) Sigma + s mean(diag) I -- its diagonal equals NanoQuant's get_shrunk_stats(i_norm)."""
    S = Sigma.double()
    if 0.0 < shrinkage < 1.0:
        S = (1.0 - shrinkage) * S + shrinkage * S.diagonal().mean() * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    return S


class CovSide:
    """Everything the ADMM needs about one input covariance: i_norm (its diagonal), Lr, and eig(C)."""
    @torch.no_grad()
    def __init__(self, Sigma_shrunk, eps=1e-10):
        S = Sigma_shrunk.double()
        d = S.diagonal().clamp_min(eps); Dinv = 1.0 / d.sqrt()
        C = Dinv.view(-1, 1) * S * Dinv.view(1, -1); C = 0.5 * (C + C.mT)
        self.i_norm = d.float()
        lam, Q = torch.linalg.eigh(C)
        self.lam = lam.clamp_min(1e-8).float(); self.Q = Q.float()            # C = Q diag(lam) Q^T
        self.Lr = (self.Q * self.lam.sqrt().view(1, -1))                        # C = Lr Lr^T (symmetric root)
        self.is_identity = False

    @classmethod
    def identity(cls, n, i_norm, device):
        o = cls.__new__(cls); o.i_norm = i_norm.float(); o.is_identity = True
        o.lam = torch.ones(n, device=device); o.Q = torch.eye(n, device=device); o.Lr = torch.eye(n, device=device)
        return o


@torch.no_grad()
def _sylvester_step(X, Y, Z, U, rho, reg, side, eps=1e-12):
    """Solve  min_B ||(Y - X B) Lr||^2 + rho||B - (Z-U)||^2  exactly, NanoQuant conventions for reg/rho:
       (X^T X) B C + (rho * mean diag(X^T X) + reg) B = X^T Y C + rho (Z - U).
    Returns B in X's dtype. With C = I reduces to NanoQuant's _admm_solve_step (then just call that)."""
    if side.is_identity:
        return _admm_solve_step(X, Y, Z, U, rho, reg, eps)
    dt = X.dtype
    X, Y, Z, U = (t.float() for t in (X, Y, Z, U))
    M = X.mT @ X; M = 0.5 * (M + M.mT)
    diag_mean = M.diagonal().mean().abs()
    shift = torch.clamp(rho * diag_mean + reg, min=eps)              # NanoQuant's stabiliser, on B in the ORIGINAL metric
    mu, P = torch.linalg.eigh(M)                                     # M = P diag(mu) P^T   (mu >= 0)
    Q, lam = side.Q, side.lam
    C = (Q * lam.view(1, -1)) @ Q.mT
    R = (X.mT @ Y) @ C + rho * (Z - U)                               # (mid, in)
    Rh = P.mT @ R @ Q
    Bh = Rh / (mu.clamp_min(0).view(-1, 1) * lam.view(1, -1) + shift)   # every denominator >= reg
    return (P @ Bh @ Q.mT).to(dt)


@torch.no_grad()
def factorize_admm_cov(W, side, o_norm, mid_rank, outer_iters=400, inner_iters=5, reg=3e-2,
                       is_transpose=False, eps=1e-12, rho_scheduler="linear"):
    """Drop-in for factorize_admm_nanoquant(W, i_norm, o_norm, ...) with `side: CovSide` replacing i_norm."""
    # NOTE: the transposed variant (_factorize_T) is numerically unreliable on ill-conditioned real covariances
    # (cond ~4e3: k_proj Sigma-err 0.132 vs 0.049 non-transposed, diag 0.067). The covariance lives on the input
    # side, which is the B-step's dimension, so the non-transposed form is used for every shape.
    if is_transpose:
        pass
    device = W.device; out_features, in_features = W.shape; dt = W.dtype
    norm_i = side.i_norm.sqrt().clamp(eps)
    norm_o = o_norm.sqrt().clamp(eps).unsqueeze(1)
    W_norm = W * norm_i.unsqueeze(0) * norm_o
    Lr = side.Lr
    W_w = W_norm if side.is_identity else (W_norm.float() @ Lr)      # whitened target for the A-step

    A_ls = torch.randn((out_features, mid_rank), device=device, dtype=dt)
    B_ls = torch.randn((mid_rank, in_features), device=device, dtype=dt)
    A_z, B_z = A_ls, B_ls
    if outer_iters > 0:
        A_z = rank1_approx(A_ls, inner_iters, eps); B_z = rank1_approx(B_ls, inner_iters, eps)
    A_u = A_ls - A_z; B_u = B_ls - B_z
    rho_f = RHO[rho_scheduler]
    for itt in range(outer_iters):
        rho = rho_f(itt / outer_iters)
        # A-step (ordinary solve in whitened coordinates; A itself is not transformed)
        Bz_w = B_z if side.is_identity else (B_z.float() @ Lr).to(dt)
        mid_norm_b = Bz_w.norm(dim=1).clamp(eps)
        X_A = Bz_w.mT / mid_norm_b
        A_ls = _admm_solve_step(X_A, W_w.mT, A_z.mT, A_u.mT, rho, reg, eps).mT
        # B-step (exact Sylvester; penalty in the original metric)
        mid_norm_a = A_z.norm(dim=0).clamp(eps)
        X_B = A_z / mid_norm_a
        B_ls = _sylvester_step(X_B, W_norm, B_z, B_u, rho, reg, side, eps)
        A_z = rank1_approx(A_ls + A_u, inner_iters, eps)
        B_z = rank1_approx(B_ls + B_u, inner_iters, eps)
        A_u.add_(A_ls - A_z); B_u.add_(B_ls - B_z)
    return _export(A_z, B_z, A_ls, B_ls, A_u, B_u, norm_o, norm_i, outer_iters, eps)


@torch.no_grad()
def _factorize_T(W, side, o_norm, mid_rank, outer_iters, inner_iters, reg, eps, rho_scheduler):
    """W^T (in x out): the covariance weights the ROW side, so the A-step is the Sylvester one."""
    WT = W.mT; device = W.device; n_rows, n_cols = WT.shape; dt = W.dtype
    norm_row = side.i_norm.sqrt().clamp(eps).unsqueeze(1)
    norm_col = o_norm.sqrt().clamp(eps)
    W_norm = WT * norm_col.unsqueeze(0) * norm_row                    # (in, out)
    Lr = side.Lr
    W_w = W_norm if side.is_identity else (Lr.mT @ W_norm.float())   # row-whitened target for the B-step
    A_ls = torch.randn((n_rows, mid_rank), device=device, dtype=dt)
    B_ls = torch.randn((mid_rank, n_cols), device=device, dtype=dt)
    A_z, B_z = A_ls, B_ls
    if outer_iters > 0:
        A_z = rank1_approx(A_ls, inner_iters, eps); B_z = rank1_approx(B_ls, inner_iters, eps)
    A_u = A_ls - A_z; B_u = B_ls - B_z
    rho_f = RHO[rho_scheduler]
    for itt in range(outer_iters):
        rho = rho_f(itt / outer_iters)
        # A-step: min ||Lr^T (W_norm - A B_z)||^2 + rho||A - V||^2  ==  Sylvester on A^T
        mid_norm_b = B_z.norm(dim=1).clamp(eps)
        X_A = B_z.mT / mid_norm_b                                    # (out, mid)
        A_ls = _sylvester_step(X_A, W_norm.mT, A_z.mT, A_u.mT, rho, reg, side, eps).mT
        # B-step: ordinary solve with row-whitened A
        Az_w = A_z if side.is_identity else (Lr.mT @ A_z.float()).to(dt)
        mid_norm_a = Az_w.norm(dim=0).clamp(eps)
        X_B = Az_w / mid_norm_a
        B_ls = _admm_solve_step(X_B, W_w, B_z, B_u, rho, reg, eps)
        A_z = rank1_approx(A_ls + A_u, inner_iters, eps)
        B_z = rank1_approx(B_ls + B_u, inner_iters, eps)
        A_u.add_(A_ls - A_z); B_u.add_(B_ls - B_z)
    r = _export(A_z, B_z, A_ls, B_ls, A_u, B_u, norm_row, norm_col, outer_iters, eps)
    return {"W_final": r["W_final"].mT, "A": r["B"], "B": r["A"], "A_latent": r["B_latent"], "B_latent": r["A_latent"],
            "scale_pre": r["scale_post"], "scale_post": r["scale_pre"]}


@torch.no_grad()
def _export(A_z, B_z, A_ls, B_ls, A_u, B_u, norm_o, norm_i, outer_iters, eps):
    """Verbatim NanoQuant post-processing (admm_nq.py 'Final export')."""
    A_unbalanced = A_z / norm_o; B_unbalanced = B_z / norm_i
    A_latent_unb = (A_ls + A_u) / norm_o; B_latent_unb = (B_ls + B_u) / norm_i
    bf = (B_unbalanced.norm().clamp(eps) / A_unbalanced.norm().clamp(eps)).sqrt()
    A_final = A_unbalanced * bf; B_final = B_unbalanced / bf
    A_latent = A_latent_unb * bf; B_latent = B_latent_unb / bf
    if outer_iters > 0:
        A_final = A_final * (1.0 / A_z.norm(dim=0).clamp(eps))
    scale_pre = B_final.abs().mean(dim=0).view(1, -1); scale_post = A_final.abs().mean(dim=1).view(1, -1)
    return {"W_final": F.linear(A_final, B_final.mT), "A": A_final.mT, "B": B_final,
            "A_latent": A_latent.mT, "B_latent": B_latent, "scale_pre": scale_pre, "scale_post": scale_post}
