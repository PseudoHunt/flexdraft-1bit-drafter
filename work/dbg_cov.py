import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open
from admm_cov import nq, factorize_admm_cov, CovSide, shrink_cov
from nanoquant_seed import set_seed
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); dev = "cuda"
f = safe_open(os.path.join(S, "ckpt/flexdraft-qwen3-8b/model.safetensors"), "pt")
W = f.get_tensor("layers.5.self_attn.q_proj.weight").to(dev); in_f = W.shape[1]; r = 2016
def rel(X, Y): return ((X.float() - Y.float()).square().sum() / Y.float().square().sum()).item()
print("rel sanity:", rel(W, W), rel(0.9 * W.float(), W))
ones = torch.ones(in_f, device=dev); o = torch.ones(W.shape[0], device=dev)
g = torch.Generator(device=dev).manual_seed(3)
rnd = torch.rand(in_f, generator=g, device=dev) * 3 + 0.2
Qm, _ = torch.linalg.qr(torch.randn(in_f, in_f, generator=g, device=dev))
Cc = Qm @ torch.diag(torch.logspace(0, -3, in_f, device=dev)) @ Qm.mT; Cc = Cc / Cc.diagonal().mean()
d = rnd.sqrt(); Sig = shrink_cov(d.view(-1, 1) * Cc * d.view(1, -1), 0.4); side = CovSide(Sig)
print(f"i_norm ranges: rnd [{rnd.min():.2f},{rnd.max():.2f}]  shrunk-diag [{side.i_norm.min():.2f},{side.i_norm.max():.2f}]")
for name, inorm in [("ones", ones), ("rnd", rnd), ("shrunk diag", side.i_norm)]:
    set_seed(0); res = nq.factorize_admm_nanoquant(W, inorm, o, mid_rank=r, outer_iters=150, rho_scheduler="linear")
    Wf = res["W_final"]; w = inorm.sqrt().view(1, -1)
    print(f"NanoQuant i_norm={name:12s} unweighted {rel(Wf, W):.4f}  diag-weighted {rel((W.float()-Wf.float())*w, W.float()*w):.4f}")
# how much of ||W D|| lives in the correlated directions?  compare Sigma-weighted error of the SAME W_final
L = side.i_norm.sqrt().view(-1, 1) * side.Lr
print(f"same W_final: Sigma-weighted {rel((W.float()-Wf.float())@L, W.float()@L):.4f}")
# eigen-spectrum of the whitening: how anisotropic is L L^T along W's row space?
E = (W.float() - Wf.float()); print("err energy fraction along top-64 eigvecs of C:", ((E @ side.Q[:, -64:]).square().sum() / E.square().sum()).item())
