import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NanoQuant/src"))
from fd_common import load_admm_nq
from safetensors import safe_open
nq = load_admm_nq()

def rank_for(i,o,b=1.0):
    r=(i*o*b)/(i+o)-16; r=(int(r)//32)*32; r=32 if r==0 else max(r,32); return min(r,min(i,o))

p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpt/flexdraft-qwen3-8b/model.safetensors")
f = safe_open(p, "pt")
for key in ["layers.5.self_attn.q_proj.weight", "layers.5.self_attn.o_proj.weight", "layers.5.self_attn.k_proj.weight"]:
    W = f.get_tensor(key).cuda()
    out_f, in_f = W.shape
    r = rank_for(in_f, out_f)
    i_n = torch.ones(in_f, device="cuda"); o_n = torch.ones(out_f, device="cuda")
    res = nq.factorize_admm_nanoquant(W, i_n, o_n, mid_rank=r, outer_iters=400, inner_iters=5,
                                      reg=3e-2, is_transpose=(out_f < in_f), rho_scheduler='linear')
    Wf = res["W_final"].float()
    # what NanoQuantLinear actually computes (do_train=False, 2 scales, no scale_mid)
    sgn = lambda x: torch.where(x >= 0, 1.0, -1.0)
    V  = sgn(res["B"].float())        # (mid, in)
    U  = sgn(res["A"].float().mT)     # (out, mid)
    sp = res["scale_pre"].float()     # (1, in)
    so = res["scale_post"].float()    # (1, out)
    W_bin = ((U @ (V * sp)) * so.mT)
    # and the do_train=True init (latent signs)
    Vl = sgn(res["B_latent"].float()); Ul = sgn(res["A_latent"].float().mT)
    W_lat = ((Ul @ (Vl * sp)) * so.mT)
    n = W.float().square().sum()
    rel = lambda X: ((X - W.float()).square().sum()/n).item()
    print(f"{key.split('.',1)[1]:28s} r={r:5d}  W_final(continuous)={rel(Wf):.4f}  "
          f"binarized(deployed)={rel(W_bin):.4f}  latent-init={rel(W_lat):.4f}")
