import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_common import load_admm_nq
nq = load_admm_nq()

def rank_for(a, b, bits=1.0):   # a=in_features, b=out_features  (NanoQuant calculate_ranks)
    r = (a * b * bits) / (a + b) - 16
    r = (int(r) // 32) * 32
    r = 32 if r == 0 else max(r, 32)
    return min(r, min(a, b))

for (out_f, in_f, name) in [(4096,4096,"q_proj"), (1024,4096,"k_proj")]:
    r = rank_for(in_f, out_f)
    W = (torch.randn(out_f, in_f, device="cuda", dtype=torch.bfloat16) * 0.02)
    i_n = torch.rand(in_f, device="cuda") + 0.5
    o_n = torch.rand(out_f, device="cuda") + 0.5
    for iters in (25,):
        torch.cuda.synchronize(); t=time.time()
        res = nq.factorize_admm_nanoquant(W, i_n, o_n, mid_rank=r, outer_iters=iters,
              inner_iters=5, reg=3e-2, is_transpose=(out_f<in_f), rho_scheduler='linear')
        torch.cuda.synchronize(); dt=time.time()-t
        bpw = (r+16)*(in_f+out_f)/(in_f*out_f)
        print(f"{name} {out_f}x{in_f} rank={r} bpw={bpw:.4f} {iters} iters -> {dt:.1f}s "
              f"(=> 400 iters ~ {dt/iters*400:.0f}s)  peakGB={torch.cuda.max_memory_allocated()/2**30:.1f}")
        torch.cuda.reset_peak_memory_stats()
