#!/usr/bin/env python
"""Sensitivity-aware per-block bit multipliers for NanoQuant's rank allocation (RESULTS.md 4.5 / idea H).

  python llm_ext/block_bits.py --ref llm_ext/results/q06_tuned_nanoquant.json --gamma 0.15

Signal: log of the per-block reconstruction error of a reference run (accumulated drift vs the FP chain, blocks with
large error are where the model is hard to compress).  mult_i = exp(gamma * z_i), z = standardised log-error, then a
global scale is searched so the REALISED bpw (after NanoQuant's rank rounding to multiples of 32) equals the uniform
allocation's.  Prints the multiplier string for --block_bits and the per-block ranks.
"""
import argparse, json, math
SHAPES = {"q_proj": (1024, 2048), "k_proj": (1024, 1024), "v_proj": (1024, 1024), "o_proj": (2048, 1024),
          "gate_proj": (1024, 3072), "up_proj": (1024, 3072), "down_proj": (3072, 1024)}   # Qwen3-0.6B (in, out)
def rank_for(a, b, bits):
    r = int((a * b * bits) / (a + b) - 16); r = (r // 32) * 32
    r = max(r, 32); return min(r, min(a, b))
def bpw(mults, bits):
    tot = par = 0
    for m in mults:
        for a, b in SHAPES.values():
            r = rank_for(a, b, bits * m); tot += r * (a + b) + 16 * (a + b); par += a * b
    return tot / par
ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--gamma", type=float, default=0.15)
ap.add_argument("--bits", type=float, default=1.0); ap.add_argument("--smooth", type=int, default=1, help="moving-average window on log err")
args = ap.parse_args()
d = json.load(open(args.ref)); bs = {b["block"]: b["block_out_err"] for b in d["block_stats"]}; n = len(bs)
le = [math.log(bs[i]) for i in range(n)]
if args.smooth > 1:
    w = args.smooth; le = [sum(le[max(0, i - w // 2):i + w // 2 + 1]) / len(le[max(0, i - w // 2):i + w // 2 + 1]) for i in range(n)]
mu = sum(le) / n; sd = (sum((x - mu) ** 2 for x in le) / n) ** 0.5
z = [(x - mu) / sd for x in le]
base = [math.exp(args.gamma * zi) for zi in z]
target = bpw([1.0] * n, args.bits)
lo, hi = 0.5, 2.0
for _ in range(60):                                    # scale so realised bpw matches uniform
    mid = (lo + hi) / 2; m = [mid * x for x in base]
    if bpw(m, args.bits) > target: hi = mid
    else: lo = mid
m = [lo * x for x in base]
print("uniform bpw %.4f   allocated bpw %.4f   gamma %.2f" % (target, bpw(m, args.bits), args.gamma))
print("mults:", ",".join("%.4f" % x for x in m))
print("bits per block:", " ".join("%.2f" % (args.bits * x) for x in m))
print("q_proj rank per block:", " ".join(str(rank_for(1024, 2048, args.bits * x)) for x in m))
print("down_proj rank per block:", " ".join(str(rank_for(3072, 1024, args.bits * x)) for x in m))
