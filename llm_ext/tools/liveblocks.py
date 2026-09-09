#!/usr/bin/env python3
"""Live per-block table from the two running arm logs."""
import os, re, sys, os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "q06_tuned_%s.log")
arms = ["nanoquant", "cov"]
d = {}
for a in arms:
    err, ppl = {}, {}
    try:
        txt = open(LOG % a, errors="ignore").read()
    except FileNotFoundError:
        txt = ""
    for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)", txt):
        err[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)", txt):
        ppl[int(m.group(1))] = float(m.group(2))
    d[a] = (err, ppl)
blocks = sorted(set(d["nanoquant"][0]) | set(d["cov"][0]) | set(d["nanoquant"][1]) | set(d["cov"][1]))
print(f"{'blk':>3} | {'diagonal err':>12} {'cov err':>9} {'Δ%':>7} | {'diagonal ppl':>12} {'cov ppl':>9} {'Δ ppl':>8}")
print("-" * 74)
for b in blocks:
    en, pn = d["nanoquant"][0].get(b), d["nanoquant"][1].get(b)
    ec, pc = d["cov"][0].get(b), d["cov"][1].get(b)
    de = f"{(ec-en)/en*100:+.1f}" if (en and ec) else "-"
    dp = f"{pc-pn:+.3f}" if (pn and pc) else "-"
    f = lambda v, w, p: (f"{v:{w}.{p}f}" if v is not None else f"{'-':>{w}}")
    print(f"{b:>3} | {f(en,12,4)} {f(ec,9,4)} {de:>7} | {f(pn,12,3)} {f(pc,9,3)} {dp:>8}")
print("-" * 74)
print(f"FP reference PPL 12.669 | blocks done: diagonal {len(d['nanoquant'][1])}, cov {len(d['cov'][1])} of 28")
