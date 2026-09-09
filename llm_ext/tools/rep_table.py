#!/usr/bin/env python3
"""Replicate full runs vs originals: per-block PPL (and err) for diag/cov, run 1 vs run 2."""
import os, re, json, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def from_json(f):
    d = json.load(open(f)); return {b["block"]: (b["block_out_err"], b.get("ppl")) for b in d["block_stats"]}
def from_log(f):
    err, ppl = {}, {}
    if os.path.exists(f):
        t = open(f, errors="ignore").read()
        for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)", t): err[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)", t): ppl[int(m.group(1))] = float(m.group(2))
    return {b: (err.get(b), ppl.get(b)) for b in set(err) | set(ppl)}
cols = {"diag#1": from_json(f"{R}/results/q06_tuned_nanoquant.json"), "cov#1": from_json(f"{R}/results/q06_tuned_cov.json")}
for a, lab in [("nanoquant", "diag#2"), ("cov", "cov#2")]:
    jf = f"{R}/results/q06_tuned_rep_{a}.json"
    cols[lab] = from_json(jf) if os.path.exists(jf) else from_log(f"{R}/logs/q06_tuned_rep_{a}.log")
names = list(cols)
print(f"{'blk':>3} |" + "".join(f"{n:>9}" for n in names) + f"{'d2-d1':>8}{'c2-c1':>8}{'c-d #1':>8}{'c-d #2':>8}")
print("-" * (5 + 9 * 4 + 32))
for b in range(28):
    v = [cols[n].get(b, (None, None))[1] for n in names]
    row = f"{b:>3} |" + "".join(f"{x:9.3f}" if x is not None else f"{'-':>9}" for x in v)
    d1, c1, d2, c2 = v
    row += f"{(d2-d1):+8.3f}" if (d1 and d2) else f"{'-':>8}"
    row += f"{(c2-c1):+8.3f}" if (c1 and c2) else f"{'-':>8}"
    row += f"{(c1-d1):+8.3f}" if (c1 and d1) else f"{'-':>8}"
    row += f"{(c2-d2):+8.3f}" if (c2 and d2) else f"{'-':>8}"
    print(row)
for a in ("nanoquant", "cov"):
    jf = f"{R}/results/q06_tuned_rep_{a}.json"
    if os.path.exists(jf):
        d = json.load(open(jf)); print(f"FINAL {a} replicate: test PPL {d['eval']['wikitext2']['ppl']:.3f}  (run 1: "
              f"{json.load(open(f'{R}/results/q06_tuned_{a}.json'))['eval']['wikitext2']['ppl']:.3f})")
