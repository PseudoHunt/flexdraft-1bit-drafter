#!/usr/bin/env python3
"""Block-3 screen: per-block PPL and block error, baselines (full-run JSON) + live variant logs."""
import os, re, json, os, sys
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cols = {}
for a, lab in [("nanoquant", "diag"), ("cov", "cov")]:
    d = json.load(open(f"{R}/results/q06_tuned_{a}.json"))
    cols[lab] = {b["block"]: (b["block_out_err"], b["ppl"]) for b in d["block_stats"] if b["block"] < NB}
for v in ["diag_rep", "cov_b0", "cov_b25", "cov_b50", "cov_b75", "cov_attn", "cov_oproj", "cov_mlp"]:
    f = f"{R}/logs/q06_b3_{v}.log"
    err, ppl = {}, {}
    if os.path.exists(f):
        t = open(f, errors="ignore").read()
        for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)", t): err[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)", t): ppl[int(m.group(1))] = float(m.group(2))
    jf = f"{R}/results/q06_b3_{v}.json"          # finished run: prefer the JSON (logs can be clobbered)
    if os.path.exists(jf):
        d = json.load(open(jf))
        for b in d["block_stats"]:
            err[b["block"]] = b["block_out_err"]; ppl[b["block"]] = b.get("ppl")
    cols[v.replace("cov_", "").replace("diag_rep", "diagR")] = {b: (err.get(b), ppl.get(b)) for b in set(err) | set(ppl)}
# beta-search run: test PPL of the chosen candidate, annotated with the chosen beta
f = f"{R}/logs/q06_b3_betasearch.log"; chosen = {}; err, ppl = {}, {}
if os.path.exists(f):
    t = open(f, errors="ignore").read()
    for m in re.finditer(r"Block (\d+): chosen beta=([\d.]+)", t): chosen[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)", t): err[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)", t): ppl[int(m.group(1))] = float(m.group(2))
cols["search"] = {b: (err.get(b), ppl.get(b)) for b in set(err) | set(ppl)}
names = list(cols)
def table(idx, title, fmt):
    print(title)
    print(f"{'blk':>3} |" + "".join(f"{n:>9}" for n in names))
    print("-" * (5 + 9 * len(names)))
    for b in range(NB):
        row = f"{b:>3} |"
        for n in names:
            v = cols[n].get(b, (None, None))[idx]
            row += f"{fmt % v:>9}" if v is not None else f"{'-':>9}"
        print(row)
table(1, "wikitext2 PPL after block b  (baselines from the full runs; b0 = plain cov replicate)", "%.3f")
print()
table(0, "relative block output error (calibration set)", "%.4f")
done = {n: sum(1 for b in cols[n] if cols[n][b][1] is not None) for n in names[2:]}
print("\nblocks done:", ", ".join(f"{n} {k}" for n, k in done.items()), f"of {NB}")
if chosen: print("beta-search chosen beta per block:", " ".join(f"b{b}={v:g}" for b, v in sorted(chosen.items())))
