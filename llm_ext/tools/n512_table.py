#!/usr/bin/env python3
"""512-sample screen vs the 128-sample runs, blocks 0-9: PPL after block b and calibration block error."""
import os, re, json, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def from_json(f, nb=10):
    d = json.load(open(f)); return {b["block"]: (b["block_out_err"], b.get("ppl")) for b in d["block_stats"] if b["block"] < nb}
def from_log(f):
    err, ppl = {}, {}
    if os.path.exists(f):
        t = open(f, errors="ignore").read()
        for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)", t): err[int(m.group(1))] = float(m.group(2))
        for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)", t): ppl[int(m.group(1))] = float(m.group(2))
    return {b: (err.get(b), ppl.get(b)) for b in set(err) | set(ppl)}
cols = {"d128#1": from_json(f"{R}/results/q06_tuned_nanoquant.json"), "d128#2": from_json(f"{R}/results/q06_tuned_rep_nanoquant.json"),
        "c128#1": from_json(f"{R}/results/q06_tuned_cov.json"), "c128#2": from_json(f"{R}/results/q06_tuned_rep_cov.json")}
for arm, lab in (("nanoquant", "d512"), ("cov", "c512")):
    for r in (1, 2):
        jf = f"{R}/results/q06_n512_{arm}_r{r}.json"
        cols[f"{lab}#{r}"] = from_json(jf) if os.path.exists(jf) else from_log(f"{R}/logs/q06_n512_{arm}_r{r}.log")
names = list(cols)
def table(idx, title, fmt):
    print(title); print(f"{'blk':>3} |" + "".join(f"{n:>9}" for n in names)); print("-" * (5 + 9 * len(names)))
    for b in range(10):
        print(f"{b:>3} |" + "".join((f"{fmt % cols[n][b][idx]:>9}" if b in cols[n] and cols[n][b][idx] is not None else f"{'-':>9}") for n in names))
table(1, "wikitext2 PPL after block b   (128-sample runs = reference; 512-sample runs at matched optimizer steps)", "%.3f")
print(); table(0, "relative block output error on the run's OWN calibration set (128 vs 512 sequences)", "%.4f")
print("\nblocks done:", ", ".join(f"{n} {sum(1 for b in cols[n] if cols[n][b][1] is not None)}" for n in names[4:]), "of 10")
