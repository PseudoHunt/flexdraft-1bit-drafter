#!/usr/bin/env python
"""Summarise llm_ext/results/*.json: PPL, zero-shot, and the layer-wise ADMM errors per arm."""
import json, sys, glob, os
from collections import defaultdict

files = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results", "*.json")))
runs = {}
for f in files:
    d = json.load(open(f))
    runs[os.path.basename(f)[:-5]] = d

print(f"{'run':<28} {'bpw':>6} {'wiki2 ppl':>10} {'w-err':>7} {'out-err':>8} {'time(min)':>9}")
for k, d in runs.items():
    ppl = d.get("eval", {}).get("wikitext2", {}).get("ppl")
    bpw = d.get("alloc", {}).get("bpw")
    print(f"{k:<28} {('%.3f'%bpw) if bpw else '  fp16':>6} {ppl if ppl is None else round(ppl,3):>10} "
          f"{('%.4f'%d['mean_weight_err']) if d.get('mean_weight_err') else '   -':>7} "
          f"{('%.4f'%d['mean_out_err']) if d.get('mean_out_err') else '    -':>8} "
          f"{d.get('total_time',0)/60:>9.1f}")
    for task, v in (d.get("eval") or {}).items():
        if task == "wikitext2" or not isinstance(v, dict):
            continue
        acc = v.get("acc_norm,none", v.get("acc,none"))
        if acc is not None:
            print(f"    {task:<24} {acc*100:.2f}")

# per-projection breakdown, arm vs arm
by_arm = {}
for k, d in runs.items():
    if not d.get("layer_stats"):
        continue
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for s in d["layer_stats"]:
        a = agg[s["name"]]
        a[0] += s["weight_err"]; a[1] += (s["out_err"] or 0.0); a[2] += 1
    by_arm[k] = {n: (v[0]/v[2], v[1]/v[2]) for n, v in agg.items()}
if len(by_arm) >= 2:
    names = sorted(next(iter(by_arm.values())).keys())
    arms = list(by_arm)
    print("\nper-projection mean over blocks (weight-err | Sigma-weighted output-err)")
    print(f"{'proj':<22}" + "".join(f"{a[-18:]:>34}" for a in arms))
    for n in names:
        row = f"{n:<22}"
        for a in arms:
            w, o = by_arm[a][n]
            row += f"{w:>16.4f} {o:>17.4f}"
        print(row)
