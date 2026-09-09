#!/usr/bin/env python3
"""Patch SUMMARY.md finding 3 and ledger rows E13-E15 from the rank-allocation JSONs (leaves 'in progress' if absent)."""
import os, json, os, re, statistics as st
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def fin(n):
    f=f"{R}/results/{n}.json"; return json.load(open(f))["eval"]["wikitext2"]["ppl"] if os.path.exists(f) else None
def pair(a,b): return f"{st.mean([a,b]):.2f} ± {abs(a-b)/2:.2f}"
u=[fin("q06_ra_uniform_r1"),fin("q06_ra_uniform_r2")]; a=[fin("q06_ra_alloc_r1"),fin("q06_ra_alloc_r2")]
g3=[fin("q06_ra3_alloc_r1"),fin("q06_ra3_alloc_r2")]; a128=[fin("q06_ra128_alloc_r1"),fin("q06_ra128_alloc_r2")]
s=open(f"{R}/SUMMARY.md").read()
def fmt(v): return " / ".join(f"{x:.2f}" for x in v if x is not None) + ("" if all(v) else " (1 of 2 done)") if any(v) else None
if all(u) and all(a):
    s=re.sub(r"\| E13 \|[^\n]*", f"| E13 | **sensitivity-aware rank allocation** γ=0.15 vs uniform, full model, 2 runs each (`--block_bits`) | 512 | pre-KD 28.23 / 28.54 vs 30.13 / 32.71; **post-KD {fmt(a)} vs {fmt(u)}** | §6 |", s)
    s=s.replace("Post-KD finals,\na γ=0.3 pair, and a pair at NanoQuant's unmodified 128-sample defaults are running at the time of writing (see\n\"In progress\" below); §6 is regenerated when they land.",
                f"**Post-KD: allocated {pair(*a)} vs uniform {pair(*u)}** (wikitext2 PPL, two replicates each, equal bpw).")
if any(g3): s=re.sub(r"\| E14 \|[^\n]*", f"| E14 | rank allocation γ=0.3, 2 runs | 512 | post-KD {fmt(g3)} | §6 |", s)
if any(a128): s=re.sub(r"\| E15 \|[^\n]*", f"| E15 | rank allocation γ=0.15 at NanoQuant's exact defaults, 2 runs | 128, 8/8/8 | post-KD {fmt(a128)} vs uniform 28.10 / 33.69 (paper: 27.56, n=1) | §6 |", s)
open(f"{R}/SUMMARY.md","w").write(s); print("summary patched; finals:", {"uniform":u,"g0.15":a,"g0.3":g3,"g0.15@128":a128})
