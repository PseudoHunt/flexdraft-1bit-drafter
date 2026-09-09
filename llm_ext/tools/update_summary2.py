#!/usr/bin/env python3
"""Add E16 (delayed finalisation) and E17 (refreshed statistics) rows + finding 4 to SUMMARY.md from logs/JSON."""
import os, json, os, re
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def load(tag, r):
    f=f"{R}/results/{tag}_r{r}.json"
    if os.path.exists(f): return {b["block"]:b["ppl"] for b in json.load(open(f))["block_stats"]}
    lf=f"{R}/logs/{tag}_r{r}.log"
    if not os.path.exists(lf): return {}
    return {int(m.group(1)):float(m.group(2)) for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)", open(lf,errors="ignore").read())}
def col(tag, rs, b): v=[load(tag,r).get(b) for r in rs]; v=[x for x in v if x]; return " / ".join(f"{x:.2f}" for x in v) if v else "-"
ref3="16.06 / 19.27 / 17.87 / 20.7"; ref5="16.55 / 19.00 / 18.52"
d3=col("q06_delay",(1,2),3); d5=col("q06_delay",(1,2),5); r2=col("q06_refresh",(1,2,3,4),2); r3=col("q06_refresh",(1,2,3,4),3); r5=col("q06_refresh",(1,2,3,4),5); r9=col("q06_refresh",(3,4),9)
import json as _j
_fin=[];_pre=[]
for _r in (1,2):
    _ff=f"{R}/results/q06_refresh_full_r{_r}.json"
    if os.path.exists(_ff): _d=_j.load(open(_ff)); _fin.append(_d['eval']['wikitext2']['ppl']); _pre.append([b for b in _d['block_stats'] if b['block']==27][0]['ppl'])
    else:
        _l=load("q06_refresh_full",_r)
        if 27 in _l: _pre.append(_l[27])
full=(" / ".join(f"{x:.2f}" for x in _pre)+" pre-KD" if _pre else "in progress")+(" / **"+" / ".join(f"{x:.2f}" for x in _fin)+" post-KD**" if _fin else "")
s=open(f"{R}/SUMMARY.md").read()
if "| E16 |" not in s:
    s=s.replace("| E15 |", "| E16 | **delayed finalisation** (agenda idea 2, `--delay_finalize`): latents of all projections stay live until the block is binary, then a joint pass | 128, blocks 0–5 | block 0 ≈24 vs 18.1 (v1 and v2, 3 runs); block 3 %s vs %s | §7 |\n| E17 | **refreshed statistics** (agenda idea 22, `--refresh_stats`): `i_norm` re-estimated per block on the compressed-prefix inputs | 128, blocks 0–5 / 0–9 | block 2 %s vs 18.2–18.9; **block 3 %s vs %s**; block 5 %s vs %s; block 9 %s vs 17.24 / 18.90; **full model (2 runs) %s** | §7 |\n| E15 |" % (d3, ref3, r2, r3, ref3, r5, ref5, r9, full), 1)
else:
    s=re.sub(r"\| E16 \|[^\n]*", "| E16 | **delayed finalisation** (agenda idea 2, `--delay_finalize`): latents of all projections stay live until the block is binary, then a joint pass | 128, blocks 0–5 | block 0 ≈24 vs 18.1 (v1 and v2, 3 runs); block 3 %s vs %s | §7 |" % (d3, ref3), s)
    s=re.sub(r"\| E17 \|[^\n]*", "| E17 | **refreshed statistics** (agenda idea 22, `--refresh_stats`): `i_norm` re-estimated per block on the compressed-prefix inputs | 128, blocks 0–5 / 0–9 | block 2 %s vs 18.2–18.9; **block 3 %s vs %s**; block 5 %s vs %s; block 9 %s vs 17.24 / 18.90; **full model (2 runs) %s** | §7 |" % (r2, r3, ref3, r5, ref5, r9, full), s)
f4="""
**4. Re-estimating the input statistics on the compressed prefix removes most of the block-3 problem at 128 samples.**
(§7, last hour) NanoQuant's `i_norm` (and Σ) are collected once on the FP chain; blocks 1–2 are where the massive-activation
channels form, so the cached statistics are most wrong exactly where the tuner sees drifted inputs. One extra forward pass per
block to re-estimate `i_norm` on the block's actual inputs (`--refresh_stats`) gives block 2 PPL %s (uniform 18.2–18.9)
and block 3 PPL %s (uniform 16.1–20.7) at 128 samples — where only 4× more data had reached before. Full model (two runs, NanoQuant 8/8/8 at 128 samples): %s. It also suggests the covariance objective was handicapped by stale Σ.
""" % (r2, r3, full)
if "**4. Re-estimating the input statistics" not in s:
    s=s.replace("## Experiment ledger", f4.strip()+"\n\n## Experiment ledger",1)
else:
    s=re.sub(r"\*\*4\. Re-estimating the input statistics.*?\n\n## Experiment ledger", f4.strip()+"\n\n## Experiment ledger", s, flags=re.S)
open(f"{R}/SUMMARY.md","w").write(s); print("summary: E16/E17 + finding 4 written;", {"delay@3":d3,"refresh@2":r2,"refresh@3":r3,"refresh@5":r5,"refresh@9":r9})
