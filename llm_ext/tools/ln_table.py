#!/usr/bin/env python3
import os, re, json, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def fj(f): d=json.load(open(f)); return {b["block"]:(b["block_out_err"],b.get("ppl")) for b in d["block_stats"] if b["block"]<10}
def fl(f):
    e,p={},{}
    if os.path.exists(f):
        t=open(f,errors="ignore").read()
        for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)",t): e[int(m.group(1))]=float(m.group(2))
        for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)",t): p[int(m.group(1))]=float(m.group(2))
    return {b:(e.get(b),p.get(b)) for b in set(e)|set(p)}
cols={"o_norm#1":fj(f"{R}/results/q06_n512_nanoquant_r1.json"),"o_norm#2":fj(f"{R}/results/q06_n512_nanoquant_r2.json")}
for r in (1,2):
    jf=f"{R}/results/q06_ln_inv_var_r{r}.json"; cols[f"inv_var#{r}"]=fj(jf) if os.path.exists(jf) else fl(f"{R}/logs/q06_ln_inv_var_r{r}.log")
n=list(cols); print("diagonal, 512 samples, matched steps: NanoQuant o_norm loss vs inv_var (per-channel normalised) loss -- PPL after block b | block err")
print(f"{'blk':>3} |"+"".join(f"{x:>10}" for x in n)+" |"+"".join(f"{x:>10}" for x in n)); print("-"*(5+20*len(n)))
for b in range(10): print(f"{b:>3} |"+"".join((f"{cols[x][b][1]:10.3f}" if b in cols[x] and cols[x][b][1] else f"{'-':>10}") for x in n)+" |"+"".join((f"{cols[x][b][0]:10.4f}" if b in cols[x] and cols[x][b][0] else f"{'-':>10}") for x in n))
print("blocks done:", ", ".join(f"{x} {sum(1 for b in cols[x] if cols[x][b][1])}" for x in n[2:]))
