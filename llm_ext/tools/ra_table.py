#!/usr/bin/env python3
import os, re, json, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def fj(f): d=json.load(open(f)); return {b["block"]:(b["block_out_err"],b.get("ppl")) for b in d["block_stats"]}, d["eval"]["wikitext2"]["ppl"], d.get("alloc",{}).get("bpw")
def fl(f):
    e,p={},{}
    if os.path.exists(f):
        t=open(f,errors="ignore").read()
        for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)",t): e[int(m.group(1))]=float(m.group(2))
        for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)",t): p[int(m.group(1))]=float(m.group(2))
        b=re.search(r"factorized bpw = ([\d.]+)",t)
    return {b_:(e.get(b_),p.get(b_)) for b_ in set(e)|set(p)}, None, (float(b.group(1)) if b else None)
cols={}; fin={}; bpw={}
cols["d128#1"],fin["d128#1"],bpw["d128#1"]=fj(f"{R}/results/q06_tuned_nanoquant.json"); cols["d128#2"],fin["d128#2"],bpw["d128#2"]=fj(f"{R}/results/q06_tuned_rep_nanoquant.json")
for v in ("uniform","alloc"):
    for r in (1,2):
        k=f"{v[:4]}512#{r}"; jf=f"{R}/results/q06_ra_{v}_r{r}.json"
        cols[k],fin[k],bpw[k]=fj(jf) if os.path.exists(jf) else fl(f"{R}/logs/q06_ra_{v}_r{r}.log")
for r in (1,2):
    k=f"g.3#{r}"; jf=f"{R}/results/q06_ra3_alloc_r{r}.json"
    cols[k],fin[k],bpw[k]=fj(jf) if os.path.exists(jf) else fl(f"{R}/logs/q06_ra3_alloc_r{r}.log")
for r in (1,2):
    k=f"g.15@128#{r}"; jf=f"{R}/results/q06_ra128_alloc_r{r}.json"
    cols[k],fin[k],bpw[k]=fj(jf) if os.path.exists(jf) else fl(f"{R}/logs/q06_ra128_alloc_r{r}.log")
n=list(cols)
print("PPL after block b: 128 uniform (2) | 512 uniform (2) | 512 allocated gamma=0.15 (2) | 512 allocated gamma=0.3 (2); equal bpw")
print(f"{'blk':>3} |"+"".join(f"{x[-9:]:>10}" for x in n)); print("-"*(5+9*len(n)))
for b in range(28):
    row=[cols[x].get(b,(None,None))[1] for x in n]
    if any(v is not None for v in row[2:]) or b<1: print(f"{b:>3} |"+"".join((f"{v:10.3f}" if v else f"{'-':>10}") for v in row))
print("bpw:", {k:(round(v,4) if v else None) for k,v in bpw.items() if k not in ('d128#1','d128#2')})
print("FINAL post-KD PPL:", {k:(round(v,3) if v else None) for k,v in fin.items()})
print("blocks done:", ", ".join(f"{x} {sum(1 for b in cols[x] if cols[x][b][1])}" for x in n[2:]))
