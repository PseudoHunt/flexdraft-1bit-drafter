#!/usr/bin/env python3
"""RESULTS.md section 7: delayed finalisation (agenda idea 2) and refreshed statistics (idea 22), blocks 0-5 @128."""
import os, json, os, re
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def load(tag, r):
    f=f"{R}/results/{tag}_r{r}.json"
    if os.path.exists(f): return {b["block"]:b for b in json.load(open(f))["block_stats"]}
    t=open(f"{R}/logs/{tag}_r{r}.log",errors="ignore").read() if os.path.exists(f"{R}/logs/{tag}_r{r}.log") else ""
    err={int(m.group(1)):float(m.group(2)) for m in re.finditer(r"Block (\d+): relative block output error = ([\d.eE+-]+)",t)}
    ppl={int(m.group(1)):float(m.group(2)) for m in re.finditer(r"Block (\d+): Test Data PPL\s+= ([\d.eE+-]+)",t)}
    return {b:{"block_out_err":err.get(b),"ppl":ppl.get(b)} for b in set(err)|set(ppl)}
ref={"diag #1":{b["block"]:b for b in json.load(open(f"{R}/results/q06_tuned_nanoquant.json"))["block_stats"]},
     "diag #2":{b["block"]:b for b in json.load(open(f"{R}/results/q06_tuned_rep_nanoquant.json"))["block_stats"]},
     "diag rep":{b["block"]:b for b in json.load(open(f"{R}/results/q06_b3_diag_rep.json"))["block_stats"]}}
cols=dict(ref); cols["delayed #1"]=load("q06_delay",1); cols["delayed #2"]=load("q06_delay",2); cols["refresh #1"]=load("q06_refresh",1); cols["refresh #2"]=load("q06_refresh",2); cols["refresh #3"]=load("q06_refresh",3); cols["refresh #4"]=load("q06_refresh",4); cols["refresh full #1"]=load("q06_refresh_full",1); cols["refresh full #2"]=load("q06_refresh_full",2)
names=list(cols); g=lambda c,b,k: (f"{cols[c][b][k]:.3f}" if b in cols[c] and cols[c][b].get(k) is not None else "-")
rows=["| block | "+" | ".join(names)+" |","|---|"+"---|"*len(names)]
for b in range(28): rows.append(f"| {b} | "+" | ".join(g(c,b,"ppl") for c in names)+" |")
rows.append("| err@3 | "+" | ".join(g(c,3,"block_out_err") for c in names)+" |")
tbl="\n".join(rows)
done={c:sum(1 for b in cols[c] if cols[c][b].get("ppl")) for c in names[3:]}
fin=[]; pre=[]
for r in (1,2):
    ff=f"{R}/results/q06_refresh_full_r{r}.json"
    if os.path.exists(ff):
        d=json.load(open(ff)); bb={b["block"]:b for b in d["block_stats"]}; fin.append(d["eval"]["wikitext2"]["ppl"]); pre.append(bb[27]["ppl"])
    elif 27 in cols[f"refresh full #{r}"] and cols[f"refresh full #{r}"][27].get("ppl"): pre.append(cols[f"refresh full #{r}"][27]["ppl"])
j=lambda v:" / ".join(f"{x:.3f}" for x in v) if v else "pending"
full_line=f"\n**Full model, refreshed statistics, 128 samples, NanoQuant 8/8/8: pre-KD {j(pre)}, post-KD {j(fin)}** (uniform 128, same setting: 32.57 / 36.94 pre-KD, 28.10 / 33.69 post-KD; paper 27.56; allocation-512 24.25 / 23.97)."
sec=f"""## 7. Two quick screens from the research agenda (blocks 0–5 / 0–9, 128 samples, NanoQuant epochs)

Run in the last GPU hour with one or two replicates each; both are **first signals**, not results. Reference columns are
the three uniform diagonal chains at the same setting (block-3 draws 16.06 / 19.27 / 17.87 in the full table).

* **Delayed finalisation** (agenda idea 2, `--delay_finalize`): every projection of a block keeps its latent sign
  factors after its own `tune_fact`; once the whole block is binary, one joint pass over all latents (`--joint_epochs 2`,
  i.e. *extra* compute — not the agenda's equal-compute control), then all are finalised. Tests whether letting later
  quantisation revise earlier sign decisions changes where block 3 lands. A first version left gradients from
  `tune_nonfact` accumulating on the live latents (its optimizer only zeroes the FP weights): block-0 PPL 24.9 / 24.6 vs
  ≈18.1 (`logs/q06_delay_v1_*`); `tune_fact` now zeroes the block's gradients first. With the fix, per-layer
  `tune_fact` of projection *k* also updates the live latents of projections < *k* — continuous revision, as the idea
  intends — followed by the joint pass.
* **Refreshed statistics** (agenda idea 22, `--refresh_stats`): per block, `i_norm` of its linears is re-estimated from
  the block's *actual* (compressed-prefix) inputs — raw input second moments with NanoQuant's shrinkage — instead of the
  FP-chain cache (`o_norm` is left as cached: it comes from NanoQuant's backward hook, not from output energy; a first
  version that also overwrote it gave block-0 PPL 27.3 and was discarded). Block 0 has no upstream drift, so it is the
  control: refreshed ≈ cached there. Tests whether stale statistics are why upstream drift hurts.

{tbl}

Blocks completed: {", ".join(f"{k} {v}" for k,v in done.items())} (runs 1–2: blocks 0–5; runs 3–4 stopped at 15:04 to free the card for the full run).
{full_line}
"""
p=f"{R}/RESULTS.md"; s=open(p).read()
if "## 7. Two quick screens" in s: a=s.index("## 7. Two quick screens"); b=s.index("## Cost"); s=s[:a]+sec+"\n"+s[b:]
else: i=s.index("## Cost"); s=s[:i]+sec+"\n"+s[i:]
s=s.replace("§3–6 on one H200","§3–7 on one H200"); open(p,"w").write(s); print("section 7 written:", done)
