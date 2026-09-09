#!/usr/bin/env python3
"""RESULTS.md section 6 (rank allocation) + short negatives (5.1 grad-accum, 5.2 inv_var loss) from JSONs/logs."""
import os, json, os, re, statistics as st
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def load(name):
    f=f"{R}/results/{name}.json"
    if os.path.exists(f):
        d=json.load(open(f)); return {b["block"]:b for b in d["block_stats"]}, d["eval"]["wikitext2"]["ppl"], d["alloc"]["bpw"]
    f=f"{R}/results/{name}_preKD.json"
    if os.path.exists(f):
        d=json.load(open(f)); return {b["block"]:b for b in d["block_stats"]}, None, d["alloc"]["bpw"]
    return None, None, None
runs={"uniform #1":load("q06_ra_uniform_r1"),"uniform #2":load("q06_ra_uniform_r2"),"γ=0.15 #1":load("q06_ra_alloc_r1"),"γ=0.15 #2":load("q06_ra_alloc_r2"),
      "γ=0.3 #1":load("q06_ra3_alloc_r1"),"γ=0.3 #2":load("q06_ra3_alloc_r2"),
      "uniform@128 #1":load("q06_tuned_nanoquant"),"uniform@128 #2":load("q06_tuned_rep_nanoquant"),
      "γ=0.15@128 #1":load("q06_ra128_alloc_r1"),"γ=0.15@128 #2":load("q06_ra128_alloc_r2")}
runs={k:v for k,v in runs.items() if v[0]}
names=list(runs)
bits15=[1.2543,1.1569,0.9645,0.8188,0.7585,0.7790,0.7988,0.8139,0.8277,0.8456,0.8668,0.8861,0.9017,0.9121,0.9284,0.9578,0.9977,1.0398,1.0774,1.1097,1.1363,1.1475,1.1512,1.1458,1.1416,1.1352,1.1506,1.1575]
bits30=[1.5530,1.3213,0.9184,0.6618,0.5679,0.5991,0.6299,0.6540,0.6763,0.7059,0.7417,0.7751,0.8026,0.8213,0.8509,0.9057,0.9827,1.0674,1.1458,1.2157,1.2747,1.2999,1.3083,1.2961,1.2865,1.2722,1.3070,1.3228]
hdr="| run | bpw | pre-KD PPL (block 27) | **post-KD PPL** | KD gain | block 3 | block 16 | block 23 |"
rows=[hdr,"|---|---|---|---|---|---|---|---|"]
for k in names:
    b,fin,bpw=runs[k]; last=max(b); pre=b[last]["ppl"]; g=lambda i: (f"{b[i]['ppl']:.2f}" if i in b else "-")
    pre_s=f"{pre:.3f}" if last==27 else f"({pre:.2f} @ block {last}, stopped)"
    rows.append(f"| {k} | {bpw:.4f} | {pre_s} | {('**%.3f**'%fin) if fin else ('-' if last<27 else 'KD OOM-killed (300 GB cap), pre-KD only')} | {('%.2f'%(pre-fin)) if fin else '-'} | {g(3)} | {g(16)} | {g(23)} |")
summary="\n".join(rows)
blk=["| block | bits γ=.15 | bits γ=.3 | "+" | ".join(names)+" |","|---|---|---|"+"---|"*len(names)]
for i in range(28):
    blk.append(f"| {i} | {bits15[i]:.2f} | {bits30[i]:.2f} | "+" | ".join((f"{runs[k][0][i]['ppl']:.2f}" if i in runs[k][0] else "-") for k in names)+" |")
blocktable="\n".join(blk)
def pair(a,b): return f"{st.mean([a,b]):.2f} ± {abs(a-b)/2:.2f}"
def fin_pair(p): v=[runs[k][1] for k in names if k.startswith(p) and runs[k][1]]; return pair(*v) if len(v)==2 else (f"{v[0]:.2f} (n=1)" if v else "pending")
def pre_pair(p): v=[runs[k][0][27]["ppl"] for k in names if k.startswith(p) and 27 in runs[k][0]]; return pair(*v) if len(v)==2 else (f"{v[0]:.2f} (n=1)" if v else "stopped early")
sec=f"""## 6. Sensitivity-aware rank allocation at equal bpw (idea H) — the first win

NanoQuant gives every block the same rank. §3's per-block curve says the damage is not uniform: block error rises
10× over blocks 15–27 and the last block alone costs +4–5 PPL, while blocks 3–15 sit at 0.002. `block_bits.py`
turns a reference run's per-block error into per-block bit multipliers (`mult_i = exp(γ·z_i)`, z = standardised
log-error, 3-block smoothing), then scales them so the **realised bpw after rank rounding matches the uniform
allocation** (0.9722–0.9726 vs 0.9729 — never above). `--block_bits` applies them in `calculate_ranks`.
Runs: 512 calibration samples, block loop at matched steps (EPOCHS=2), model KD = NanoQuant's default budget (first
128 samples × 8 epochs), diagonal arm, two replicates per setting, all concurrent on the H200; plus two allocated runs at
NanoQuant's unmodified defaults (128 samples, 8/8/8 epochs) against the two uniform runs of §3.

{summary}

| | 512 samples: uniform | γ=0.15 | γ=0.3 | 128 samples (NanoQuant defaults): uniform | γ=0.15 |
|---|---|---|---|---|---|
| pre-KD (block 27) | {pre_pair('uniform #')} | {pre_pair('γ=0.15 #')} | {pre_pair('γ=0.3')} | {pre_pair('uniform@128')} | {pre_pair('γ=0.15@128')} |
| **post-KD** | {fin_pair('uniform #')} | {fin_pair('γ=0.15 #')} | {fin_pair('γ=0.3')} | {fin_pair('uniform@128')} | {fin_pair('γ=0.15@128')} |

The 128-sample columns are NanoQuant's exact published setting (128 × 2048 calibration tokens, 8/8/8 epochs; the paper
reports 27.56 for Qwen3-0.6B at 1 bit, single run).

<details><summary>Per-block PPL, all runs, with the per-block bit budget of each allocation</summary>

{blocktable}

</details>

The γ=0.15 curve is the mechanism in one column: with 0.76–0.93 bits through blocks 4–16 it runs ~0.5–0.8 PPL behind
the better uniform run, then with 1.04–1.16 bits from block 17 it repays the debt by block 23 and finishes 1.9 / 4.2
PPL below the two uniform runs before KD; the last block costs it +3.6/+3.9 instead of +4.9/+5.2. Replicates are
within 0.05 PPL at 24 of 28 blocks. The allocation was derived from a *128-sample* run's error curve and a first-guess
γ; it has not been tuned.

### 6.1 Two levers that did not work (for the record)

* **Gradient accumulation in Step 3** (`--nonfact_batch_size 16 --fact_batch_size 8`, same sample-passes, 128
  samples): block-0 PPL 33.9 / 36.9 vs 18.1. With NanoQuant's learning rates (1e-4 / 1e-5) and a cosine schedule,
  4× fewer optimizer steps under-trains the block badly — Step 3 is *step-count-limited*, which is why batch-1 works.
  Stopped after block 0 (`logs/q06_bs_*`).
* **Per-channel-normalised block loss** (`--loss_norm inv_var`, 512 samples, matched steps): block 0 20.1 / 21.1 vs
  17.7 / 17.0 for NanoQuant's `o_norm` weighting, calibration error higher (0.141 vs 0.132); block 1 level. Taking the
  massive-activation channels out of the loss makes the early blocks worse, not better — those channels evidently need
  to be reconstructed accurately. Stopped after block 1 (`logs/q06_ln_*`).
"""
p=f"{R}/RESULTS.md"; s=open(p).read()
if "## 6. Sensitivity-aware rank allocation" in s:
    a=s.index("## 6. Sensitivity-aware rank allocation"); b=s.index("## Cost"); s=s[:a]+sec+"\n"+s[b:]
else:
    i=s.index("## Cost"); s=s[:i]+sec+"\n"+s[i:]
s=s.replace("§1–2 on one NVIDIA L4-24GB, §3–5 on one H200", "§1–2 on one NVIDIA L4-24GB, §3–6 on one H200")
open(p,"w").write(s); print("section 6 written; runs:", names, "finals:", {k:runs[k][1] for k in names})
