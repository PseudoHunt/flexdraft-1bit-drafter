#!/usr/bin/env python3
"""Markdown tables for RESULTS.md sections 3-4 from llm_ext/results/*.json."""
import os, json, os, statistics as st
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = lambda n: json.load(open(f"{R}/{n}.json"))
FP = 12.669
def pb(d): return {b["block"]: b for b in d["block_stats"]}
out = {}

# --- section 3: four full runs ---
runs = {"diag #1": J("q06_tuned_nanoquant"), "cov #1": J("q06_tuned_cov"),
        "diag #2": J("q06_tuned_rep_nanoquant"), "cov #2": J("q06_tuned_rep_cov")}
rows = ["| run | wikitext2 PPL | Δ vs FP | pre-KD PPL | KD gain | block 3 PPL | mean weight err | mean Σ-out err |", "|---|---|---|---|---|---|---|---|"]
for k, d in runs.items():
    p = d["eval"]["wikitext2"]["ppl"]; pre = pb(d)[27]["ppl"]; b3 = pb(d)[3]["ppl"]
    rows.append(f"| {k} | **{p:.3f}** | +{p-FP:.2f} | {pre:.3f} | {pre-p:.2f} | {b3:.3f} | {d['mean_weight_err']:.4f} | {d['mean_out_err']:.4f} |")
out["full"] = "\n".join(rows)
dg = [runs["diag #1"]["eval"]["wikitext2"]["ppl"], runs["diag #2"]["eval"]["wikitext2"]["ppl"]]
cv = [runs["cov #1"]["eval"]["wikitext2"]["ppl"], runs["cov #2"]["eval"]["wikitext2"]["ppl"]]
out["stats"] = dict(dmean=st.mean(dg), cmean=st.mean(cv), dspread=abs(dg[0]-dg[1]), cspread=abs(cv[0]-cv[1]),
                    gaps=[cv[0]-dg[0], cv[1]-dg[1]])
# per-block replicate table (compact: every block)
p1, c1, p2, c2 = (pb(runs[k]) for k in runs)
rows = ["| block | diag #1 | cov #1 | diag #2 | cov #2 | cov−diag #1 | cov−diag #2 |", "|---|---|---|---|---|---|---|"]
for b in range(28):
    rows.append(f"| {b} | {p1[b]['ppl']:.3f} | {c1[b]['ppl']:.3f} | {p2[b]['ppl']:.3f} | {c2[b]['ppl']:.3f} | {c1[b]['ppl']-p1[b]['ppl']:+.3f} | {c2[b]['ppl']-p2[b]['ppl']:+.3f} |")
out["rep_blocks"] = "\n".join(rows)

# --- section 4a: block-3 screen ---
scr = {"diag #1": pb(runs["diag #1"]), "cov #1": pb(runs["cov #1"]), "diag rep": pb(J("q06_b3_diag_rep")),
       "cov rep (β=0)": pb(J("q06_b3_cov_b0")), "β=0.25": pb(J("q06_b3_cov_b25")), "β=0.5": pb(J("q06_b3_cov_b50")),
       "β=0.75": pb(J("q06_b3_cov_b75")), "attn only": pb(J("q06_b3_cov_attn")), "o_proj only": pb(J("q06_b3_cov_oproj")),
       "MLP only": pb(J("q06_b3_cov_mlp"))}
if os.path.exists(f"{R}/q06_b3_betasearch.json"): scr["β search"] = pb(J("q06_b3_betasearch"))
names = list(scr)
rows = ["| block | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for b in range(10):
    rows.append(f"| {b} | " + " | ".join(f"{scr[n][b]['ppl']:.2f}" if b in scr[n] and scr[n][b].get('ppl') else "-" for n in names) + " |")
rows.append("| err@3 | " + " | ".join(f"{scr[n][3]['block_out_err']:.4f}" if 3 in scr[n] else "-" for n in names) + " |")
out["screen"] = "\n".join(rows)

# --- section 4b: beta search candidates ---
if "β search" in scr:
    d = J("q06_b3_betasearch"); bs = pb(d)
    rows = ["| block | β=0 | β=0.25 | β=0.5 | β=0.75 | β=1 (diag) | chosen | test PPL |", "|---|---|---|---|---|---|---|---|"]
    for b in range(10):
        c = {x["beta"]: x["val_ppl"] for x in bs[b]["candidates"]}; ch = bs[b]["chosen_beta"]
        cells = [(f"**{c[k]:.2f}**" if k == ch else f"{c[k]:.2f}") for k in (0, 0.25, 0.5, 0.75, 1)]
        rows.append(f"| {b} | " + " | ".join(cells) + f" | {ch:g} | {bs[b]['ppl']:.3f} |")
    out["search"] = "\n".join(rows)
    out["search_final"] = d["eval"]["wikitext2"]["ppl"]
    out["search_chosen"] = [bs[b]["chosen_beta"] for b in range(10)]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.json"), "w"), indent=1)
print(out["full"]); print(); print(out["stats"]); print(); print(out["screen"]); print("search:", "search" in out)
