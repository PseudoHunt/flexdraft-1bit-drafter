"""Regenerates RESULTS.md from results/*.json. Usage: python work/gen_results.py > RESULTS.md"""
import json, os, sys, statistics as st
import numpy as np
W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results") + "/"
J = lambda f: json.load(open(W + f))
def paired(x, y):
    d = np.array(x) - np.array(y); s = d.std(ddof=1) / np.sqrt(len(d)); return d.mean(), s, d.mean() / s, int((d > 0).sum())
main, tf, tt, op, o2, o4, ab, al, res, ca, ha, kd, on, g1, g2, cf = [J(f) for f in
    ["main.json", "tf.json", "tt.json", "op.json", "op2x2.json", "op4k.json", "ablate.json", "alloc.json", "res.json", "calib.json",
     "hadamard.json", "kd.json", "onorm.json", "covgate.json", "covgate2.json", "covfull.json"]]
FP = tf["tau_fp"]["tau"]; fpr = tf["tau_fp"]
def row(name, r, ref=None, bpw="", extra=""):
    t = r["tau"]; out = f"| {name} | {bpw} | {t:.3f} ± {r['tau_sem']:.3f} | {100*t/FP:.1f}% |"
    if ref is not None:
        m, s, tt_, b = paired(r["per_sample_tau"], ref["per_sample_tau"]); out += f" {m:+.3f} ± {s:.3f} | {tt_:+.1f} | {b}/40 |"
    return out + extra
P = print
P("# Results\n")
P("All numbers: FlexDraft-Qwen3-8B drafter, Qwen3-8B target, 40 GSM8K test prompts (test[0:40]), block_size 16, temperature 0, draft-confidence-threshold 0.01, `cumulative_product`, max_new_tokens 256. **τ** = mean accepted length per drafting step (`Avg Acceptance Length` in FlexDraft's `inference.py`), micro-averaged over all steps; ± is SEM over steps. Paired statistics use per-prompt τ (n = 40). Ranks: q/o 2016, k/v 800 (NanoQuant `calculate_ranks`, bits=1.0) → 0.993 bpw incl. 16-bit scales; residual base r₂ = 256/96 → 1.127 bpw.\n")
P("Two numerically-different-but-equivalent code paths were used. **Fused** = FlexDraft's `_build_fused_qkv` (FP baseline τ = 7.101). **Unfused** = per-module path required once weights are `NanoQuantLinear` (FP baseline τ = 7.312). Never compare across paths.\n")
P("## Headline ladder (unfused path, % of FP τ = 7.312)\n\n| stage | ADMM-only τ | tuned τ | % FP (best) |\n|---|---|---|---|")
P(f"| NanoQuant ADMM, uniform i_norm | {tf['tau_admm_only']['tau']:.3f} | | {100*tf['tau_admm_only']['tau']/FP:.1f}% |")
P(f"| + activation-calibrated i_norm (diagonal) | {op['admm_gen']['tau']:.3f} | {op['runs']['ce@lat0.0001_sc1e-05']['gen']['tau']:.3f} | {100*op['runs']['ce@lat0.0001_sc1e-05']['gen']['tau']/FP:.1f}% |")
P(f"| + residual binary base (1.13 bpw) | {res['admm_gen']['tau']:.3f} | {res['tuned']['gen']['tau']:.3f} | {100*res['tuned']['gen']['tau']/FP:.1f}% |")
P(f"| **full-covariance ADMM** (base only) | {g2['admm_gen']['tau']:.3f} | | {100*g2['admm_gen']['tau']/FP:.1f}% |")
P(f"| **full-covariance ADMM + residual + target-teacher tuning** | {cf['admm_gen']['tau']:.3f} | **{cf['tuned']['gen']['tau']:.3f}** | **{100*cf['tuned']['gen']['tau']/FP:.1f}%** |\n")

P("## 1. Fused path: ADMM-only with activation-calibrated `i_norm` (`run_all.py`, `main.json`)\n")
P(f"| model | τ | % FP |\n|---|---|---|\n| FP16 drafter | {main['tau_fp']['tau']:.3f} ± {main['tau_fp']['tau_sem']:.3f} | 100% |\n| ADMM 1-bit, W_final written back | {main['tau_q']['tau']:.3f} ± {main['tau_q']['tau_sem']:.3f} | {100*main['tau_q']['tau']/7.101:.1f}% |\n")
P("Per-projection ADMM relative reconstruction error (mean of 10 layers): " + ", ".join(f"{p}={np.mean([r['rel_err'] for r in main['admm'] if r['proj']==p]):.3f}" for p in ['q_proj','k_proj','v_proj','o_proj']) + ". Deployed binarized weight matches W_final to <3e-4 relative error (`gap_probe.py`).\n")
P("## 2. Fused path: which projection carries the loss (`ablate_proj.py`, `ablate.json`)\n\nOnly one projection type binarized (all 10 layers), FP elsewhere.\n\n| binarized | τ | % FP | rel. err |\n|---|---|---|---|")
for p in ["k_proj", "q_proj", "v_proj", "o_proj"]:
    r = ab[p]; P(f"| {p} only | {r['tau']:.3f} ± {r['tau_sem']:.3f} | {100*r['tau']/7.101:.1f}% | {r['mean_rel_err']:.3f} |")
P(f"| all four | {main['tau_q']['tau']:.3f} | {100*main['tau_q']['tau']/7.101:.1f}% | |\n")
P("## 3. Fused path: budget-neutral rank reallocation, ADMM-only (`alloc.py`, `alloc.json`)\n\n| ranks q/k/v/o | bits vs baseline | τ | % FP | Δ vs baseline | t | better |\n|---|---|---|---|---|---|---|")
base = al["baseline"]
for k, r in al.items():
    rk = r["ranks"]; m, s, t_, b = paired(r["per_sample_tau"], base["per_sample_tau"]) if k != "baseline" else (0, 0, 0, 0)
    P(f"| {rk['q_proj']}/{rk['k_proj']}/{rk['v_proj']}/{rk['o_proj']} ({k}) | {r['bits_vs_baseline']:.4f}× | {r['tau']:.3f} ± {r['tau_sem']:.3f} | {100*r['tau']/7.101:.1f}% | {m:+.3f} | {t_:+.1f} | {b}/40 |")
P("\n## 4. Unfused path: NanoQuant `tune_fact` vs target-as-teacher (`run_tune_fact.py`, `run_target_teacher.py`; `tf.json`, `tt.json`)\n\nADMM with **uniform** `i_norm`, 128 windows of chat-formatted GSM8K-train *reference answers* (off-policy).\n\n| tuning | bpw | τ | % FP | Δ vs ADMM-only | t | better |\n|---|---|---|---|---|---|---|")
P(row("FP16 drafter", fpr, bpw="16")); P(row("ADMM only, uniform i_norm", tf["tau_admm_only"], bpw="0.993"))
P(row("+ NanoQuant tune_fact (MSE vs FP drafter, 8 ep)", tf["tau_tuned"], tf["tau_admm_only"], "0.993"))
for k in sorted(tt["runs"]): P(row(f"+ {k.split('@')[0]} vs target, tied lr {k.split('@')[1]}", tt["runs"][k]["gen"], tf["tau_admm_only"], "0.993"))
P("\n`acc` = −Σ_k Π_{j≤k} p_j (surrogate of expected accepted length); `ce` = cross-entropy vs the target's greedy token. At lr 1e-4 both memorise the 1,920 supervised positions and τ falls below ADMM-only.\n")
P("## 5. Unfused path: on-policy data, calibrated `i_norm`, decoupled LRs (`run_onpolicy.py`; `op.json`, `op2x2.json`, `op4k.json`)\n\n1,024 windows × 192 tokens whose context is the *target's own greedy continuation* of a GSM8K-train prompt; labels = next 16 target-greedy tokens. Scales lr 1e-5, binary latents lr swept, 6 epochs, batch 4. Dev set = test[40:48].\n")
adm = op["admm_gen"]
P("| config | bpw | τ | % FP | Δ vs ADMM-only | t | better | flips | dev τ |\n|---|---|---|---|---|---|---|---|---|")
P(row("ADMM only, calibrated i_norm", adm, bpw="0.993") + " | | |")
for k, r in op["runs"].items(): P(row(k, r["gen"], adm, "0.993", f" {100*r['flip_frac']:.3f}% | {r['dev']['tau']:.3f} |"))
for k, r in o2["runs"].items(): P(row(k + " (2×2)", r["gen"], adm, "0.993", f" {100*r['flip_frac']:.3f}% | {r['dev']['tau']:.3f} |"))
r = o4["runs"]["ce@lat0.0001_sc1e-05"]; P(row("ce@lat1e-4, **4096** windows", r["gen"], adm, "0.993", f" {100*r['flip_frac']:.3f}% | {r['dev']['tau']:.3f} |"))
def pooled(keys): return np.mean([op["runs"][k]["gen"]["per_sample_tau"] for k in keys], 0)
m, s, t_, _ = paired(pooled([f"acc@lat{l}_sc1e-05" for l in ["1e-05", "0.0001", "0.0003"]]), pooled([f"ce@lat{l}_sc1e-05" for l in ["1e-05", "0.0001", "0.0003"]]))
P(f"\nContrasts: acc vs ce (pooled, lat ≤ 3e-4) {m:+.3f} ± {s:.3f}, t = {t_:+.1f}. ", end="")
m, s, t_, _ = paired(pooled(["acc@lat0.0003_sc1e-05", "ce@lat0.0003_sc1e-05"]), pooled(["acc@lat1e-05_sc1e-05", "ce@lat1e-05_sc1e-05"])); P(f"latent lr 3e-4 vs 1e-5 {m:+.3f} ± {s:.3f}, t = {t_:+.1f}. ", end="")
m, s, t_, _ = paired(op["runs"]["ce@lat0.0003_sc1e-05"]["gen"]["per_sample_tau"], o2["runs"]["ce@lat0_sc1e-05"]["gen"]["per_sample_tau"]); P(f"both vs scales-only {m:+.3f} ± {s:.3f}, t = {t_:+.1f}. ", end="")
m, s, t_, _ = paired(o4["runs"]["ce@lat0.0001_sc1e-05"]["gen"]["per_sample_tau"], op["runs"]["ce@lat0.0001_sc1e-05"]["gen"]["per_sample_tau"]); P(f"4096 vs 1024 windows {m:+.3f} ± {s:.3f}, t = {t_:+.1f}.\n")
P("## 6. Unfused path: residual binary base, and what did not lift its ceiling (`run_residual.py`, `run_final.py`; `res.json`, `calib.json`, `hadamard.json`, `kd.json`)\n\nW ≈ W₁ + W₂: second NanoQuant ADMM on the residual, r₂ = 256/96 → 1.127 bpw. Tuning = ce, latents 1e-4, scales 1e-5, 1,024 on-policy windows.\n")
rt = res["tuned"]["gen"]
P("| variant | bpw | E_ADMM (base→+res) | ADMM-only τ | tuned τ | % FP | Δ vs residual-tuned | t | better |\n|---|---|---|---|---|---|---|---|---|")
P(f"| per-channel scales (§5, ce@lat1e-4) | 0.993 | 0.2518 → – | {adm['tau']:.3f} | {op['runs']['ce@lat0.0001_sc1e-05']['gen']['tau']:.3f} | {100*op['runs']['ce@lat0.0001_sc1e-05']['gen']['tau']/FP:.1f}% | | | |")
P(f"| **+ residual base** | 1.127 | {res['rel_err_base']:.4f} → {res['rel_err_residual']:.4f} | {res['admm_gen']['tau']:.3f} | **{rt['tau']:.3f} ± {rt['tau_sem']:.3f}** | **{100*rt['tau']/FP:.1f}%** | — | — | — |")
for name, d in [("+ post-calibrator fine-tuned in FP (lr 1e-5)", ca), ("+ randomized-Hadamard input rotation, i_norm=diag(RᵀΣR)", ha), ("+ top-256 soft-label KD (last epoch)", kd)]:
    g = d["tuned"]["gen"]; m, s, t_, b = paired(g["per_sample_tau"], rt["per_sample_tau"]); ao = f"{d['admm_gen']['tau']:.3f}" if "admm_gen" in d else "(same)"
    P(f"| {name} | 1.127 | {d['rel_err_base']:.4f} → {d['rel_err_residual']:.4f} | {ao} | {g['tau']:.3f} ± {g['tau_sem']:.3f} | {100*g['tau']/FP:.1f}% | {m:+.3f} ± {s:.3f} | {t_:+.1f} | {b}/40 |")
g = kd["tuned_best_epoch"]["gen"]; m, s, t_, b = paired(g["per_sample_tau"], rt["per_sample_tau"])
P(f"| + KD, early-stopped on held-out proxy (epoch {kd['best_epoch']}) | 1.127 | | | {g['tau']:.3f} ± {g['tau_sem']:.3f} | {100*g['tau']/FP:.1f}% | {m:+.3f} ± {s:.3f} | {t_:+.1f} | {b}/40 |")
P("\n### Prefill-layout proxy τ (exact leading-match count on 15 mask positions, no generation)\n\n| | train windows (first 128) | held-out windows (128) |\n|---|---|---|")
P(f"| FP drafter | – | {ca['proxy_fp']['prefill_tau']:.2f} |\n| ADMM+residual only (diag) | {ca['proxy_admm_train']['prefill_tau']:.2f} | {ca['proxy_admm']['prefill_tau']:.2f} |")
P(f"| tuned, ce (diag) | {ca['tuned']['proxy_train']['prefill_tau']:.2f} | {ca['tuned']['proxy']['prefill_tau']:.2f} |")
P(f"| tuned, KD by epoch (held-out) | | {', '.join(f'{c[chr(104)+chr(111)]:.2f}' for c in kd['epoch_curve'])} |")
P(f"| tuned, ce (covariance pipeline, §8) | {cf['tuned']['proxy_train']['prefill_tau']:.2f} | {cf['tuned']['proxy']['prefill_tau']:.2f} |\n")
P("## 7. Not run\n\n* Group-128 scales: at init they reduce weight error by ~1% in this factored format (dry run, `run_g128.py`), so the full run was skipped.\n* bpw frontier (0.75 / 1.25 / 1.5): out of scope (≤ ~1.1 bpw). k/v are capped at r ≤ 1024 → 1.27 bpw in this factorisation.\n* Learned (Cayley) rotation, random-orthogonal control, EAGLE-3 drafter.\n")
P("## 8. The ADMM objective (`run_onorm.py`, `admm_cov.py`, `run_covgate.py`, `run_covfull.py`; `onorm.json`, `covgate*.json`, `covfull.json`)\n")
P("NanoQuant minimises ‖diag(√o_norm)·(W−AB)·diag(√i_norm)‖_F. Two changes were tested. **(A)** `o_norm` from the target-teacher loss gradients instead of 1 (NanoQuant's own `collect_stats` recipe, driven by our loss). **(B)** replace the diagonal input weighting by the full input covariance, ‖(W−AB)·L‖_F with LLᵀ = Σ: `admm_cov.py` keeps NanoQuant's alternating structure and Z/U/export steps, whitens the A-step, and solves the B-step exactly as a Sylvester equation via eigendecompositions (Σ = D·C·D with C the input correlation; with C = I the code path is bit-identical to NanoQuant — `test_admm_cov.py`). A naive 'whiten the proximal term too' variant is algebraically NanoQuant's original step and was discarded; the transposed variant is numerically unreliable on the real correlation (cond ≈ 4·10³ after 0.4 shrinkage) and is not used. Real drafter inputs are strongly correlated: effective rank ≈ 700 / 4096.\n")
P("### Held-out output error ‖(W−Ŵ)x‖²/‖Wx‖² on real mask inputs, base-only ADMM (same seeds, same `i_norm`)\n\n| | q_proj | k_proj | v_proj | o_proj | all | weight err (all) |\n|---|---|---|---|---|---|---|")
e = g2["errors"]; ed = on["diag"]
P("| diag i_norm (NanoQuant) | " + " | ".join(f"{e['diag'][p]['output_err']:.4f}" for p in ["q_proj","k_proj","v_proj","o_proj","all"]) + f" | {e['diag']['all']['weight_err']:.4f} |")
P("| (A) o_norm from target loss | " + " | ".join(f"{ed['learned'][p]['output_err']:.4f}" for p in ["q_proj","k_proj","v_proj","o_proj","all"]) + f" | {ed['learned']['all']['weight_err']:.4f} |")
P("| (B) full covariance | " + " | ".join(f"**{e['cov'][p]['output_err']:.4f}**" for p in ["q_proj","k_proj","v_proj","o_proj","all"]) + f" | {e['cov']['all']['weight_err']:.4f} |\n")
P("(A) lowers the *loss-weighted* output error (0.0851 → 0.0766) by design while raising the plain one; (B) halves the plain output error. Weight error goes *up* in both — weight error is not the quantity to minimise.\n")
P("### τ\n\n| pipeline | bpw | ADMM-only τ | % FP | tuned τ | % FP | tuned Δ vs diag pipeline | t | better |\n|---|---|---|---|---|---|---|---|---|")
P(f"| diag i_norm + residual + ce tuning (§6 reference) | 1.127 | {res['admm_gen']['tau']:.3f} | {100*res['admm_gen']['tau']/FP:.1f}% | {rt['tau']:.3f} ± {rt['tau_sem']:.3f} | {100*rt['tau']/FP:.1f}% | — | — | — |")
g = on["tuned"]["gen"]; m, s, t_, b = paired(g["per_sample_tau"], rt["per_sample_tau"])
P(f"| (A) o_norm from target loss + residual + tuning | 1.127 | {on['admm_gen']['tau']:.3f} | {100*on['admm_gen']['tau']/FP:.1f}% | {g['tau']:.3f} ± {g['tau_sem']:.3f} | {100*g['tau']/FP:.1f}% | {m:+.3f} ± {s:.3f} | {t_:+.1f} | {b}/40 |")
P(f"| (B) covariance ADMM, base only, no tuning (gate v1, k/v transposed) | 0.993 | {g1['admm_gen']['tau']:.3f} | {100*g1['admm_gen']['tau']/FP:.1f}% | | | | | |")
P(f"| (B) covariance ADMM, base only, no tuning (gate v2) | 0.993 | {g2['admm_gen']['tau']:.3f} | {100*g2['admm_gen']['tau']/FP:.1f}% | | | | | |")
g = cf["tuned"]["gen"]; m, s, t_, b = paired(g["per_sample_tau"], rt["per_sample_tau"])
P(f"| **(B) covariance ADMM + covariance residual + ce tuning** | 1.127 | {cf['admm_gen']['tau']:.3f} | {100*cf['admm_gen']['tau']/FP:.1f}% | **{g['tau']:.3f} ± {g['tau_sem']:.3f}** | **{100*g['tau']/FP:.1f}%** | {m:+.3f} ± {s:.3f} | {t_:+.1f} | {b}/40 |")
m, s, t_, b = paired(g2["admm_gen"]["per_sample_tau"], op["admm_gen"]["per_sample_tau"])
P(f"\nCovariance ADMM base-only vs diagonal base-only (both 0.993 bpw, no tuning): {m:+.3f} ± {s:.3f}, t = {t_:+.1f}, {b}/40. ", end="")
m, s, t_, b = paired(cf["tuned"]["gen"]["per_sample_tau"], cf["admm_gen"]["per_sample_tau"]); P(f"Tuning on top of the covariance init: {m:+.3f} ± {s:.3f}, t = {t_:+.1f}. ", end="")
m, s, t_, b = paired(cf["tuned"]["gen"]["per_sample_tau"], fpr["per_sample_tau"]); P(f"Best result vs FP: {m:+.3f} ± {s:.3f}, t = {t_:+.1f} (worse on {40-b}/40 prompts). Cost: cov-ADMM {g2['admm_sec']['cov']:.0f}s vs {g2['admm_sec']['diag']:.0f}s for 40 matrices (eigendecomposition per iteration).")
