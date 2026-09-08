# Results

All numbers: FlexDraft-Qwen3-8B drafter, Qwen3-8B target, 40 GSM8K test prompts (test[0:40]), block_size 16, temperature 0, draft-confidence-threshold 0.01, `cumulative_product`, max_new_tokens 256. **τ** = mean accepted length per drafting step (`Avg Acceptance Length` in FlexDraft's `inference.py`), micro-averaged over all steps; ± is SEM over steps. Paired statistics use per-prompt τ (n = 40). Ranks: q/o 2016, k/v 800 (NanoQuant `calculate_ranks`, bits=1.0) → 0.993 bpw incl. 16-bit scales; residual base r₂ = 256/96 → 1.127 bpw.

Two numerically-different-but-equivalent code paths were used. **Fused** = FlexDraft's `_build_fused_qkv` (FP baseline τ = 7.101). **Unfused** = per-module path required once weights are `NanoQuantLinear` (FP baseline τ = 7.312). Never compare across paths.

## Headline ladder (unfused path, % of FP τ = 7.312)

| stage | ADMM-only τ | tuned τ | % FP (best) |
|---|---|---|---|
| NanoQuant ADMM, uniform i_norm | 5.057 | | 69.1% |
| + activation-calibrated i_norm (diagonal) | 5.530 | 6.269 | 85.7% |
| + residual binary base (1.13 bpw) | 5.950 | 6.360 | 87.0% |
| **full-covariance ADMM** (base only) | 6.328 | | 86.5% |
| **full-covariance ADMM + residual + target-teacher tuning** | 6.457 | **6.675** | **91.3%** |

## 1. Fused path: ADMM-only with activation-calibrated `i_norm` (`run_all.py`, `main.json`)

| model | τ | % FP |
|---|---|---|
| FP16 drafter | 7.101 ± 0.147 | 100% |
| ADMM 1-bit, W_final written back | 5.612 ± 0.107 | 79.0% |

Per-projection ADMM relative reconstruction error (mean of 10 layers): q_proj=0.212, k_proj=0.211, v_proj=0.275, o_proj=0.304. Deployed binarized weight matches W_final to <3e-4 relative error (`gap_probe.py`).

## 2. Fused path: which projection carries the loss (`ablate_proj.py`, `ablate.json`)

Only one projection type binarized (all 10 layers), FP elsewhere.

| binarized | τ | % FP | rel. err |
|---|---|---|---|
| k_proj only | 7.066 ± 0.149 | 99.5% | 0.211 |
| q_proj only | 6.976 ± 0.143 | 98.2% | 0.212 |
| v_proj only | 6.817 ± 0.140 | 96.0% | 0.275 |
| o_proj only | 6.450 ± 0.129 | 90.8% | 0.304 |
| all four | 5.612 | 79.0% | |

## 3. Fused path: budget-neutral rank reallocation, ADMM-only (`alloc.py`, `alloc.json`)

| ranks q/k/v/o | bits vs baseline | τ | % FP | Δ vs baseline | t | better |
|---|---|---|---|---|---|---|
| 2016/800/800/2016 (baseline) | 1.0000× | 5.612 ± 0.107 | 79.0% | +0.000 | +0.0 | 0/40 |
| 1760/800/800/2272 (q->o_small) | 1.0000× | 5.592 ± 0.106 | 78.8% | -0.089 | -0.7 | 16/40 |
| 1536/800/800/2496 (q->o_large) | 1.0000× | 5.644 ± 0.108 | 79.5% | -0.026 | -0.2 | 19/40 |
| 2016/416/800/2240 (k->o (k<512)) | 0.9969× | 5.677 ± 0.109 | 79.9% | +0.037 | +0.3 | 19/40 |

## 4. Unfused path: NanoQuant `tune_fact` vs target-as-teacher (`run_tune_fact.py`, `run_target_teacher.py`; `tf.json`, `tt.json`)

ADMM with **uniform** `i_norm`, 128 windows of chat-formatted GSM8K-train *reference answers* (off-policy).

| tuning | bpw | τ | % FP | Δ vs ADMM-only | t | better |
|---|---|---|---|---|---|---|
| FP16 drafter | 16 | 7.312 ± 0.151 | 100.0% |
| ADMM only, uniform i_norm | 0.993 | 5.057 ± 0.088 | 69.1% |
| + NanoQuant tune_fact (MSE vs FP drafter, 8 ep) | 0.993 | 5.156 ± 0.094 | 70.5% | +0.082 ± 0.074 | +1.1 | 19/40 |
| + acc vs target, tied lr 0.0001 | 0.993 | 4.638 ± 0.082 | 63.4% | -0.476 ± 0.102 | -4.7 | 9/40 |
| + acc vs target, tied lr 1e-05 | 0.993 | 5.468 ± 0.105 | 74.8% | +0.365 ± 0.097 | +3.8 | 27/40 |
| + ce vs target, tied lr 0.0001 | 0.993 | 4.914 ± 0.089 | 67.2% | -0.157 ± 0.112 | -1.4 | 15/40 |
| + ce vs target, tied lr 1e-05 | 0.993 | 5.394 ± 0.102 | 73.8% | +0.316 ± 0.067 | +4.7 | 31/40 |

`acc` = −Σ_k Π_{j≤k} p_j (surrogate of expected accepted length); `ce` = cross-entropy vs the target's greedy token. At lr 1e-4 both memorise the 1,920 supervised positions and τ falls below ADMM-only.

## 5. Unfused path: on-policy data, calibrated `i_norm`, decoupled LRs (`run_onpolicy.py`; `op.json`, `op2x2.json`, `op4k.json`)

1,024 windows × 192 tokens whose context is the *target's own greedy continuation* of a GSM8K-train prompt; labels = next 16 target-greedy tokens. Scales lr 1e-5, binary latents lr swept, 6 epochs, batch 4. Dev set = test[40:48].

| config | bpw | τ | % FP | Δ vs ADMM-only | t | better | flips | dev τ |
|---|---|---|---|---|---|---|---|---|
| ADMM only, calibrated i_norm | 0.993 | 5.530 ± 0.107 | 75.6% | | | |
| acc@lat1e-05_sc1e-05 | 0.993 | 6.183 ± 0.130 | 84.6% | +0.683 ± 0.115 | +5.9 | 33/40 | 0.011% | 6.568 |
| acc@lat0.0001_sc1e-05 | 0.993 | 6.173 ± 0.129 | 84.4% | +0.704 ± 0.123 | +5.7 | 32/40 | 0.032% | 6.540 |
| acc@lat0.0003_sc1e-05 | 0.993 | 6.268 ± 0.131 | 85.7% | +0.799 ± 0.110 | +7.3 | 36/40 | 0.084% | 6.623 |
| acc@lat0.001_sc1e-05 | 0.993 | 5.268 ± 0.104 | 72.0% | -0.362 ± 0.125 | -2.9 | 11/40 | 0.454% | 5.418 |
| ce@lat1e-05_sc1e-05 | 0.993 | 6.156 ± 0.126 | 84.2% | +0.626 ± 0.116 | +5.4 | 32/40 | 0.012% | 6.765 |
| ce@lat0.0001_sc1e-05 | 0.993 | 6.269 ± 0.131 | 85.7% | +0.784 ± 0.116 | +6.8 | 36/40 | 0.039% | 6.739 |
| ce@lat0.0003_sc1e-05 | 0.993 | 6.252 ± 0.131 | 85.5% | +0.741 ± 0.101 | +7.4 | 37/40 | 0.091% | 6.532 |
| ce@lat0.001_sc1e-05 | 0.993 | 5.675 ± 0.113 | 77.6% | +0.129 ± 0.108 | +1.2 | 21/40 | 0.387% | 5.859 |
| ce@lat0_sc1e-05 (2×2) | 0.993 | 6.173 ± 0.127 | 84.4% | +0.688 ± 0.107 | +6.4 | 33/40 | 0.000% | 6.543 |
| ce@lat0.0003_sc0 (2×2) | 0.993 | 6.208 ± 0.128 | 84.9% | +0.713 ± 0.114 | +6.2 | 34/40 | 0.098% | 6.719 |
| ce@lat1e-4, **4096** windows | 0.993 | 6.335 ± 0.131 | 86.6% | +0.823 ± 0.108 | +7.6 | 36/40 | 0.075% | 6.680 |

Contrasts: acc vs ce (pooled, lat ≤ 3e-4) +0.011 ± 0.070, t = +0.2. latent lr 3e-4 vs 1e-5 +0.115 ± 0.078, t = +1.5. both vs scales-only +0.053 ± 0.078, t = +0.7. 4096 vs 1024 windows +0.039 ± 0.080, t = +0.5.

## 6. Unfused path: residual binary base, and what did not lift its ceiling (`run_residual.py`, `run_final.py`; `res.json`, `calib.json`, `hadamard.json`, `kd.json`)

W ≈ W₁ + W₂: second NanoQuant ADMM on the residual, r₂ = 256/96 → 1.127 bpw. Tuning = ce, latents 1e-4, scales 1e-5, 1,024 on-policy windows.

| variant | bpw | E_ADMM (base→+res) | ADMM-only τ | tuned τ | % FP | Δ vs residual-tuned | t | better |
|---|---|---|---|---|---|---|---|---|
| per-channel scales (§5, ce@lat1e-4) | 0.993 | 0.2518 → – | 5.530 | 6.269 | 85.7% | | | |
| **+ residual base** | 1.127 | 0.2518 → 0.2202 | 5.950 | **6.360 ± 0.131** | **87.0%** | — | — | — |
| + post-calibrator fine-tuned in FP (lr 1e-5) | 1.127 | 0.2518 → 0.2202 | (same) | 6.336 ± 0.131 | 86.6% | -0.038 ± 0.051 | -0.7 | 11/40 |
| + randomized-Hadamard input rotation, i_norm=diag(RᵀΣR) | 1.127 | 0.2459 → 0.2155 | 5.598 | 6.076 ± 0.125 | 83.1% | -0.338 ± 0.096 | -3.5 | 9/40 |
| + top-256 soft-label KD (last epoch) | 1.127 | 0.2518 → 0.2202 | (same) | 6.349 ± 0.130 | 86.8% | -0.006 ± 0.073 | -0.1 | 12/40 |
| + KD, early-stopped on held-out proxy (epoch 2) | 1.127 | | | 6.306 ± 0.128 | 86.2% | -0.073 ± 0.083 | -0.9 | 17/40 |

### Prefill-layout proxy τ (exact leading-match count on 15 mask positions, no generation)

| | train windows (first 128) | held-out windows (128) |
|---|---|---|
| FP drafter | – | 10.83 |
| ADMM+residual only (diag) | 8.38 | 8.27 |
| tuned, ce (diag) | 12.52 | 8.97 |
| tuned, KD by epoch (held-out) | | 9.08, 9.12, 9.03, 8.97, 9.01, 9.02 |
| tuned, ce (covariance pipeline, §8) | 12.20 | 9.44 |

## 7. Not run

* Group-128 scales: at init they reduce weight error by ~1% in this factored format (dry run, `run_g128.py`), so the full run was skipped.
* bpw frontier (0.75 / 1.25 / 1.5): out of scope (≤ ~1.1 bpw). k/v are capped at r ≤ 1024 → 1.27 bpw in this factorisation.
* Learned (Cayley) rotation, random-orthogonal control, EAGLE-3 drafter.

## 8. The ADMM objective (`run_onorm.py`, `admm_cov.py`, `run_covgate.py`, `run_covfull.py`; `onorm.json`, `covgate*.json`, `covfull.json`)

NanoQuant minimises ‖diag(√o_norm)·(W−AB)·diag(√i_norm)‖_F. Two changes were tested. **(A)** `o_norm` from the target-teacher loss gradients instead of 1 (NanoQuant's own `collect_stats` recipe, driven by our loss). **(B)** replace the diagonal input weighting by the full input covariance, ‖(W−AB)·L‖_F with LLᵀ = Σ: `admm_cov.py` keeps NanoQuant's alternating structure and Z/U/export steps, whitens the A-step, and solves the B-step exactly as a Sylvester equation via eigendecompositions (Σ = D·C·D with C the input correlation; with C = I the code path is bit-identical to NanoQuant — `test_admm_cov.py`). A naive 'whiten the proximal term too' variant is algebraically NanoQuant's original step and was discarded; the transposed variant is numerically unreliable on the real correlation (cond ≈ 4·10³ after 0.4 shrinkage) and is not used. Real drafter inputs are strongly correlated: effective rank ≈ 700 / 4096.

### Held-out output error ‖(W−Ŵ)x‖²/‖Wx‖² on real mask inputs, base-only ADMM (same seeds, same `i_norm`)

| | q_proj | k_proj | v_proj | o_proj | all | weight err (all) |
|---|---|---|---|---|---|---|
| diag i_norm (NanoQuant) | 0.0129 | 0.0287 | 0.1193 | 0.2018 | 0.0907 | 0.2475 |
| (A) o_norm from target loss | 0.0138 | 0.0328 | 0.1313 | 0.2025 | 0.0951 | 0.2686 |
| (B) full covariance | **0.0069** | **0.0138** | **0.0566** | **0.1107** | **0.0470** | 0.2827 |

(A) lowers the *loss-weighted* output error (0.0851 → 0.0766) by design while raising the plain one; (B) halves the plain output error. Weight error goes *up* in both — weight error is not the quantity to minimise.

### τ

| pipeline | bpw | ADMM-only τ | % FP | tuned τ | % FP | tuned Δ vs diag pipeline | t | better |
|---|---|---|---|---|---|---|---|---|
| diag i_norm + residual + ce tuning (§6 reference) | 1.127 | 5.950 | 81.4% | 6.360 ± 0.131 | 87.0% | — | — | — |
| (A) o_norm from target loss + residual + tuning | 1.127 | 5.875 | 80.3% | 6.268 ± 0.130 | 85.7% | -0.170 ± 0.140 | -1.2 | 15/40 |
| (B) covariance ADMM, base only, no tuning (gate v1, k/v transposed) | 0.993 | 6.414 | 87.7% | | | | | |
| (B) covariance ADMM, base only, no tuning (gate v2) | 0.993 | 6.328 | 86.5% | | | | | |
| **(B) covariance ADMM + covariance residual + ce tuning** | 1.127 | 6.457 | 88.3% | **6.675 ± 0.140** | **91.3%** | +0.337 ± 0.124 | +2.7 | 27/40 |

Covariance ADMM base-only vs diagonal base-only (both 0.993 bpw, no tuning): +0.873 ± 0.135, t = +6.4, 35/40. Tuning on top of the covariance init: +0.218 ± 0.090, t = +2.4. Best result vs FP: -0.631 ± 0.124, t = -5.1 (worse on 36/40 prompts). Cost: cov-ADMM 501s vs 147s for 40 matrices (eigendecomposition per iteration).
