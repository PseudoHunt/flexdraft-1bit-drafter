# Next steps (written 8 Sep 2026, end of day 1)

State at close: **91.3 % of FP τ at 1.13 bpw** (`results/covfull.json`) with the four-step method in the README.
Best weights: GitHub release `v0.1-day1` → `params_covfull.pt` (trainable tensors of the tuned covariance pipeline;
load with `draft.named_parameters()` name matching after re-running `run_covfull.py`'s quantisation step, or just
re-run — 50 min). Covariance factors: `cov_sides.pt` in the same release (3 min to regenerate with `run_covgate.py`).
Everything else regenerates from `setup.sh` + the commands in the README.

## A. Drafter: ideas ranked by expected value / cost

| # | idea | why | cost | expected |
|---|---|---|---|---|
| 1 | **Sequential covariance for o_proj** (BoA/GPTQ "inter-layer dependency"): collect Σ for o_proj's input from the drafter with q/k/v *already quantised*, refit o_proj only | o_proj is 60 % of remaining output error (0.111 of 0.047·4); its input distribution shifts once v is binarised | covariance re-collection + 40 ADMM solves for o only, ~20 min GPU | +0.1–0.3 τ at init |
| 2 | **Covariance-aware residual & scales**: the residual base and `s_pre`/`s_post` extraction still follow NanoQuant's export; fit the residual and re-derive scales under ‖·L‖ | same objective end to end | small code, ~30 min GPU | +0.1 |
| 3 | **BoA-style two-sided weighting** (`docs/llm_extension.md` §3): per-head row weighting K_hᵀK_h (q), Q_hᵀQ_h (k), W_outᵀW_out with attention-weighted input cov (v); A-step becomes a Sylvester too | principled; but targets q/k/v whose output error is already 0.007/0.014/0.057 | ~1 h impl, ~30 min GPU | +0.1–0.2 |
| 4 | **o_proj rank / bits**: o_proj alone carried 9 of the 21 lost points; rank 2016→2528 for o only (+0.06 bpw avg) | the only capacity lever left; allocation was null *before* the covariance objective — retest | ~25 min | unknown, could be the last 3 points |
| 5 | Learned input rotation (Cayley) now that the weighting follows the basis — **online only** for this drafter (32 MiB/layer dense, or Hadamard-structured) | Hadamard hurt with diagonal weighting; may reverse with covariance | 1 day | speculative |
| 6 | Second drafter family (EAGLE-3) | generality for a paper; standalone head = easy case | 1 day | — |

Do not repeat (all null with paired stats, see RESULTS.md): acceptance-weighted loss, latent-vs-scale, latent LR,
4× data, budget-neutral rank reallocation on the *diagonal* objective, g128 scales, calibrator fine-tuning,
Hadamard with diagonal `i_norm`, soft-label KD, early stopping, `o_norm` from the target loss.

## B. Process notes that cost time today

* `NanoQuantLinear` has no `.weight` → FlexDraft's fused path must be bypassed (`force_unfused`); FP baseline is 7.312 on that path.
* Save tuned params (`--save-params`) — three runs had to be repeated for lack of them.
* Held-out prefill-τ proxy on same-distribution windows does **not** detect overfitting; select on generation-τ over disjoint prompts.
* ADMM with sign projections is chaotic: bit-identical reproduction needs identical ops *and* dtypes (bf16 norms).
* A "whitened proximal term" ADMM is vacuous (C cancels) — the B-step must be a real Sylvester solve.
* Keep NanoQuant's `reg` on B in the original metric (not multiplied by C) or ill-conditioned directions blow up.
* The transposed covariance path fails at real condition numbers (~4·10³); always use the non-transposed form.
