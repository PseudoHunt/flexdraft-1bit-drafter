# Extending NanoQuant with the covariance objective — plan for LLMs

Goal: test whether `admm_cov.py` (full-input-covariance ADMM, exact Sylvester B-step) improves NanoQuant on a plain
LLM, judged **after NanoQuant's own refinement stages**, on perplexity and zero-shot — the rotation-note protocol
(E_ADMM, E after Step 3, task metric).

## 1. Why it should transfer

The covariance we exploited *is* an LLM's: the drafter's q/k/v read Qwen3-8B's layernorm outputs, o_proj reads its
attention outputs. Effective rank ≈ 700/4096, raw condition ≈ 3·10⁵. Massive-activation channels + low-rank activation
structure are universal in LLMs; MLP `down_proj` inputs are usually worse. Layer-wise output error halved on the drafter
(0.091 → 0.047) with weight error going *up* — weight error is the wrong target and NanoQuant's diagonal `i_norm` only
approximates output error.

## 2. What NOT to expect

* The **target-as-teacher** lever (+10 points on the drafter) does not exist for an LLM: the FP model *is* the target,
  and KD from it is already NanoQuant's `compress_model_recon`. Only the *on-policy data* sub-idea carries over
  (calibrate on the model's own generations, LLM-QAT style) — probably minor.
* NanoQuant's Step 3 (`tune_nonfact` → `tune_fact` → KD) is far stronger than our tuning (all positions supervised,
  128 × 2048 tokens, FP weights of the block adjustable). It may recover part of the diagonal's deficit by itself, so the
  post-Step-3 gain will be smaller than the init gain. **Report both.**
* The paper says "Hessian-aware"; the release implements the diagonal. Re-run the released code first so the
  comparison baseline is the code, not the paper table.

## 3. Wiring it into NanoQuant's pipeline (Qwen3-0.6B first, then 1.7B)

1. **Covariance collection** — extend `nanoquant/core/importance.py` `collect_stats`: in the forward hook also
   accumulate `x.T @ x / tokens` per linear (fp32, on `stats_device`). Memory: hidden 1024 (0.6B) → 4 MB per matrix,
   trivial; 2048 (1.7B) → 16 MB; MLP `down_proj` input = intermediate size (3072 / 6144) → 36 / 150 MB per layer.
   Shrink with NanoQuant's `calib_shrinkage` as `(1−s)Σ + s·mean(diag)·I` (its diagonal equals `get_shrunk_stats`).
   Store per (layer, linear); note q/k/v share one Σ, gate/up share one.
2. **Factorisation** — in `compress_block.factorize_and_replace`, add `admm_type == "cov"` calling
   `factorize_admm_cov(W, CovSide(Σ_shrunk), o_norm, mid_rank=rank, ...)`. Pass `o_norm` as NanoQuant computes it
   (its gradient statistic; on the drafter it was null but on an LLM it is their default — keep it). **Never
   transpose** (`is_transpose` is ignored in `admm_cov.py` on purpose): for `down_proj`/`fc2` the covariance lives on
   the (wide) input side and the non-transposed Sylvester handles it. `_export` is NanoQuant's own, so
   `NanoQuantLinear.__quant_convert__` works unchanged.
3. **Everything downstream unchanged** — `tune_nonfact`, `tune_fact`, `compress_model_recon`, kernels.
4. **Cost** — eigh on the mid dimension per iteration: r ≈ 500 (0.6B q_proj) is ~5 ms, r ≈ 1500 (1.7B MLP) ~25 ms;
   400 iterations × 7 linears × 28 layers ≈ +30–60 min over NanoQuant's ADMM on one A100. Eigendecompose each Σ once.
5. **Ablation arms** — diagonal (released), covariance, and covariance + on-policy calibration text. Report
   E_ADMM (weight and *output* error on held-out tokens), E after Step 3, wikitext2 PPL, the zero-shot suite in
   `eval_utils.py`. Match seeds; NanoQuant's `set_seed` is called inside `factorize_and_replace`.
6. **Optional BoA-style step** (Samsung's own follow-up to GPTQ, ICML 2025, arXiv 2406.13474): per-head row weighting
   `H_row` (K_hᵀK_h for q, Q_hᵀQ_h for k, W_outᵀW_out for v with attention-weighted input covariance). In our ADMM:
   B-step uses M := X_Bᵀ H_row X_B; A-step becomes a Sylvester with H_row on the left (eigendecompose the 128×128
   head blocks once). On the drafter the expected gain is small because o_proj (unchanged by BoA) dominates; on an
   LLM at INT2 BoA reports +3–6 points over GPTQ, so worth one arm.

## 4. Pitfalls already paid for (see NEXT_STEPS.md §B)

Circular import (`nanoquant.modules.linear` before `nanoquant.core.*`); extra deps `loguru cut_cross_entropy lm_eval`;
`reg` must stay on B in the original metric; the whitened-proximal shortcut is vacuous; transposed path fails at real
condition numbers; TF32 is on globally after importing `admm_nq.py`; latents are bf16 — compute norms in bf16 to match.

## 5. Minimal command sketch

```bash
# in NanoQuant (after adding admm_type=cov + covariance stats):
python -m nanoquant.main --model_id Qwen/Qwen3-0.6B --admm_type nanoquant --ppl_task wikitext2 --qmodel_path q06-diag.pt
python -m nanoquant.main --model_id Qwen/Qwen3-0.6B --admm_type cov       --ppl_task wikitext2 --qmodel_path q06-cov.pt
```
