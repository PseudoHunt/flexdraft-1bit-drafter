# Covariance ADMM inside NanoQuant — Qwen3-0.6B-Base results

All numbers from `llm_ext/results/*.json` (regenerate the tables with `python llm_ext/compare.py`).
§1–2 on one NVIDIA L4-24GB, §3–7 on one H200 (several runs concurrently under CUDA MPS). wikitext2, seqlen 2048, `bits=1.0` → **0.973 bpw** over the factorized matrices,
128 calibration samples, `calib_shrinkage=0.4`, 400 ADMM iterations, seed 0.

The two arms share one cached statistics file, so `i_norm`, `o_norm` and Σ are bit-identical between
them and **`admm_type` is the only difference**. `diag(shrink(Σ))` equals NanoQuant's shrunk `i_norm`
to 3e-6 relative, by construction.

FP reference: **wikitext2 PPL 12.669**.

## 1. E_ADMM — factorization only, no tuning (28 blocks)

| | mean weight err | mean Σ-weighted **output** err | wikitext2 PPL | compress time |
|---|---|---|---|---|
| NanoQuant diagonal | **0.3124** | 0.1571 | 7.7e8 | 9.8 min |
| covariance | 0.3253 | **0.0957** (−39%) | 9707 | 19.5 min |

Weight error goes *up* while output error goes *down* — the signature that weight error is the wrong
target and NanoQuant's diagonal `i_norm` only approximates the output error. Neither model is usable
without Step 3; PPL at this stage is a sanity signal, not a result.

### Per-projection mean over the 28 blocks (Σ-weighted output error)

| projection | diagonal | covariance | change |
|---|---|---|---|
| `self_attn.o_proj` | 0.2090 | **0.0970** | −54% |
| `mlp.down_proj` | 0.2318 | **0.1493** | −36% |
| `mlp.up_proj` | 0.2012 | **0.1273** | −37% |
| `self_attn.v_proj` | 0.2609 | **0.1682** | −36% |
| `self_attn.k_proj` | 0.0697 | **0.0455** | −35% |
| `self_attn.q_proj` | 0.0653 | **0.0426** | −35% |
| `mlp.gate_proj` | 0.0621 | **0.0401** | −35% |

Lower on every projection type. The largest gain is `o_proj`, whose input covariance is the most
structured (effective rank 54 of 2048; the q/k/v input is 3.9 of 1024).

## 2. After NanoQuant's Step 3 — one block quantized, the other 27 left in FP

Block 13 of 28, `tune_nonfact` 8 epochs + `tune_fact` 8 epochs (NanoQuant's defaults). Quantizing a
single block keeps full-model PPL meaningful.

| | ADMM out-err (init) | tuned block err | wikitext2 PPL | Δ vs FP |
|---|---|---|---|---|
| NanoQuant diagonal | 0.1567 | 0.00204 | 12.8693 | +0.2002 |
| covariance | **0.0904** | **0.00184** | **12.8411** | **+0.1720** (−14%) |

The init advantage **shrinks but survives** tuning, exactly as `docs/llm_extension.md` §2 predicted:
Step 3 recovers most of the diagonal's deficit by itself. A 0.028 PPL difference on one block is a
single sample without error bars — directionally consistent with the block error, not conclusive alone.

## 3. Full model + full Step 3 — two runs per arm

All 28 blocks, `tune_nonfact` 8 + `tune_fact` 8 + model KD 8 (NanoQuant's defaults), 128 calibration samples,
wikitext2 PPL evaluated after every block (`--ppl_after_block`). **Each arm was run twice** with identical seed,
statistics cache and code (`run_full_tuned_parallel.sh`, then again with `TAG=q06_tuned_rep`); both arms of a pair
ran concurrently on one H200.

| run | wikitext2 PPL | Δ vs FP | pre-KD PPL | KD gain | block 3 PPL | mean weight err | mean Σ-out err |
|---|---|---|---|---|---|---|---|
| diag #1 | **28.105** | +15.44 | 32.567 | 4.46 | 16.057 | 0.3158 | 0.1598 |
| cov #1 | **30.817** | +18.15 | 34.119 | 3.30 | 20.036 | 0.3287 | 0.0977 |
| diag #2 | **33.695** | +21.03 | 36.935 | 3.24 | 19.271 | 0.3158 | 0.1600 |
| cov #2 | **35.902** | +23.23 | 36.130 | 0.23 | 19.244 | 0.3287 | 0.0977 |

**Verdict: no detectable difference between the two objectives on full-model PPL at 0.973 bpw.** Covariance is
worse in both pairs (+2.71, +2.21), but the *within-arm* spread between two identical runs is
5.6 (diagonal) and 5.1 (covariance) — twice the between-arm gap. Means 30.9 vs 33.4, two
samples each: t ≈ 0.9. A consistent sign across two pairs is a coin flip. What *is* reproducible is the calibration
objective: the ADMM stage is deterministic to four decimals (weight err 0.3158 / 0.3287, Σ-weighted output err
0.160 / 0.0977 in both runs), so the covariance objective's **−39% Σ-weighted output error is real** — and it buys
nothing measurable on held-out text. Neither arm is usable at these settings (2.2–2.8× the FP reference).

Two secondary observations, n = 2 each, suggestive only:

* **KD recovers less on the covariance arm**: 4.5 / 3.2 PPL for the diagonal vs 3.3 / 0.2 for covariance. In run 2 the
  covariance chain finished the block loop 0.8 *ahead* and ended 2.2 *behind* after KD.
* **All of the run-to-run variance is Step 3.** The factorisation is bit-reproducible; the 5–6 PPL spread comes from the
  batch-1 tuning loop (`tune_nonfact` / `tune_fact`, non-deterministic CUDA kernels), and it is injected at one block:

<details><summary>Per-block PPL, all four runs (blocks 0..b quantised, rest FP)</summary>

| block | diag #1 | cov #1 | diag #2 | cov #2 | cov−diag #1 | cov−diag #2 |
|---|---|---|---|---|---|---|
| 0 | 18.095 | 16.592 | 18.070 | 16.906 | -1.503 | -1.164 |
| 1 | 14.266 | 14.193 | 14.295 | 14.394 | -0.073 | +0.099 |
| 2 | 18.548 | 16.029 | 18.229 | 16.174 | -2.519 | -2.056 |
| 3 | 16.057 | 20.036 | 19.271 | 19.244 | +3.979 | -0.027 |
| 4 | 16.160 | 20.004 | 18.964 | 18.877 | +3.844 | -0.087 |
| 5 | 16.548 | 20.330 | 19.000 | 19.289 | +3.782 | +0.289 |
| 6 | 16.767 | 19.817 | 19.006 | 19.339 | +3.050 | +0.332 |
| 7 | 16.890 | 19.730 | 19.184 | 19.478 | +2.840 | +0.294 |
| 8 | 17.003 | 19.570 | 19.178 | 19.589 | +2.567 | +0.411 |
| 9 | 17.239 | 19.434 | 19.742 | 19.838 | +2.195 | +0.095 |
| 10 | 17.638 | 19.491 | 19.927 | 20.211 | +1.852 | +0.284 |
| 11 | 17.914 | 19.904 | 20.265 | 20.542 | +1.989 | +0.277 |
| 12 | 18.175 | 20.021 | 20.635 | 20.790 | +1.846 | +0.155 |
| 13 | 18.463 | 20.494 | 21.002 | 21.016 | +2.031 | +0.013 |
| 14 | 18.624 | 20.783 | 21.165 | 21.267 | +2.159 | +0.102 |
| 15 | 18.846 | 20.743 | 21.338 | 21.567 | +1.897 | +0.230 |
| 16 | 19.409 | 21.255 | 21.876 | 22.079 | +1.846 | +0.202 |
| 17 | 19.927 | 21.911 | 22.390 | 22.571 | +1.984 | +0.180 |
| 18 | 20.371 | 22.228 | 23.007 | 23.126 | +1.857 | +0.118 |
| 19 | 21.558 | 23.530 | 24.246 | 24.739 | +1.972 | +0.493 |
| 20 | 22.273 | 24.088 | 24.967 | 25.331 | +1.815 | +0.363 |
| 21 | 23.020 | 24.901 | 25.887 | 26.261 | +1.881 | +0.374 |
| 22 | 23.669 | 25.609 | 26.558 | 26.884 | +1.940 | +0.326 |
| 23 | 24.371 | 26.326 | 27.398 | 27.583 | +1.955 | +0.185 |
| 24 | 25.217 | 27.057 | 28.374 | 28.286 | +1.840 | -0.088 |
| 25 | 26.258 | 28.240 | 29.735 | 29.309 | +1.982 | -0.427 |
| 26 | 27.934 | 30.036 | 31.592 | 30.976 | +2.102 | -0.616 |
| 27 | 32.567 | 34.119 | 36.935 | 36.130 | +1.552 | -0.805 |

</details>

Run 1's block-3 gap (+3.98) was the diagonal drawing 16.06 against covariance's 20.04; in run 2 both drew ≈19.25
(gap −0.03) and stayed within ±0.5 of each other for 22 blocks. The diagonal's run-1 block-3 value is the best of every
run that has passed through that block (§4.1); the offset a chain draws there is carried, and grows, for the remaining
24 blocks. §1's per-projection picture is unchanged (the ADMM stage is deterministic); the per-projection means over the
28 blocks are in `compare.py`'s output.

## 4. The block-3 step: what it is, and what did not fix it

Everything in this section is blocks 0–9 only (block *i* depends only on blocks < *i*, so rows 0–9 of the full runs are
exact baselines), no model KD, PPL after every block. Driver: `run_block3_screen.sh`; per-block β search:
`run_beta_search.sh`. All runs share `stats_q06.pt`.

### 4.1 It is a lottery both arms play

| block | diag #1 | cov #1 | diag rep | cov rep (β=0) | β=0.25 | β=0.5 | β=0.75 | attn only | o_proj only | MLP only | β search |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 18.10 | 16.59 | 16.60 | 16.89 | 16.78 | 16.20 | 15.80 | 17.72 | 16.93 | 16.77 | 16.04 |
| 1 | 14.27 | 14.19 | 14.40 | 14.21 | 14.31 | 14.18 | 14.31 | 14.31 | 14.19 | 14.43 | 14.13 |
| 2 | 18.55 | 16.03 | 18.88 | 15.93 | 16.02 | 16.63 | 17.06 | 18.58 | 18.78 | 15.84 | 15.89 |
| 3 | 16.06 | 20.04 | 17.87 | 19.08 | 24.28 | 17.90 | 23.44 | 18.46 | 16.99 | 22.81 | 19.40 |
| 4 | 16.16 | 20.00 | 18.28 | 18.17 | 23.10 | 17.78 | 21.82 | 18.51 | 17.16 | 22.15 | 17.65 |
| 5 | 16.55 | 20.33 | 18.52 | 18.22 | 22.55 | 17.90 | 21.65 | 18.86 | 17.63 | 21.82 | 17.72 |
| 6 | 16.77 | 19.82 | 18.48 | 18.10 | 22.69 | 17.84 | 21.56 | 18.61 | 17.59 | 21.70 | 17.51 |
| 7 | 16.89 | 19.73 | 18.62 | 18.22 | 21.93 | 17.87 | 21.48 | 18.63 | 17.68 | 21.44 | 17.40 |
| 8 | 17.00 | 19.57 | 18.62 | 18.53 | 21.99 | 17.94 | 21.61 | 18.81 | 17.76 | 21.29 | 17.45 |
| 9 | 17.24 | 19.43 | 18.90 | 18.83 | 22.22 | 18.13 | 21.91 | 19.17 | 18.05 | 21.74 | 17.69 |
| err@3 | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0015 | 0.0014 | 0.0014 | 0.0013 | 0.0013 |

`err@3` is the calibration block error at block 3: identical (0.0013–0.0015) for every column. Block-3 test PPL across
the runs that reached it — diagonal **16.06, 17.87, 19.27, ≈20.7** (four runs); plain covariance 19.08, 19.24, 20.04;
variants 17.90 … 24.29. Run 1's 16.06 is the single best draw of fourteen. Block 3 is where the calibration error
collapses **10×** (0.015 → 0.0014): Step 3 reconstructs the block almost perfectly on the 128 calibration sequences, the
objective has no information left, and the remaining degrees of freedom land in whichever basin the non-deterministic
tuner wanders into. Those basins differ by 3–4 PPL on held-out text. The covariance objective neither causes this nor can
help with it — its extra information is also calibration-set information.

### 4.2 Correlation shrinkage (`--cov_corr_shrink β`)

`C ← (1−β)·C + β·I` on the input *correlation* only; `i_norm` untouched, so β=0 is the covariance arm and β=1 is
exactly the diagonal. Block error is monotone in β everywhere (block 2: 0.0058 → 0.0063 → 0.0079 → 0.0095 → 0.0149),
so the knob interpolates as designed. Block-3 PPL is not monotone: 19.08 / 24.28 / 17.90 / 23.44 for β = 0 / .25 / .5 /
.75. It is not a conditioning problem — on the cached Σ, cond(C) at block 3 falls from 2.7·10⁴ (q_proj) / 8.4·10⁴
(o_proj) at β=0 to 134 / 690 at β=0.25 and 16 / 78 at β=0.75. Block 3's input spectrum is also unremarkable (effective
rank of C: q_proj 184, o_proj 60, gate 159, down 263 — same as blocks 2 and 4). The step lives in the *solution*, not
the statistics.

### 4.3 Per-projection localisation (`--cov_layers`)

Block 2's −61% block error / −2.5 PPL is **entirely MLP-borne**: `attn only` and `o_proj only` sit on the diagonal
there (18.58 / 18.78 vs 18.55), `MLP only` reproduces the full gain (15.84). But the block-3 step appears with *any*
subset — attention only 18.46, MLP only 22.81 — except o_proj alone (16.99, then a steady +0.8 over the diagonal to
block 9). At block 9 `MLP only` is the worst column (21.74). No per-projection hybrid avoids the step.

### 4.4 Per-block β search on a validation split (`--cov_beta_search`)

For each block, Step 3 is run once per β ∈ {0, 0.25, 0.5, 0.75, 1}; the candidate with the lowest wikitext2
**validation**-split PPL (blocks 0..b quantised, rest FP) is kept. The validation split is disjoint from both the
calibration set (train) and the reported metric (test). β=1 uses the diagonal path directly. Cost: 5× the block loop.

| block | β=0 | β=0.25 | β=0.5 | β=0.75 | β=1 (diag) | chosen | test PPL |
|---|---|---|---|---|---|---|---|
| 0 | 16.56 | 16.57 | **16.26** | 17.52 | 19.85 | 0.5 | 16.035 |
| 1 | 14.86 | **14.81** | 15.00 | 14.83 | 15.00 | 0.25 | 14.133 |
| 2 | 16.78 | **16.69** | 16.99 | 17.93 | 20.56 | 0.25 | 15.893 |
| 3 | 21.10 | **19.84** | 19.87 | 20.64 | 20.70 | 0.25 | 19.401 |
| 4 | 18.24 | **18.15** | 18.23 | 18.30 | 18.34 | 0.25 | 17.652 |
| 5 | 18.21 | **18.20** | 18.29 | 18.22 | 18.28 | 0.25 | 17.720 |
| 6 | 18.18 | 18.14 | **18.09** | 18.16 | 18.20 | 0.5 | 17.514 |
| 7 | 18.01 | 18.00 | **17.97** | 18.10 | 18.12 | 0.5 | 17.402 |
| 8 | 18.09 | 18.08 | **18.04** | 18.06 | 18.07 | 0.5 | 17.449 |
| 9 | 18.34 | **18.24** | 18.34 | 18.32 | 18.34 | 0.25 | 17.688 |

Bold = chosen (validation PPL); last column is the chosen candidate's **test** PPL. Chosen β: 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.25 —
never 0 (full covariance) and never 1 (diagonal). Validation ranks candidates the way test does (β=1 is last at blocks 0
and 2, where the diagonal runs are worst on test). At block 3 the five candidates spanned 1.3 validation PPL and the
search landed at 19.40 — inside the band every chain lands in; it cannot manufacture a 16 when none of five draws produces
one. From block 4 it recovers block by block (19.40 → 17.65 → … ) while every fixed-β chain drifts up, and finishes ten
blocks at **17.688** against the diagonal's 17.24 / 18.90 and plain covariance's 19.43 / 18.83.

### 4.5 Why block 3, and what would actually move the number

Block 3 is not architecturally special (Qwen3-0.6B-Base: 28 identical full-attention GQA blocks, no sliding window,
identical rank allocation). What changes there is the residual stream: a handful of massive-activation channels
(277, 16, 0, 23) come to dominate the block input — effective rank of Σ at the q_proj input 14.7 → 7.6 → 7.2 →
**5.5** → 4.2 → 3.8 over blocks 0–5, top-4 channel share 17% → 31% → 31% → **37%** → 41% → 45%. The relative block error
is normalised by that energy, so once the tuner matches those channels the error reads 0.001 whatever happens on the
other ~1020 channels that carry token-level information. (NanoQuant's `importance` weight, `o_norm` of `down_proj`, is
flat — top channel 0.7% — so it is not amplifying this; the plain MSE denominator is.) Hypothesis, not yet tested: a
per-channel-normalised block loss, or excluding the massive channels from it, would give the tuner an objective that
still discriminates at block 3.

The lever is not the ADMM objective. It is constraining Step 3 at the blocks where calibration error hits the floor:
more calibration data (256–512 samples), validation-based early stopping inside `tune_nonfact` / `tune_fact`, or
best-of-K restarts per block selected on validation (§4.4's machinery with fixed β and K ≈ 10–20). Any headline
comparison at this bit-width needs ≥3 runs per arm; a single pair is inside the noise.

## 5. 512 calibration samples (the paper's Table-7 regime)

`run_n512_screen.sh`: a fresh 512-sample statistics cache (Σ, `i_norm`, `o_norm` re-collected from 512 × 2048 tokens =
1.05M, the amount NanoQuant's paper uses for its QAT comparison; its default 128 = 0.26M), blocks 0–9, no KD, PPL after
every block, **two runs per arm**. `EPOCHS=2` so the optimizer-step budget equals the 128-sample / 8-epoch baseline
(1024 sample-passes per tuning phase) — this isolates *more data* from *more compute*.

| block | diag 128 #1 | diag 128 #2 | cov 128 #1 | cov 128 #2 | diag 512 #1 | diag 512 #2 | cov 512 #1 | cov 512 #2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 18.10 | 18.07 | 16.59 | 16.91 | 17.67 | 16.99 | 16.12 | 16.32 |
| 1 | 14.27 | 14.29 | 14.19 | 14.39 | 14.09 | 14.14 | 14.03 | 14.15 |
| 2 | 18.55 | 18.23 | 16.03 | 16.17 | 15.63 | 15.50 | 14.58 | 14.80 |
| 3 | 16.06 | 19.27 | 20.04 | 19.24 | 15.45 | 15.75 | 14.88 | 15.91 |
| 4 | 16.16 | 18.96 | 20.00 | 18.88 | 15.51 | 15.77 | 15.02 | 16.00 |
| 5 | 16.55 | 19.00 | 20.33 | 19.29 | 15.88 | 16.08 | 15.34 | 16.43 |
| 6 | 16.77 | 19.01 | 19.82 | 19.34 | 15.91 | 16.02 | 15.41 | 16.50 |
| 7 | 16.89 | 19.18 | 19.73 | 19.48 | 16.05 | 16.15 | 15.59 | 16.61 |
| 8 | 17.00 | 19.18 | 19.57 | 19.59 | 16.08 | 16.21 | 15.64 | 16.76 |
| 9 | 17.24 | 19.74 | 19.43 | 19.84 | 16.32 | 16.46 | 15.87 | 16.87 |
| err@3 | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0016 | 0.0016 | 0.0014 | 0.0015 |
| err@9 | 0.0041 | 0.0041 | 0.0039 | 0.0039 | 0.0052 | 0.0052 | 0.0048 | 0.0048 |

Block error is measured on each run's own calibration set (a 512-sequence set is a harder target, so compare it only
within the 512 group). Mean ± half-spread of the two replicates:

| | 128 samples, block 3 | 512 samples, block 3 | 128 samples, block 9 | 512 samples, block 9 |
|---|---|---|---|---|
| diagonal | 17.66 ± 1.61 | **15.60 ± 0.15** | 18.49 ± 1.25 | **16.39 ± 0.07** |
| covariance | 19.64 ± 0.40 | **15.39 ± 0.52** | 19.64 ± 0.20 | **16.37 ± 0.50** |

* **The block-3 lottery disappears.** Diagonal replicates 0.3 apart (were 3.2), and both below the best of the fourteen
  128-sample draws (16.06). Calibration error at block 3 no longer collapses as far (0.0016 vs 0.0014 on a 4× larger set)
  and the tuner fits its calibration set less everywhere (err@9 0.0052 vs 0.0041) while generalising better — the
  signature the §4.5 hypothesis predicted. Same optimizer budget; the extra information, not extra steps, did it.
* **Both objectives improve by 2–3 PPL at ten blocks and land on the same mean** (16.39 vs 16.37). Block 2's covariance
  advantage at 128 samples (−2.5 PPL) turns out to be a data effect too: at 512 the diagonal reaches 15.5–15.6 there on
  its own, and covariance goes further (14.6–14.8, block error 0.0020). The covariance arm's replicates are wider at 512
  (1.0 vs 0.1); `cov 512 #1` is the best chain in the table from block 2 on, `cov 512 #2` is not.
* This reproduces the direction of the paper's Table 4 (Llama-2-7B: 10.34 → 8.85 PPL from 0.26M → 2.10M tokens) at the
  block level on a 12× smaller model, and it is the first setting here where an arm comparison is not dominated by noise.

Not yet run: the full 28-block + KD model at 512 samples (two runs per arm; ~3 h on the H200 with the paper's 8 epochs),
which is what a headline number in the paper's regime would need.

## 6. Sensitivity-aware rank allocation at equal bpw (idea H) — the first win

NanoQuant gives every block the same rank. §3's per-block curve says the damage is not uniform: block error rises
10× over blocks 15–27 and the last block alone costs +4–5 PPL, while blocks 3–15 sit at 0.002. `block_bits.py`
turns a reference run's per-block error into per-block bit multipliers (`mult_i = exp(γ·z_i)`, z = standardised
log-error, 3-block smoothing), then scales them so the **realised bpw after rank rounding matches the uniform
allocation** (0.9722–0.9726 vs 0.9729 — never above). `--block_bits` applies them in `calculate_ranks`.
Runs: 512 calibration samples, block loop at matched steps (EPOCHS=2), model KD = NanoQuant's default budget (first
128 samples × 8 epochs), diagonal arm, two replicates per setting, all concurrent on the H200; plus two allocated runs at
NanoQuant's unmodified defaults (128 samples, 8/8/8 epochs) against the two uniform runs of §3.

| run | bpw | pre-KD PPL (block 27) | **post-KD PPL** | KD gain | block 3 | block 16 | block 23 |
|---|---|---|---|---|---|---|---|
| uniform #1 | 0.9729 | 30.126 | KD OOM-killed (300 GB cap), pre-KD only | - | 15.04 | 17.95 | 21.89 |
| uniform #2 | 0.9729 | 32.707 | KD OOM-killed (300 GB cap), pre-KD only | - | 16.48 | 19.42 | 23.86 |
| γ=0.15 #1 | 0.9722 | 28.386 | **24.246** | 4.14 | 15.05 | 18.70 | 21.95 |
| γ=0.15 #2 | 0.9722 | 28.307 | **23.972** | 4.33 | 15.01 | 18.71 | 21.89 |
| γ=0.3 #1 | 0.9726 | (16.95 @ block 8, stopped) | - | - | 15.11 | - | - |
| γ=0.3 #2 | 0.9726 | (16.94 @ block 8, stopped) | - | - | 15.07 | - | - |
| uniform@128 #1 | 0.9729 | 32.567 | **28.105** | 4.46 | 16.06 | 19.41 | 24.37 |
| uniform@128 #2 | 0.9729 | 36.935 | **33.695** | 3.24 | 19.27 | 21.88 | 27.40 |
| γ=0.15@128 #1 | 0.9722 | (21.19 @ block 8, stopped) | - | - | 19.23 | - | - |
| γ=0.15@128 #2 | 0.9722 | (18.16 @ block 8, stopped) | - | - | 16.96 | - | - |

| | 512 samples: uniform | γ=0.15 | γ=0.3 | 128 samples (NanoQuant defaults): uniform | γ=0.15 |
|---|---|---|---|---|---|
| pre-KD (block 27) | 31.42 ± 1.29 | 28.35 ± 0.04 | stopped early | 34.75 ± 2.18 | stopped early |
| **post-KD** | pending | 24.11 ± 0.14 | pending | 30.90 ± 2.79 | pending |

The 128-sample columns are NanoQuant's exact published setting (128 × 2048 calibration tokens, 8/8/8 epochs; the paper
reports 27.56 for Qwen3-0.6B at 1 bit, single run).

<details><summary>Per-block PPL, all runs, with the per-block bit budget of each allocation</summary>

| block | bits γ=.15 | bits γ=.3 | uniform #1 | uniform #2 | γ=0.15 #1 | γ=0.15 #2 | γ=0.3 #1 | γ=0.3 #2 | uniform@128 #1 | uniform@128 #2 | γ=0.15@128 #1 | γ=0.15@128 #2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.25 | 1.55 | 18.64 | 17.22 | 13.94 | 13.89 | 13.33 | 13.33 | 18.10 | 18.07 | 14.21 | 14.13 |
| 1 | 1.16 | 1.32 | 14.10 | 14.20 | 13.61 | 13.62 | 13.36 | 13.40 | 14.27 | 14.29 | 13.91 | 13.69 |
| 2 | 0.96 | 0.92 | 15.56 | 15.69 | 15.20 | 15.26 | 14.89 | 14.94 | 18.55 | 18.23 | 19.12 | 18.42 |
| 3 | 0.82 | 0.66 | 15.04 | 16.48 | 15.05 | 15.01 | 15.11 | 15.07 | 16.06 | 19.27 | 19.23 | 16.96 |
| 4 | 0.76 | 0.57 | 15.22 | 16.55 | 15.55 | 15.49 | 16.02 | 15.88 | 16.16 | 18.96 | 20.01 | 17.81 |
| 5 | 0.78 | 0.60 | 15.57 | 16.99 | 16.11 | 16.06 | 16.93 | 16.91 | 16.55 | 19.00 | 20.70 | 18.18 |
| 6 | 0.80 | 0.63 | 15.59 | 16.95 | 16.19 | 16.12 | 17.01 | 16.93 | 16.77 | 19.01 | 20.85 | 18.07 |
| 7 | 0.81 | 0.65 | 15.79 | 17.12 | 16.31 | 16.26 | 17.02 | 16.89 | 16.89 | 19.18 | 21.14 | 18.07 |
| 8 | 0.83 | 0.68 | 15.83 | 17.21 | 16.35 | 16.32 | 16.95 | 16.94 | 17.00 | 19.18 | 21.19 | 18.16 |
| 9 | 0.85 | 0.71 | 16.08 | 17.45 | 16.68 | 16.64 | - | - | 17.24 | 19.74 | - | - |
| 10 | 0.87 | 0.74 | 16.35 | 17.76 | 17.09 | 17.04 | - | - | 17.64 | 19.93 | - | - |
| 11 | 0.89 | 0.78 | 16.60 | 18.00 | 17.33 | 17.34 | - | - | 17.91 | 20.26 | - | - |
| 12 | 0.90 | 0.80 | 16.79 | 18.20 | 17.52 | 17.53 | - | - | 18.17 | 20.63 | - | - |
| 13 | 0.91 | 0.82 | 17.02 | 18.43 | 17.75 | 17.75 | - | - | 18.46 | 21.00 | - | - |
| 14 | 0.93 | 0.85 | 17.20 | 18.67 | 17.92 | 17.93 | - | - | 18.62 | 21.16 | - | - |
| 15 | 0.96 | 0.91 | 17.51 | 18.95 | 18.25 | 18.25 | - | - | 18.85 | 21.34 | - | - |
| 16 | 1.00 | 0.98 | 17.95 | 19.42 | 18.70 | 18.71 | - | - | 19.41 | 21.88 | - | - |
| 17 | 1.04 | 1.07 | 18.41 | 19.95 | 19.12 | 19.07 | - | - | 19.93 | 22.39 | - | - |
| 18 | 1.08 | 1.15 | 18.80 | 20.41 | 19.47 | 19.41 | - | - | 20.37 | 23.01 | - | - |
| 19 | 1.11 | 1.22 | 19.45 | 21.16 | 19.96 | 19.94 | - | - | 21.56 | 24.25 | - | - |
| 20 | 1.14 | 1.27 | 20.08 | 21.92 | 20.49 | 20.37 | - | - | 22.27 | 24.97 | - | - |
| 21 | 1.15 | 1.30 | 20.68 | 22.52 | 20.96 | 20.80 | - | - | 23.02 | 25.89 | - | - |
| 22 | 1.15 | 1.31 | 21.25 | 23.11 | 21.41 | 21.31 | - | - | 23.67 | 26.56 | - | - |
| 23 | 1.15 | 1.30 | 21.89 | 23.86 | 21.95 | 21.89 | - | - | 24.37 | 27.40 | - | - |
| 24 | 1.14 | 1.29 | 22.59 | 24.66 | 22.62 | 22.46 | - | - | 25.22 | 28.37 | - | - |
| 25 | 1.14 | 1.27 | 23.61 | 25.70 | 23.48 | 23.32 | - | - | 26.26 | 29.74 | - | - |
| 26 | 1.15 | 1.31 | 25.24 | 27.50 | 24.79 | 24.65 | - | - | 27.93 | 31.59 | - | - |
| 27 | 1.16 | 1.32 | 30.13 | 32.71 | 28.39 | 28.31 | - | - | 32.57 | 36.94 | - | - |

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

## Cost

The covariance ADMM is ~4× the diagonal's factorization time (42 s vs 8 s per block here), which is
negligible against Step 3. It adds one eigendecomposition of Σ per input group per block (in float64 —
much cheaper on a datacentre GPU than on an L4) and one eigendecomposition of the mid-dimension per
ADMM iteration (6–23 ms at these ranks).
