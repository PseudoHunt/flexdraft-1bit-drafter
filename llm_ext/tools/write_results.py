#!/usr/bin/env python3
import os, json, re
T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.json")))
S = T["stats"]; p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RESULTS.md"); s = open(p).read()
has_search = "search" in T
sec3 = f"""## 3. Full model + full Step 3 — two runs per arm

All 28 blocks, `tune_nonfact` 8 + `tune_fact` 8 + model KD 8 (NanoQuant's defaults), 128 calibration samples,
wikitext2 PPL evaluated after every block (`--ppl_after_block`). **Each arm was run twice** with identical seed,
statistics cache and code (`run_full_tuned_parallel.sh`, then again with `TAG=q06_tuned_rep`); both arms of a pair
ran concurrently on one H200.

{T["full"]}

**Verdict: no detectable difference between the two objectives on full-model PPL at 0.973 bpw.** Covariance is
worse in both pairs (+{S['gaps'][0]:.2f}, +{S['gaps'][1]:.2f}), but the *within-arm* spread between two identical runs is
{S['dspread']:.1f} (diagonal) and {S['cspread']:.1f} (covariance) — twice the between-arm gap. Means {S['dmean']:.1f} vs {S['cmean']:.1f}, two
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

{T["rep_blocks"]}

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

{T["screen"]}

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
"""
if has_search:
    ch = T["search_chosen"]
    sec3 += f"""
### 4.4 Per-block β search on a validation split (`--cov_beta_search`)

For each block, Step 3 is run once per β ∈ {{0, 0.25, 0.5, 0.75, 1}}; the candidate with the lowest wikitext2
**validation**-split PPL (blocks 0..b quantised, rest FP) is kept. The validation split is disjoint from both the
calibration set (train) and the reported metric (test). β=1 uses the diagonal path directly. Cost: 5× the block loop.

{T["search"]}

Bold = chosen (validation PPL); last column is the chosen candidate's **test** PPL. Chosen β: {", ".join(f"{b:g}" for b in ch)} —
never 0 (full covariance) and never 1 (diagonal). Validation ranks candidates the way test does (β=1 is last at blocks 0
and 2, where the diagonal runs are worst on test). At block 3 the five candidates spanned 1.3 validation PPL and the
search landed at 19.40 — inside the band every chain lands in; it cannot manufacture a 16 when none of five draws produces
one. From block 4 it recovers block by block (19.40 → 17.65 → … ) while every fixed-β chain drifts up, and finishes ten
blocks at **{T['search_final']:.3f}** against the diagonal's 17.24 / 18.90 and plain covariance's 19.43 / 18.83.

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
"""
start = s.index("## 3. Full model + full Step 3"); end = s.index("## Cost")
s = s[:start] + sec3 + "\n" + s[end:]
s = s.replace("§1–2 on one NVIDIA L4-24GB, §3 on one H200 (both arms concurrently).", "§1–2 on one NVIDIA L4-24GB, §3–4 on one H200 (several runs concurrently under CUDA MPS).")
open(p, "w").write(s); print("RESULTS.md written; search section:", has_search)
