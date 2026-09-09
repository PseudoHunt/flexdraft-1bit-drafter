# Summary for review — covariance ADMM, NanoQuant Step 3 diagnosis, and rank allocation (Qwen3-0.6B-Base, 0.973 bpw)

One day on one H200, 2026-09-09. Everything below is reproducible from this branch (`llm_ext/README.md`); every
number comes from `llm_ext/results/*.json` (`python llm_ext/compare.py`), and the full write-up with per-block tables
is `llm_ext/RESULTS.md` (§1–6). FP16 reference: wikitext2 PPL **12.669**. NanoQuant's paper reports **27.56** for
this model at 1 bit (single run, 128 calibration samples).

## The four findings

**1. The covariance ADMM objective does not change full-model PPL — and the reason is that the factorisation is not
where NanoQuant loses.** (§3–4) Replacing NanoQuant's diagonal weighting by the full input covariance cuts the
Σ-weighted output error it optimises by 39% (aggregate reproduced to 4 decimals across runs) and buys nothing on
held-out text: two runs per arm at the paper's default setting give diagonal 28.1 / 33.7, covariance 30.8 / 35.9 —
the within-arm spread (5–6 PPL) is twice the between-arm gap. the ADMM stage's aggregate errors reproduce to 4 decimals across runs (factor tensors were not hashed); the
run-to-run variance we observe arises in Step 3 (block tuning), mostly at one block (block 3) where the calibration block error collapses 10× and
the tuning objective stops discriminating between solutions that differ by 3–4 PPL on test. Fourteen runs through
that block at identical calibration error span 16–24 PPL; NanoQuant's paper reports single runs.

**2. More calibration data at the same optimizer budget removes the variance and improves both arms by 2–3 PPL.**
(§5) 512 samples (the paper's own Table-7 amount) with EPOCHS=2 (same sample-passes as 128 × 8): at ten blocks
diagonal 16.39 ± 0.07 (was 18.49 ± 1.25), covariance 16.37 ± 0.50 (was 19.64 ± 0.20). The two objectives tie.
Calibration error on the larger set stays *higher* while test PPL falls — less overfitting, more transfer. Caveat: this
screen changes the statistics cache *and* the tuning data together; a 2×2 (statistics × tuning data) control is still to do.

**3. Allocating rank by measured per-block sensitivity beats NanoQuant's published number at equal bpw: post-KD wikitext2
PPL 24.25 / 23.97 (two runs) vs the paper's 27.56 and our best uniform run 28.10.** (§6) NanoQuant gives
every block the same rank; the per-block PPL curve shows the damage is concentrated in blocks 0–2 and 17–27. Moving
bits there (`block_bits.py`, γ=0.15, realised bpw 0.9722 vs uniform 0.9729): pre-KD PPL **28.23 / 28.54 vs 30.13 /
32.71** (512 samples, two replicates each, allocated replicates within 0.05 PPL at 24 of 28 blocks). Post-KD (KD on NanoQuant's default 128 samples × 8 epochs): **24.25 / 23.97**. The uniform-512 post-KD comparator was
lost twice to the container's 300 GB RAM cap during KD (pre-KD comparison replicated: 30.13 / 32.71 and 32.70 / 30.35 vs
28.23 / 28.54 and 28.39 / 28.31); the γ=0.3 and 128-sample-default allocation runs were stopped at block 8 (γ=0.3 is
behind γ=0.15 in the 0.6-bit middle; at 128 samples the allocation is lottery-dominated). Caveat: the headline combines
512-sample block tuning at matched steps with the allocation.

**4. Re-estimating the input statistics on the compressed prefix beats NanoQuant's published number at NanoQuant's own
setting: full model, 128 samples, 8/8/8 epochs, post-KD wikitext2 PPL 27.01 (run 2 pending) vs paper 27.56 and our uniform
28.10 / 33.70 — one extra forward pass per block.**
(§7, last hour) NanoQuant's `i_norm` (and Σ) are collected once on the FP chain; blocks 1–2 are where the massive-activation
channels form, so the cached statistics are most wrong exactly where the tuner sees drifted inputs. One extra forward pass per
block to re-estimate `i_norm` on the block's actual inputs (`--refresh_stats`) gives block 2 PPL 14.79 / 14.75 (uniform 18.2–18.9)
and block 3 PPL 15.25 / 14.99 (uniform 16.1–20.7) at 128 samples — where only 4× more data had reached before. Full model (two runs, NanoQuant 8/8/8 at 128 samples): 30.89 pre-KD / **27.01 post-KD**. It also suggests the covariance objective was handicapped by stale Σ.

## Experiment ledger

| id | experiment | setting | result | where |
|---|---|---|---|---|
| E1 | ADMM only, both arms, 28 blocks | 128 samples | cov −39% Σ-out-err, +4% weight err; PPL unusable without Step 3 | §1 |
| E2 | one block (13) + Step 3 | 128 | cov 12.84 vs diag 12.87 — does not extrapolate | §2 |
| E3 | full model + Step 3 + KD, **2 runs/arm** | 128 | diag 28.1/33.7, cov 30.8/35.9: no detectable difference | §3 |
| E4 | per-block PPL instrumentation (`--ppl_after_block`) | all runs | block-3 step; flat "debt" carried 24 blocks; ADMM deterministic | §3, §4.1 |
| E5 | correlation shrinkage β ∈ {0,.25,.5,.75} (`--cov_corr_shrink`) | 128, blocks 0–9 | monotone on calibration error, scrambled on PPL; cond(C) drops 100–1000× so not conditioning | §4.2 |
| E6 | per-projection covariance (`--cov_layers`: attn / o_proj / MLP) | 128, blocks 0–9 | block-2 gain is MLP-borne; the block-3 step appears with any subset; MLP-only worst at block 9 | §4.3 |
| E7 | replicates: diagonal ×2 (10 blocks), both arms ×2 (full) | 128 | diagonal block-3 draws 16.06 / 17.87 / 19.27 / 20.7 — the "step" was a lucky draw | §4.1, §3 |
| E8 | per-block β search, best-of-5 on **validation** split (`--cov_beta_search`) | 128, blocks 0–9 | 17.69 at 10 blocks vs diag 17.24 / 18.90, cov 19.43 / 18.83; never picks β=0 or 1; only chain to recover from a bad block 3 | §4.4 |
| E9 | input-spectrum / massive-activation analysis on the cached Σ | — | 4 channels carry 37% of block-3 input energy (45% by block 5); o_norm importance is flat | §4.2, §4.5 |
| E10 | 512 calibration samples, 2 runs/arm, matched steps | 512, blocks 0–9 | see finding 2 | §5 |
| E11 | gradient accumulation 4× in Step 3 (`--fact_batch_size`) | 128, block 0 | 33.9 / 36.9 vs 18.1 — Step 3 is step-count-limited | §6.1 |
| E12 | per-channel-normalised block loss (`--loss_norm inv_var`) | 512, blocks 0–1 | +3 PPL at block 0, replicated — massive channels must be reconstructed | §6.1 |
| E13 | **sensitivity-aware rank allocation** γ=0.15 vs uniform, full model, 2 runs each (`--block_bits`) | 512 | pre-KD 28.23 / 28.54 (rerun 28.39 / 28.31) vs uniform 30.13 / 32.71 (rerun 32.70 / 30.35); **post-KD allocated 24.25 / 23.97**, uniform post-KD OOM-killed twice. Score is a heuristic, not a causal sensitivity | §6 |
| E14 | rank allocation γ=0.3, 2 runs | 512 | stopped at block 8: 16.95 / 16.94 vs γ=0.15's 16.35 / 16.32 — over-allocates | §6 |
| E16 | **delayed finalisation** (agenda idea 2, `--delay_finalize`): latents of all projections stay live until the block is binary, then a joint pass | 128, blocks 0–5 | block 0 ≈24 vs 18.1 (v1 and v2, 3 runs); block 3 17.26 / 22.70 vs 16.06 / 19.27 / 17.87 / 20.7 | §7 |
| E17 | **refreshed statistics** (agenda idea 22, `--refresh_stats`): `i_norm` re-estimated per block on the compressed-prefix inputs | 128, blocks 0–5 / 0–9 | block 2 14.79 / 14.75 vs 18.2–18.9; **block 3 15.25 / 14.99 vs 16.06 / 19.27 / 17.87 / 20.7**; block 5 16.10 / 15.82 vs 16.55 / 19.00 / 18.52; block 9 - vs 17.24 / 18.90; **full model (2 runs) 30.89 pre-KD / **27.01 post-KD**** | §7 |
| E15 | rank allocation γ=0.15 at NanoQuant's exact defaults, 2 runs | 128, 8/8/8 | stopped at block 8: 21.19 / 18.16 — replicates 3 apart, the 128-sample lottery dominates | §6 |

## What was built (all in `llm_ext/`, NanoQuant changes in `nanoquant_cov.patch` against `a9e0a43`)

* Covariance ADMM (`admm_cov.py`) with correlation-only shrinkage; per-layer objective dispatch; per-block β search
  with validation-split selection; per-block PPL + block error recording; per-block rank multipliers; loss re-weighting;
  KD subset + fp16 teacher cache; gradient-accumulation flags. Every flag is documented in `README.md`.
* Drivers: `run_full_tuned_parallel.sh`, `run_block3_screen.sh`, `run_beta_search.sh`, `run_n512_screen.sh`,
  `run_rank_alloc.sh`, `run_bs_screen.sh`, `run_lossnorm_screen.sh`; `block_bits.py` for the allocation;
  `compare.py` for all tables.

## Research agenda (external review)

`agenda/nanoquant-research-agenda.md` (+ `.csv` of the 64 ideas, `.html` render) is an independent research map written
against commit `1d36d8d`: what the experiments establish, corrections to over-strong claims in this summary (several
adopted above), prior work constraining novelty for each direction (DBF, PV-Tuning, GuidedQuant/YAQA, CBQ, QEP/CoreQ,
CALDERA, SpinQuant, MatQuant, …), 64 catalogued ideas with first experiments and cost, six paper candidates, and a staged
plan. Its recommendation — build the paper on a demonstrated binary-tuning failure mechanism (delayed finalisation /
function-space sign updates) and treat rank allocation and more data as strong baselines — is the counter-position to
the ledger below and should be read alongside it.

## Idea ledger for tomorrow (ranked)

1. **Compose the winners**: rank allocation derived from the 512-sample curve + 512 samples + validation-selected β.
2. **Covariance where it helps**: covariance objective on blocks 0–2 only (it wins block 2 at every data size), diagonal
   elsewhere, on top of 1. Also: why KD recovers less on covariance-initialised weights (0.2 vs 3.2 PPL in run 2).
3. **Allocation ablations**: γ sweep, source curve (128 vs 512), per-projection allocation.
4. **Scaling curve**: 256 / 1024 / 2048 samples at matched steps, with replicates (now cheap: `--kd_samples`).
5. **Cheap selection proxy**: does held-out block error pick the same β as validation PPL? (drops E8 from 5× to ~1.2×)
6. **Best-of-K restarts, fixed β** — how much of the block-3 lottery is recoverable by selection alone.
7. **Massive-activation input columns kept in FP** (4 of 1024 per projection at 16 bits = +0.0625 bpw — not negligible; must be paid for out of the binary rank).
8. **A second model** (Llama-3.2-1B / Qwen3-1.7B) — required for a paper.

## Caveats a reviewer should know

* Single model (Qwen3-0.6B-Base), single bit-width, wikitext2 only; n=2 per setting except where stated.
* "Matched steps" runs use EPOCHS=2 at 512 samples so that optimizer steps equal NanoQuant's 128 × 8; the paper's own
  512-sample runs use 8 epochs — we have not run that (4× the compute).
* bpw is over the factorised matrices only (as in NanoQuant); embeddings / norms / lm_head are FP.
* The rank-allocation multipliers were derived from a *128-sample* run's error curve with a first-guess γ; not tuned.
* First rank-allocation attempt's KD was OOM-killed (teacher-logit cache); the per-block data was salvaged from logs
  (`*_preKD.json`) and the runs repeated with KD on 128 samples.
