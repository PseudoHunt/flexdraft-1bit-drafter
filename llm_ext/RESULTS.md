# Covariance ADMM inside NanoQuant — Qwen3-0.6B-Base results

All numbers from `llm_ext/results/*.json` (regenerate the tables with `python llm_ext/compare.py`).
§1–2 on one NVIDIA L4-24GB, §3 on one H200 (both arms concurrently). wikitext2, seqlen 2048, `bits=1.0` → **0.973 bpw** over the factorized matrices,
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

## 3. Full model + full Step 3

All 28 blocks, `tune_nonfact` 8 + `tune_fact` 8 + model KD 8 (NanoQuant's defaults), 128 calibration
samples. Both arms ran **concurrently on one H200** from the same statistics cache
(`llm_ext/run_full_tuned_parallel.sh`), with wikitext2 PPL evaluated after every block
(`--ppl_after_block`; blocks `0..b` quantized, the rest FP).

| | bpw | **wikitext2 PPL** | Δ vs FP (12.669) | pre-KD PPL | mean weight err | mean Σ-weighted out err | block loop | KD |
|---|---|---|---|---|---|---|---|---|
| NanoQuant diagonal | 0.973 | **28.105** | +15.436 | 32.567 | **0.3158** | 0.1598 | 47 min | 7.5 min |
| covariance | 0.973 | 30.817 | +18.148 | 34.119 | 0.3287 | **0.0977** (−39%) | 65 min | 6.2 min |

**The covariance objective does not beat NanoQuant's diagonal on full-model PPL at 0.973 bpw**:
30.817 vs 28.105, **+2.713 PPL (9.7%) worse**, while cutting the Σ-weighted output error it optimises by
39%. Neither arm is close to FP: both are ~2.2–2.4× the reference PPL, and KD recovers only
4.5 / 3.3 points of the ~20-point pre-KD deficit. The one-block result of §2 (+0.2 over FP)
does not extrapolate to the full model — quantizing 28 blocks sequentially is a different problem. Single seed, no error bars; but the per-block
record below is consistent enough that the sign of the result is not in doubt.

### Per-projection mean over the 28 blocks (ADMM init, this run)

| projection | diag weight err | cov weight err | Δ | diag out err | cov out err | Δ |
|---|---|---|---|---|---|---|
| `self_attn.q_proj` | 0.2567 | 0.2646 | +3.1% | 0.0653 | 0.0425 | -35% |
| `self_attn.k_proj` | 0.2894 | 0.2984 | +3.1% | 0.0804 | 0.0523 | -35% |
| `self_attn.v_proj` | 0.3381 | 0.3477 | +2.8% | 0.2658 | 0.1721 | -35% |
| `self_attn.o_proj` | 0.3138 | 0.3340 | +6.5% | 0.2099 | 0.0978 | -53% |
| `mlp.gate_proj` | 0.3257 | 0.3400 | +4.4% | 0.0698 | 0.0448 | -36% |
| `mlp.up_proj` | 0.3526 | 0.3651 | +3.6% | 0.1937 | 0.1228 | -37% |
| `mlp.down_proj` | 0.3343 | 0.3511 | +5.0% | 0.2335 | 0.1519 | -35% |

Same trade as §1: every projection buys a 35–53% cut in Σ-weighted output error with 3–6.5% more weight error. The price is
highest exactly where the gain is largest (`o_proj`: +6.5% weight err for −53% out err), i.e. where Σ is furthest from diagonal.

### Per-block progression (the diagnostic that explains the result)

`err` = relative block output error after Step-3 tuning, `‖q_block(x_q) − fp_block(x_fp)‖² / ‖fp_block(x_fp)‖²`, on the
128 **calibration** sequences (wikitext2 *train*). `PPL` = wikitext2 **test** with blocks `0..b` quantized and `b+1..27` FP.
The two columns are measured on disjoint data.

| block | diag err | cov err | Δ | diag PPL | cov PPL | Δ PPL |
|---|---|---|---|---|---|---|
| 0 | 0.1242 | 0.1076 | -13.4% | 18.095 | 16.592 | -1.503 |
| 1 | 0.0953 | 0.0860 | -9.7% | 14.266 | 14.193 | -0.073 |
| 2 | 0.0149 | 0.0058 | -60.9% | 18.548 | 16.029 | -2.519 |
| 3 | 0.0014 | 0.0014 | -3.3% | 16.057 | 20.036 | +3.979 |
| 4 | 0.0017 | 0.0016 | -5.2% | 16.160 | 20.004 | +3.844 |
| 5 | 0.0023 | 0.0021 | -5.6% | 16.548 | 20.330 | +3.782 |
| 6 | 0.0027 | 0.0026 | -5.9% | 16.767 | 19.817 | +3.050 |
| 7 | 0.0031 | 0.0029 | -6.4% | 16.890 | 19.730 | +2.840 |
| 8 | 0.0036 | 0.0034 | -6.2% | 17.003 | 19.570 | +2.567 |
| 9 | 0.0041 | 0.0039 | -6.3% | 17.239 | 19.434 | +2.195 |
| 10 | 0.0053 | 0.0049 | -6.7% | 17.638 | 19.491 | +1.852 |
| 11 | 0.0066 | 0.0062 | -6.3% | 17.914 | 19.904 | +1.989 |
| 12 | 0.0071 | 0.0066 | -6.3% | 18.175 | 20.021 | +1.846 |
| 13 | 0.0081 | 0.0076 | -6.3% | 18.463 | 20.494 | +2.031 |
| 14 | 0.0087 | 0.0082 | -6.0% | 18.624 | 20.783 | +2.159 |
| 15 | 0.0110 | 0.0103 | -6.4% | 18.846 | 20.743 | +1.897 |
| 16 | 0.0175 | 0.0163 | -6.6% | 19.409 | 21.255 | +1.846 |
| 17 | 0.0239 | 0.0221 | -7.4% | 19.927 | 21.911 | +1.984 |
| 18 | 0.0304 | 0.0281 | -7.6% | 20.371 | 22.228 | +1.857 |
| 19 | 0.0419 | 0.0384 | -8.2% | 21.558 | 23.530 | +1.972 |
| 20 | 0.0495 | 0.0453 | -8.5% | 22.273 | 24.088 | +1.815 |
| 21 | 0.0546 | 0.0497 | -8.9% | 23.020 | 24.901 | +1.881 |
| 22 | 0.0533 | 0.0486 | -8.8% | 23.669 | 25.609 | +1.940 |
| 23 | 0.0536 | 0.0490 | -8.6% | 24.371 | 26.326 | +1.955 |
| 24 | 0.0486 | 0.0445 | -8.6% | 25.217 | 27.057 | +1.840 |
| 25 | 0.0486 | 0.0445 | -8.4% | 26.258 | 28.240 | +1.982 |
| 26 | 0.0467 | 0.0423 | -9.4% | 27.934 | 30.036 | +2.102 |
| 27 | 0.0678 | 0.0619 | -8.7% | 32.567 | 34.119 | +1.552 |

Covariance has **lower block error on all 28 blocks** (block 3 by only 3%, a tie at 4 d.p.) and **worse held-out PPL on 25 of 28**.
Three regimes:

1. **Blocks 0–2** — the high-error early blocks (err 0.12 → 0.015). Covariance wins *both* columns, by up to −61% err / −2.5 PPL.
2. **Block 3 — the step.** Block error collapses 10× for both arms to ≈0.0014 (cov 3% lower), yet the diagonal's PPL *drops* 2.5 and the
   covariance arm's *rises* 4.0. The entire final deficit is incurred here.
3. **Blocks 10–27 — steady state.** Covariance is 5–9% lower on block error on every row, and the PPL gap is flat: mean **+1.92 ± 0.13**
   over 18 blocks with no trend, while the number of covariance-quantized blocks went from 11 to 28. If the lower block error bought *any*
   held-out PPL the gap would shrink; if the objective were harmful per block it would grow. It does neither — **the block-error advantage
   converts to zero PPL**, and the debt from block 3 is carried unchanged to the end. Both arms then take the same steep hit in the last ~9
   blocks, where block error rises 10× again (0.005 → 0.05–0.07) and the last block alone costs +4.6 / +4.1 PPL.

The Σ-weighted objective is a strictly better fit to the calibration set on every block, and that fit does not transfer to test text.
Two mechanisms fit this, both testable:

* **Σ is over-parameterised for 128 samples.** The full covariance has ~n²/2 free entries against the diagonal's n, from 262k tokens at
  `calib_shrinkage=0.4`. The diagonal is a heavily regularised estimator by accident; the full Σ can fit calibration-specific correlation
  structure. → raise shrinkage, more calibration samples, or eigenvalue-clip Σ.
* **The residual is pushed into low-variance directions.** The objective makes the error small where calibration input energy is high,
  concentrating it in directions that were quiet on the calibration set; test tokens that excite those directions take the damage.
  → measure block error on a *held-out* set alongside the calibration one; if the cov advantage vanishes there, this is the mechanism.

The per-projection table does not single out one projection type as the culprit — the trade is proportional everywhere — so the cheapest
next experiment is a **localisation** one: apply the covariance objective to one projection type at a time (starting with `o_proj`, whose Σ has
effective rank 54 of 2048 and the largest out-err gain) and watch the per-block PPL column for where the block-3 step appears. That is a
one-line change in the `admm_type` dispatch in `compress_block.py`, and it separates "the objective is wrong" from "Σ is badly estimated"
more cheaply than a shrinkage sweep.

## Cost

The covariance ADMM is ~4× the diagonal's factorization time (42 s vs 8 s per block here), which is
negligible against Step 3. It adds one eigendecomposition of Σ per input group per block (in float64 —
much cheaper on a datacentre GPU than on an L4) and one eigendecomposition of the mid-dimension per
ADMM iteration (6–23 ms at these ranks).
