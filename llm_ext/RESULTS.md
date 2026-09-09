# Covariance ADMM inside NanoQuant — Qwen3-0.6B-Base results

All numbers from `llm_ext/results/*.json` (regenerate the tables with `python llm_ext/compare.py`).
One NVIDIA L4-24GB. wikitext2, seqlen 2048, `bits=1.0` → **0.973 bpw** over the factorized matrices,
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

Pending (`llm_ext/run_full_tuned.sh`): all 28 blocks, `tune_nonfact` 8 + `tune_fact` 8 + model KD 8.
~236 s/block on an L4, ≈2 h per arm; 97% of that is the tuning loop, ~8 s is the ADMM itself.

## Cost

The covariance ADMM is ~4× the diagonal's factorization time (42 s vs 8 s per block here), which is
negligible against Step 3. It adds one eigendecomposition of Σ per input group per block (in float64 —
much cheaper on a datacentre GPU than on an L4) and one eigendecomposition of the mid-dimension per
ADMM iteration (6–23 ms at these ranks).
