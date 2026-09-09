# NanoQuant on an LLM with the covariance ADMM objective

Implementation of [`docs/llm_extension.md`](../docs/llm_extension.md): NanoQuant's binary factorization
minimises `||diag(no) (W - AB) diag(sqrt(i_norm))||` (the *diagonal* of the input second-moment); this
replaces it with `||diag(no) (W - AB) L||`, `L Lᵀ = Σ`, the **full input covariance** — i.e. the true
layer-output error `E||(W - AB)x||²`. Everything downstream (`tune_nonfact`, `tune_fact`, KD, kernels,
export) is unchanged.

Model: **Qwen3-0.6B-Base**, wikitext2, `bits=1.0` → **0.973 bpw** over the factorized matrices,
128 calibration samples × 2048 tokens, `calib_shrinkage=0.4`, 400 ADMM iterations, seed 0, one L4-24GB.

## What was changed in NanoQuant (`nanoquant_cov.patch`, against commit `a9e0a43`)

| file | change |
|---|---|
| `core/admm_cov.py` | new: `factorize_admm_cov`, `CovSide`, `shrink_cov` (from `work/admm_cov.py`) |
| `core/importance.py` | `collect_stats(collect_cov=True)` accumulates `E[xxᵀ]` under the *same* online clipping as `i_norm`; one Σ per group of linears sharing an input (q/k/v, gate/up); `register_stats` attaches it as an `i_cov` buffer |
| `core/compress_block.py` | `admm_type == "cov"` branch; cached eigendecomposition per group; per-layer Σ-weighted output-error diagnostic (recorded for *both* arms) |
| `core/compress_model.py` | `block_indices` (quantize only some blocks, the rest stay FP); per-block reconstruction error; optional per-block PPL |
| `modules/quant_config.py`, `main.py` | `cov` admm type, `cov_eig_device`, `ppl_after_block` |

`diag(shrink(Σ))` equals NanoQuant's shrunk `i_norm` by construction, and was verified equal to
**3e-6 relative** on the collected statistics — so the two arms differ *only* in the correlation factor.

## Correctness checks (`test_admm_cov_llm.py`)

* `C = I` path reproduces `factorize_admm_nanoquant` **bit-for-bit** (rel diff 0.0).
* Sylvester B-step satisfies its normal equations to 5e-7.
* On real Qwen3 weights with a correlated Σ, Σ-weighted error drops on every shape tested.

## Reproduce (fresh machine, ~5 min of setup)

```bash
git clone -b llm-ext-cov https://github.com/PseudoHunt/flexdraft-1bit-drafter.git
cd flexdraft-1bit-drafter
bash llm_ext/setup_llm_ext.sh     # NanoQuant @ a9e0a43 + patch + venv + Qwen3-0.6B-Base
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src

bash llm_ext/run_pair.sh          # FP + both ADMM-only arms   (L4: ~30 min)
bash llm_ext/run_oneblock.sh      # one block + full Step 3     (L4: ~11 min)
bash llm_ext/run_full_tuned.sh    # all 28 blocks + full Step 3 (L4: ~4 h; H200: ~40 min)
python llm_ext/compare.py         # tables from llm_ext/results/*.json
```

`setup_llm_ext.sh` needs no FlexDraft checkpoint and no Qwen3-8B — only the small base model.
The calibration statistics cache (`stats_q06_admmonly.pt`, 1.7 GB) is **not** in git; it is rebuilt
automatically on the first run in ~2 min. Both arms then read the same file, so they differ only in
`admm_type`.

On a card with ≥80 GB the two arms fit side by side; run them concurrently rather than sequentially
(each peaks well under 10 GB, and at batch 1 the GPU is far from saturated).

Statistics are collected once and cached (`--stats_cache`), so both arms see identical `i_norm`,
`o_norm` and Σ; `admm_type` is the only difference between them.

## Results

See `RESULTS.md` (generated from `llm_ext/results/*.json`).
