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

On a card with ≥80 GB (A100-80 / H200) use the parallel driver instead of `run_full_tuned.sh` —
it builds the shared statistics cache once, then runs both arms side by side:

```bash
bash llm_ext/run_full_tuned_parallel.sh        # H200: 72 min wall for both arms (diag 55 min, cov 72 min)
```

Each arm peaks well under 10 GB and the tuning loop is batch-1 (≈11% MFU on an L4), so the two
overlap well. `EPOCHS=4 bash llm_ext/run_full_tuned_parallel.sh` halves the Step-3 budget if needed.

Two more drivers (RESULTS.md §4; each needs `stats_q06.pt` from the run above):

```bash
bash llm_ext/run_block3_screen.sh     # blocks 0-9, no KD: replicates, corr-shrink beta sweep, per-projection subsets (H200: ~45 min, all concurrent)
bash llm_ext/run_beta_search.sh       # per-block beta search selected on the wikitext2 VALIDATION split (H200: ~1.5 h)
TAG=q06_tuned_rep bash llm_ext/run_full_tuned_parallel.sh   # replicate pair of the full runs (same seed + cache)
bash llm_ext/run_n512_screen.sh       # 512 calibration samples (new cache), blocks 0-9, 2 runs per arm, matched steps (H200: ~35 min)
bash llm_ext/run_rank_alloc.sh        # sensitivity-aware rank allocation vs uniform, full model + KD, 512 samples, 2 runs each (H200: ~75 min)
python llm_ext/block_bits.py --ref llm_ext/results/q06_tuned_nanoquant.json --gamma 0.15 --smooth 3   # the --block_bits multipliers
```

Further `run_llm_ext.py` flags: `--block_bits m0,...,m27` (per-block multipliers on `--bits`, mean 1 keeps the budget;
`block_bits.py` derives them from a run's per-block error at matched realised bpw), `--loss_norm {o_norm,unit,inv_var}`
(per-channel weighting of the block-tuning MSE), `--kd_samples N` (model KD on the first N calibration samples — the
teacher-logit cache is samples × seqlen × vocab: ~80 GB per process at 128 samples in fp16, >300 GB at 512. **This
container is capped at 300 GB RAM** (cgroup) although `free` reports 3 TB — run at most two KD stages concurrently), `--nonfact_batch_size` /
`--fact_batch_size` (gradient-accumulation steps), `--refresh_stats` (per block, re-estimate `i_norm` on the block's actual
compressed-prefix inputs before ADMM — one extra forward pass per block; RESULTS.md §7, the cheapest win found) and
`--delay_finalize` / `--joint_epochs` (keep every projection's latent sign factors alive until the whole block is binary,
then one joint pass — a negative result, §7).

New `run_llm_ext.py` flags behind them: `--cov_corr_shrink b` (shrink the input *correlation* toward I,
`C <- (1-b)C + bI`; `i_norm` untouched, b=1 is exactly the diagonal), `--cov_layers a,b,c` (covariance objective on
those projections only, diagonal elsewhere) and `--cov_beta_search 0,0.25,0.5,0.75,1` (per block, keep the candidate
with the lowest validation PPL; every candidate's block error and validation PPL is recorded in `block_stats`).

The parallel driver passes `--ppl_after_block`, which evaluates wikitext2 PPL after every block (blocks `0..b`
quantized, the rest FP) and records it next to the block reconstruction error in `block_stats`; `compare.py`
prints the two arms' per-block progression side by side. This is ~15 s per block on an H200 and is what
exposed the calibration-vs-test decoupling and the block-3 variance in `RESULTS.md` §3–4.

### Where the time goes (measured, L4)

236 s per block, of which the ADMM itself is ~8 s (diagonal) / ~42 s (covariance) — **97% is the
Step-3 tuning loop**: 14,336 block forward+backward passes per block at ~16 ms. The loop is
bandwidth- and launch-bound, not compute-bound, so a faster card helps roughly in proportion to
memory bandwidth rather than peak FLOPS.

Statistics are collected once and cached (`--stats_cache`), so both arms see identical `i_norm`,
`o_norm` and Σ; `admm_type` is the only difference between them.

## Results

Headline numbers (Qwen3-0.6B-Base, wikitext2 PPL, FP 12.669, NanoQuant paper 27.56 at 1 bit):
`--refresh_stats` at NanoQuant's own defaults **27.01 / 27.27** (uniform 28.10 / 33.70); `--block_bits` allocation at 512 samples
**24.25 / 23.97**; both at ≤0.973 bpw.

**Start with [`SUMMARY.md`](SUMMARY.md)** (findings, experiment ledger, idea list); full write-up in
[`RESULTS.md`](RESULTS.md); external research agenda with 64 catalogued ideas in [`agenda/`](agenda/).


See `RESULTS.md` (generated from `llm_ext/results/*.json`).
