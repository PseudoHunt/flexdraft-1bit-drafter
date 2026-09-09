# Handoff for the next session (written 2026-09-09, end of day 1)

You are continuing a research project: **beating NanoQuant (Samsung, arXiv 2602.06694) at 1-bit low-rank binary
quantisation of Qwen3-0.6B-Base at equal bits-per-weight, with replicates, toward a paper.** Everything lives in
`llm_ext/` on branch `llm-ext-cov`. Read, in this order, before doing anything:

1. `llm_ext/SUMMARY.md` — the four findings, experiment ledger E1–E17, idea ledger, caveats.
2. `llm_ext/RESULTS.md` §3–§7 — the per-block tables behind every claim.
3. `llm_ext/agenda/nanoquant-research-agenda.md` — an external research map (64 ideas, 6 paper candidates); its §2
   corrections to our claims were adopted. Its recommendation (binary-commitment mechanism) is the counter-position to
   our ledger; the user wants both pursued.

## Where we are (numbers = wikitext2 PPL, FP 12.669, NanoQuant paper 27.56 single run, all ours at ≤0.973 bpw)

| method | setting | post-KD PPL (2 runs) | status |
|---|---|---|---|
| NanoQuant uniform (baseline) | 128 samples, 8/8/8 epochs | 28.10 / 33.69 | done |
| `--refresh_stats` (re-estimate `i_norm` per block on the block's real inputs) | **NanoQuant's exact defaults** | **27.01 / 27.27** | done — cheapest win |
| `--block_bits` sensitivity-aware rank allocation (γ=0.15) | 512 samples, block loop EPOCHS=2 (matched steps), KD on 128 | **24.25 / 23.97** | done — best number |
| uniform at 512 samples | same as above | pre-KD only (30.13 / 32.71; 32.70 / 30.35) — **KD OOM-killed twice** | MISSING comparator |
| covariance ADMM objective (the original idea) | 128 and 512 | no detectable effect (within run-to-run noise) | closed as headline; revisit on blocks 0–2 only |
| delayed finalisation (within-block), `inv_var` loss, grad accumulation | — | negatives (§6.1, §7) | closed |

Key facts that took a day to learn: the factorisation objective is not where NanoQuant loses; Step-3 block tuning is, and
its run-to-run spread at 128 samples is 5.6 PPL (the paper reports single runs). Variance enters at block 3, where the
massive-activation channels form and the cached FP-chain statistics are wrong for the drifted inputs. Two replicates per
setting minimum; single-run gaps were shown to be noise twice.

## Tomorrow's queue (user's priorities, in order)

1. **Combine the winners:** `--refresh_stats --block_bits <mults>` at 512 samples, EPOCHS=2, KD on 128, **two runs**.
   Multipliers: `python llm_ext/block_bits.py --ref llm_ext/results/q06_tuned_nanoquant.json --gamma 0.15 --smooth 3`
   (consider deriving from a 512-sample run's error curve instead — untested).
2. **Uniform-512 post-KD comparator**, two runs — run at most TWO KD stages at a time (see memory cap below).
3. **Covariance objective on blocks 0–2 only** (`--arm cov --cov_layers ...` is per-projection; per-block needs a small
   change: dispatch on block index in `compress_block.factorize_and_replace` / `_compress_block_layers`), on top of
   refresh. Note Σ (`i_cov`) is NOT refreshed by `--refresh_stats` — add that (accumulate x xᵀ in `_refresh_block_stats`)
   if covariance is revisited; stale Σ is the suspected reason it never helped.
4. **Cross-block delayed finalisation** (keep block *i−1*'s latents alive through block *i*'s tuning) on top of refresh,
   with an equal-compute control. The within-block variant was negative (§7).
5. **Second model** (Llama-3.2-1B or Qwen3-1.7B) for the paper.
Also cheap and paper-necessary: 2×2 statistics × tuning-data control for §5; γ sweep {0.1, 0.2} for allocation.

## Fresh-machine setup (~5 min + caches)

```bash
git clone -b llm-ext-cov https://github.com/PseudoHunt/flexdraft-1bit-drafter.git && cd flexdraft-1bit-drafter
bash llm_ext/setup_llm_ext.sh          # NanoQuant @ a9e0a43 + llm_ext/nanoquant_cov.patch + venv + model
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
# statistics caches are gitignored (*.pt) and rebuilt by the drivers on first use:
#   stats_q06.pt (128 samples, ~1 min)  stats_q06_n512.pt (512 samples, ~4 min)
nvidia-cuda-mps-control -d             # lets concurrent runs overlap on one GPU (used all day on an H200)
```

Drivers: `run_full_tuned_parallel.sh` (both ADMM arms, full model), `run_n512_screen.sh`, `run_rank_alloc.sh`,
`run_beta_search.sh`, `run_block3_screen.sh`, `run_lossnorm_screen.sh`, `run_bs_screen.sh`. Every `run_llm_ext.py` flag is
documented in `llm_ext/README.md`. Results → `llm_ext/results/*.json` (per-block `block_stats` incl. `ppl` when run with
`--ppl_after_block`), logs → `llm_ext/logs/`. `python llm_ext/compare.py` prints all tables.

## Environment gotchas (each cost real time)

- **RAM is cgroup-capped at 300 GB** although `free` says 3 TB. KD's teacher-logit cache is ~80 GB per process at
  `--kd_samples 128` (fp16). Never run more than two KD stages concurrently; 512-sample KD without `--kd_samples` is >300 GB.
- The block loop is cheap in RAM/VRAM (~5–10 GB GPU per run); 6–8 concurrent runs on the H200 were fine (~2–4 min/block
  with sharing, ~80 s/block alone at 128 samples).
- `pgrep -f`/`pkill -f` with a pattern that also appears *literally* anywhere in the same shell command kills the calling
  shell (exit 144). Use a bracket pattern (`q06_b[s]_`) in a command that mentions the name nowhere else, or kill in a
  separate command.
- Never edit a bash driver script while it is running (bash reads incrementally; it clobbered a log and re-ran an entry).
- `o_norm` in NanoQuant is a backward-hook (output-gradient) importance, not output energy — do not overwrite it.
- Leaving latent factors trainable across phases accumulates gradients (per-phase optimisers only zero their own params);
  `tune_fact` now calls `block.zero_grad()` first.
- NanoQuant's Step 3 is optimizer-step-limited at its learning rates: gradient accumulation with fixed sample-passes
  under-trains badly (33.9 vs 18.1 at block 0).

## Working style the user expects

Per-block PPL after every block (`--ppl_after_block`) and the accumulating table re-shown on request; ≥2 replicates
before any claim; say plainly when an earlier reading was wrong; push results/logs/summary to GitHub continuously
(checkpoint commits, `git -c user.name="Dhiroo Pulipaka" -c user.email="dhiroo.pulipaka@gmail.com" commit`); concise
status between monitor events; ask before spending the last GPU hour on something speculative. Live tables:
`llm_ext/tools/*_table.py` (paths are relative to the repo); write-up generators `llm_ext/tools/write_sec6.py`,
`write_sec7.py`, `update_summary*.py` regenerate RESULTS.md §6/§7 and SUMMARY.md from the JSONs.
