# 1-bit FlexDraft drafter via NanoQuant ADMM

Post-training binarization of the [FlexDraft](https://github.com/Yaro1214/FlexDraft) speculative-decoding drafter
(Qwen3-8B target) with [NanoQuant](https://github.com/SamsungLabs/NanoQuant)'s ADMM binary factorization, plus the
tuning stage that actually recovers acceptance length. One day of experiments on a single A100-80GB; every number
in [`RESULTS.md`](RESULTS.md) is generated from the JSON files in [`results/`](results/). A one-page summary for readers in a hurry is [`docs/memo.html`](docs/memo.html) (open locally or via [htmlpreview](https://htmlpreview.github.io/?https://github.com/PseudoHunt/flexdraft-1bit-drafter/blob/master/docs/memo.html)).

**Headline.** The drafter's 40 attention projections (800 MiB bf16) go to **56 MiB at 1.13 bpw** and keep
**91.3 % of the FP drafter's acceptance length τ** (6.68 vs 7.31); the same pipeline on NanoQuant's diagonal objective
gets 87.0 %, and the covariance objective alone — no tuning at all — gets 86.5 % at 0.99 bpw. The target model and
FlexDraft's bonus-guided post-calibrator are untouched, so generation stays lossless by construction — quantizing the
drafter trades only speed/memory, never output quality. 95 % was the goal and was **not** reached; see *What did not work*.

## Method (four steps)

1. **Activation-calibrated ADMM with the full input covariance** (`work/admm_cov.py`). NanoQuant's binary factorization
   `W ≈ diag(s₁)·S_A·S_B·diag(s₂)` (ranks 2016 q/o, 800 k/v) but minimising ‖(W−AB)·L‖ with LLᵀ = Σ, the covariance of
   the drafter's real mask-token inputs, instead of NanoQuant's diagonal ‖(W−AB)·diag(√i_norm)‖. Same alternating
   structure, same Z/U/export steps; the A-step is whitened and the B-step is solved exactly as a Sylvester equation
   via eigendecompositions. With Σ diagonal the code path is bit-identical to NanoQuant (`work/test_admm_cov.py`).
   The drafter's inputs have effective rank ≈ 700/4096, and this step halves held-out output error (0.091 → 0.047)
   and lifts ADMM-only τ from 5.53 to 6.33 (t = 6.4). 3.4× slower than diagonal ADMM (~8 min for the drafter).
2. **Residual base** — a second factorization of `W − Ŵ₁` at rank 256/96 with the same objective, +0.13 bpw.
3. **Target-as-teacher tuning** — plain cross-entropy between the drafter's mask-token logits and the *target's*
   greedy tokens, on contexts the target itself generated. NanoQuant's own refinement (MSE against the FP drafter)
   recovers 4 % of the gap; the target as teacher recovers 35–41 % on the diagonal init and a further +0.22 τ on the
   covariance init. The FP drafter is not the ground truth — the target is, and it's already running.
4. (Ablation baseline) NanoQuant's diagonal `i_norm`, which itself is worth 7 points over the uniform fallback.

τ as % of FP: uniform-`i_norm` ADMM **69 %** → calibrated diagonal **76 %** → **covariance ADMM 86.5 %** → + residual **88 %** → + target-teacher tuning **91.3 %**.

## What did not work (all with paired statistics over 40 prompts, see RESULTS.md)

Acceptance-weighted surrogate vs plain CE (t = 0.2) · tuning binary latents vs scales vs both (each alone reaches the same τ)
· latent LR up to 3e-4 (t = 1.5; 1e-3 hurts) · 4× more calibration windows (t = 0.5) · budget-neutral rank reallocation
across q/k/v/o (t ≤ 0.7 despite o_proj being 5× more sensitive than q) · group-128 scales (−1 % weight error, not run)
· fine-tuning the FP post-calibrator (t = −0.7) · randomized-Hadamard input rotation (hurts, t = −3.5: diagonal `i_norm`
cannot express the rotated importance) · top-256 soft-label KD (t = −0.1) · early stopping.

The tuned drafter fits its calibration windows *beyond* the FP drafter (train prefill-τ 12.2 vs FP 10.8) while held-out
stays at ≈ 9.0–9.4 under every tuning variant. What moved that plateau was not the tuning stage but the factorization
objective: the diagonal-`i_norm` ceiling of 87 % became 91.3 % once ADMM minimised output error under the real input
covariance. The remaining gap to 95 % is open; the levers not in this repo are more bits, higher precision on `o_proj`,
or QAT with the drafter's training pipeline.

## Setup

```bash
./setup.sh            # clones NanoQuant + FlexDraft at the pinned commits, venv (transformers==4.57.3), Qwen3-8B, drafter ckpt
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
T=$(ls -d hf/hub/models--Qwen--Qwen3-8B/snapshots/*); D=ckpt/flexdraft-qwen3-8b
```

Requires a CUDA GPU with ~40 GB free. The drafter checkpoint comes from the FlexDraft authors' Google Drive link (see their README).

## Reproduce

| what | command | output |
|---|---|---|
| ADMM-only τ, fused path (§1) | `venv/bin/python work/run_all.py --target $T --draft $D --eval-samples 40 --calib-samples 128 --out main.json` | `results/main.json` |
| per-projection sensitivity (§2) | `venv/bin/python work/ablate_proj.py --target $T --draft $D --out ablate.json` | `results/ablate.json` |
| rank reallocation (§3) | `venv/bin/python work/alloc.py --target $T --draft $D --out alloc.json` | `results/alloc.json` |
| `tune_fact` vs target-teacher (§4) | `work/run_tune_fact.py` then `work/run_target_teacher.py --lrs 1e-5,1e-4,1e-3 --modes ce,acc` | `tf.json`, `tt.json` |
| on-policy + decoupled LR sweep (§5) | `venv/bin/python work/run_onpolicy.py --target $T --draft $D --latent-lrs 1e-5,1e-4,3e-4,1e-3 --modes acc,ce --out op.json` | `results/op.json` |
| 2×2 scales/latents, 4k windows (§5) | `run_onpolicy.py --pairs 1e-5:0,0:3e-4 --modes ce` · `--train-windows 4096 --gen-prompts 7000 --pairs 1e-5:1e-4 --modes ce` | `op2x2.json`, `op4k.json` |
| residual base (§6) | `venv/bin/python work/run_residual.py --target $T --draft $D --cache full_cache.pt --out res.json` | `results/res.json` |
| o_norm from target loss (§8 A) | `venv/bin/python work/run_onorm.py --target $T --draft $D --cache full_cache_toks.pt --out onorm.json` | `results/onorm.json` |
| covariance ADMM gate (§8 B) | `venv/bin/python work/run_covgate.py --target $T --draft $D --cache full_cache_toks.pt --save-cov cov_sides.pt --out covgate2.json` | `results/covgate2.json` |
| **covariance pipeline (§8 B, best)** | `venv/bin/python work/run_covfull.py --target $T --draft $D --cache full_cache_toks.pt --load-cov cov_sides.pt --out covfull.json` | `results/covfull.json` |
| unit tests for `admm_cov.py` | `venv/bin/python work/test_admm_cov.py 400` | — |
| calibrator / Hadamard / KD arms (§6) | `work/run_final.py ... --tune-calibrator` · `--rotate hadamard` · `--mode kd --kd-topk 256 --early-stop` | `calib.json`, `hadamard.json`, `kd.json` |

Defaults reproduce the reported settings (40 eval prompts = GSM8K test[0:40], 8 dev prompts = test[40:48], seed 0).
`--cache` stores calibration statistics and on-policy windows (~2 GB) so later arms skip the 12-minute data build.

## Things to know before trusting or extending this

* **τ** is FlexDraft's *Avg Acceptance Length* (mean accepted tokens per drafting step, bonus included), computed by the
  released `dual_attn_parallel_generate` at temperature 0. FlexDraft's code never calls it τ.
* **Two code paths.** `NanoQuantLinear` has no `.weight`, so FlexDraft's fused-QKV path is bypassed for anything quantized
  (`force_unfused`). The unfused FP baseline is 7.312 vs 7.101 fused (bf16 summation order). Every comparison in
  RESULTS.md is within one path.
* **Repo bug in FlexDraft as released:** the checkpoint's config key is `dflash_config`; `flexdraft/utils.py` reads
  `flexdraft_config`, so `scripts/inference.py` fails. `work/fd_common.py` maps the key at load time.
* **NanoQuant import order:** `nanoquant.modules.linear` must be imported before `nanoquant.core.*` (circular import).
* `o_norm` (NanoQuant's gradient statistic) is 1 everywhere: FlexDraft ships no training loss to backpropagate.
  NanoQuant's `tune_nonfact` is structurally unavailable — the fused block's remaining FP weights belong to the target.
* The drafter's `layers.N.mlp.*`, `layers.N.*_layernorm` and `norm` exist in the checkpoint but are never executed by the
  released forward; only `self_attn.{q,k,v,o}_proj` (and the tiny q/k norms, mask embedding, post-calibrator) matter.
* Weights are written back / evaluated in bf16 — this measures the accuracy of the 1-bit representation, not kernel speed.
  The FP post-calibrator alone is 164 MiB, three times the binarized drafter.

## Where this goes next

[`NEXT_STEPS.md`](NEXT_STEPS.md) ranks the remaining ideas for the drafter (sequential covariance for o_proj, covariance-aware residual/scales, BoA-style two-sided weighting, o_proj rank) and lists what not to repeat. [`docs/llm_extension.md`](docs/llm_extension.md) is the plan for testing the covariance objective inside NanoQuant's own LLM pipeline (Qwen3-0.6B/1.7B, PPL + zero-shot, judged after Step 3). Tuned weights of the 91.3 % model and the covariance factors are attached to the `v0.1-day1` release.

## Layout

```
work/        admm_cov.py (full-covariance ADMM) + test_admm_cov.py, run_covgate.py, run_covfull.py, run_onorm.py,
             fd_common.py (loading, τ), run_all.py (ADMM + i_norm calibration), run_tune_fact.py, run_target_teacher.py,
             run_onpolicy.py, run_residual.py, run_final.py (calibrator / rotation / KD arms), run_g128.py,
             ablate_proj.py, alloc.py, gap_probe.py, admm_probe.py, smoke.py, gen_results.py (regenerates RESULTS.md)
results/     one JSON per run (all per-prompt τ included), cleaned logs in results/logs/
RESULTS.md   every table, generated by work/gen_results.py from results/*.json
docs/memo.html  one-page results memo (print-ready)
docs/llm_extension.md  plan for the NanoQuant-on-LLMs extension
NEXT_STEPS.md  ranked follow-ups and process notes
setup.sh     environment + upstream repos at the exact commits used
```

Upstream: NanoQuant (Apache-2.0, commit `a9e0a43`), FlexDraft (no license file in the repo at commit `07b869b`; see the
authors), Qwen3-8B (Apache-2.0). Scripts here are thin drivers around those two codebases; this repo has no license file yet.
