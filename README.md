# 1-bit FlexDraft drafter via NanoQuant ADMM

Post-training binarization of the [FlexDraft](https://github.com/Yaro1214/FlexDraft) speculative-decoding drafter
(Qwen3-8B target) with [NanoQuant](https://github.com/SamsungLabs/NanoQuant)'s ADMM binary factorization, plus the
tuning stage that actually recovers acceptance length. One day of experiments on a single A100-80GB; every number
in [`RESULTS.md`](RESULTS.md) is generated from the JSON files in [`results/`](results/).

**Headline.** The drafter's 40 attention projections (800 MiB bf16) go to **56 MiB at 1.13 bpw** and keep
**87.0 % of the FP drafter's acceptance length τ** (6.36 vs 7.31); at 0.99 bpw it is 85.7 %. The target model and
FlexDraft's bonus-guided post-calibrator are untouched, so generation stays lossless by construction — quantizing the
drafter trades only speed/memory, never output quality. 95 % was the goal and was **not** reached; see *What did not work*.

## Method (three steps)

1. **Activation-calibrated ADMM** — NanoQuant `factorize_admm_nanoquant` with `i_norm` = per-input-channel second
   moments collected from real FlexDraft drafting (mask-token positions), `o_norm` = 1. Ranks from NanoQuant's
   `calculate_ranks` at `bits=1.0`: 2016 (q/o), 800 (k/v). Uniform `i_norm` costs ~7 points of τ.
2. **Optional residual base** — a second ADMM factorization of `W − Ŵ₁` at r₂ = 256/96 (+0.13 bpw). Worth +0.42 τ at init,
   +0.10 after tuning.
3. **Target-as-teacher tuning** — plain cross-entropy between the drafter's mask-token logits and the *target model's*
   greedy tokens, on **on-policy** contexts (the target's own continuations of GSM8K-train prompts). The FP drafter is
   *not* the teacher: NanoQuant's own `tune_fact` (MSE vs FP-drafter hidden states) recovers 4 % of the gap; the target
   as teacher recovers 35–41 %.

τ as % of FP: uniform-`i_norm` ADMM **69 %** → calibrated `i_norm` **76 %** → + residual base **81 %** → + target-teacher tuning **87 %**.

## What did not work (all with paired statistics over 40 prompts, see RESULTS.md)

Acceptance-weighted surrogate vs plain CE (t = 0.2) · tuning binary latents vs scales vs both (each alone reaches the same τ)
· latent LR up to 3e-4 (t = 1.5; 1e-3 hurts) · 4× more calibration windows (t = 0.5) · budget-neutral rank reallocation
across q/k/v/o (t ≤ 0.7 despite o_proj being 5× more sensitive than q) · group-128 scales (−1 % weight error, not run)
· fine-tuning the FP post-calibrator (t = −0.7) · randomized-Hadamard input rotation (hurts, t = −3.5: diagonal `i_norm`
cannot express the rotated importance) · top-256 soft-label KD (t = −0.1) · early stopping.

The tuned drafter fits its calibration windows *beyond* the FP drafter (train prefill-τ 12.4 vs FP 10.8) while held-out
stays at ≈ 9.0 under every variant. The ~87 % ceiling is a generalization limit of the ~1 bpw function class at
PTQ-scale data, not an optimization or data-quantity problem. Getting past it needs more bits, higher precision on
`o_proj`, or QAT with the drafter's training pipeline — none of which is in this repo.

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

## Layout

```
work/        fd_common.py (loading, τ), run_all.py (ADMM + i_norm calibration), run_tune_fact.py, run_target_teacher.py,
             run_onpolicy.py, run_residual.py, run_final.py (calibrator / rotation / KD arms), run_g128.py,
             ablate_proj.py, alloc.py, gap_probe.py, admm_probe.py, smoke.py
results/     one JSON per run (all per-prompt τ included), cleaned logs in results/logs/
RESULTS.md   every table, generated from results/*.json
setup.sh     environment + upstream repos at the exact commits used
```

Upstream: NanoQuant (Apache-2.0, commit `a9e0a43`), FlexDraft (no license file in the repo at commit `07b869b`; see the
authors), Qwen3-8B (Apache-2.0). Scripts here are thin drivers around those two codebases; this repo has no license file yet.
