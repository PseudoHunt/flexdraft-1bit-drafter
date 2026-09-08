"""FlexDraft drafter -> NanoQuant ADMM-only 1-bit quantization -> tau before/after.

Pipeline:
  1. tau (avg accepted length / drafting step) with the released bf16 drafter
  2. activation calibration (i_norm) on gsm8k TRAIN prompts, through the real draft path
  3. NanoQuant factorize_admm_nanoquant on every drafter attention projection
     (ADMM step only: no tune_nonfact, no tune_fact, no model-level KD)
  4. tau again
The bonus-guided post calibrator (anchor_bias_mlp) is left in bf16.
"""
import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_common import SCRATCH, load_models, build_prompts, measure_tau, load_admm_nq
from flexdraft.draft_model import FlexDraftModel

nq = load_admm_nq()
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------- calibration
_CAL = {"on": False, "in_sum": defaultdict(lambda: None), "n": defaultdict(int)}


def _acc(key, x):
    v = x.detach().float().reshape(-1, x.shape[-1]).square().mean(0)
    cur = _CAL["in_sum"][key]
    _CAL["in_sum"][key] = v if cur is None else cur + v
    _CAL["n"][key] += 1


def install_calibration(draft):
    """Capture the true inputs of the draft attention projections.

    q/k/v share the mask-token slice of input_layernorm(hidden); in the released
    path they are folded into a fused F.linear, so a module hook never fires --
    we wrap _dual_attn_layer_forward instead. o_proj *is* called as a module.
    """
    orig = FlexDraftModel._dual_attn_layer_forward

    def wrapped(self, target_layer, draft_layer, hidden_states, candidate_len,
                attention_mask, position_embeddings, past_key_value, fused_qkv=None):
        if _CAL["on"]:
            x = target_layer.input_layernorm(hidden_states)
            _acc((draft_layer._cal_idx, "qkv"), x[:, candidate_len:, :])
        return orig(self, target_layer=target_layer, draft_layer=draft_layer,
                    hidden_states=hidden_states, candidate_len=candidate_len,
                    attention_mask=attention_mask, position_embeddings=position_embeddings,
                    past_key_value=past_key_value, fused_qkv=fused_qkv)

    FlexDraftModel._dual_attn_layer_forward = wrapped

    handles = []
    for i, lyr in enumerate(draft.layers):
        lyr._cal_idx = i
        handles.append(lyr.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp, idx=i: _acc((idx, "o"), inp[0]) if _CAL["on"] else None))
    return handles


def collect_i_norm(draft, target, tokenizer, prompts, block_size, max_new_tokens,
                   thr, strategy, shrinkage):
    _CAL["on"] = True
    for j, ids in enumerate(prompts):
        draft.dual_attn_parallel_generate(
            target=target, input_ids=ids, mask_token_id=tokenizer.mask_token_id,
            max_new_tokens=max_new_tokens, stop_token_ids=[tokenizer.eos_token_id],
            temperature=0.0, block_size=block_size, is_profiling=True,
            draft_confidence_threshold=thr, pruning_strategy=strategy)
        if (j + 1) % 16 == 0:
            print(f"    calib {j+1}/{len(prompts)}", flush=True)
    _CAL["on"] = False

    stats = {}
    for key, s in _CAL["in_sum"].items():
        t = (s / max(_CAL["n"][key], 1)).clone()
        if 0.0 < shrinkage < 1.0:                       # NanoQuant get_shrunk_stats
            t.mul_(1.0 - shrinkage).add_(t.mean() * shrinkage)
        stats[key] = t
    return stats


# ------------------------------------------------------------------ ADMM step
def rank_for(in_f, out_f, bits):
    """Mirrors NanoQuant utils.calculate_ranks (_get_rank/_finalize_rank/_validate_rank),
    num_scales=2 for admm_type='nanoquant'."""
    r = (in_f * out_f * bits) / (in_f + out_f) - 16
    r = (int(r) // 32) * 32
    r = 32 if r == 0 else max(r, 32)
    if r > min(in_f, out_f):
        r = min(in_f, out_f)
    return r


@torch.no_grad()
def quantize_drafter(draft, stats, bits, outer_iters, inner_iters, reg, scheduler, min_rank, projs=PROJ, ranks=None):
    report = []
    for i, lyr in enumerate(draft.layers):
        for pname in projs:
            lin = getattr(lyr.self_attn, pname)
            W = lin.weight.data
            out_f, in_f = W.shape
            r = (ranks or {}).get(pname) or rank_for(in_f, out_f, bits)
            assert r > min_rank, f"rank {r} <= {min_rank} for layer {i}.{pname}"
            i_norm = stats[(i, "o" if pname == "o_proj" else "qkv")].to(W.device)
            assert i_norm.numel() == in_f
            o_norm = torch.ones(out_f, device=W.device, dtype=torch.float32)

            t0 = time.time()
            res = nq.factorize_admm_nanoquant(
                W, i_norm, o_norm, mid_rank=r,
                outer_iters=outer_iters, inner_iters=inner_iters, reg=reg,
                is_transpose=(out_f < in_f), rho_scheduler=scheduler)
            dt = time.time() - t0

            Wf = res["W_final"]
            err = (Wf.float() - W.float()).square().sum().item()
            nrm = W.float().square().sum().item()
            bpw = (r + 16) * (in_f + out_f) / (in_f * out_f)
            W.copy_(Wf.to(W.dtype))

            rec = {"layer": i, "proj": pname, "shape": [out_f, in_f], "rank": r,
                   "bpw": bpw, "rel_err": err / nrm, "sec": dt}
            report.append(rec)
            print(f"  [ADMM] L{i:02d}.{pname:7s} {out_f}x{in_f} r={r} "
                  f"bpw={bpw:.4f} rel_err={err/nrm:.4f} ({dt:.1f}s)", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--eval-samples", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--draft-confidence-threshold", type=float, default=0.01)
    ap.add_argument("--pruning-strategy", default="cumulative_product")
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--calib-new-tokens", type=int, default=128)
    ap.add_argument("--calib-shrinkage", type=float, default=0.4)
    ap.add_argument("--bits", type=float, default=1.0)
    ap.add_argument("--admm-outer-iters", type=int, default=400)
    ap.add_argument("--admm-inner-iters", type=int, default=5)
    ap.add_argument("--admm-reg", type=float, default=3e-2)
    ap.add_argument("--admm-scheduler", default="linear")
    ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    target, draft, tok = load_models(a.target, a.draft)
    dev = target.device
    out = {"args": vars(a), "target_layer_ids": list(draft.target_layer_ids),
           "anchor_bias_rank": draft.anchor_bias_rank}

    eval_prompts = build_prompts(tok, a.dataset, a.eval_samples, dev)   # gsm8k test
    ev = dict(block_size=a.block_size, max_new_tokens=a.max_new_tokens,
              temperature=a.temperature, thr=a.draft_confidence_threshold,
              strategy=a.pruning_strategy)

    print(f"\n===== 1) tau, released bf16 drafter ({len(eval_prompts)} prompts) =====", flush=True)
    out["tau_fp"] = measure_tau(target, draft, tok, eval_prompts, tag="fp16", **ev)
    print(json.dumps({k: v for k, v in out["tau_fp"].items() if k != "per_sample_tau"}, indent=2), flush=True)

    print(f"\n===== 2) calibration ({a.calib_samples} gsm8k TRAIN prompts) =====", flush=True)
    install_calibration(draft)
    from datasets import load_dataset as _ld
    raw = _ld("openai/gsm8k", "main", split="train").shuffle(seed=a.seed).select(range(a.calib_samples))
    cal_prompts = []
    from flexdraft.utils import _MATH_FINAL_ANSWER_INSTRUCTION
    for ex in raw:
        msgs = [{"role": "user", "content": ex["question"] + _MATH_FINAL_ANSWER_INSTRUCTION}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        cal_prompts.append(tok.encode(txt, return_tensors="pt").to(dev))
    t0 = time.time()
    stats = collect_i_norm(draft, target, tok, cal_prompts, a.block_size,
                           a.calib_new_tokens, a.draft_confidence_threshold,
                           a.pruning_strategy, a.calib_shrinkage)
    print(f"  calibration done in {time.time()-t0:.0f}s, {len(stats)} stat tensors", flush=True)

    print(f"\n===== 3) NanoQuant ADMM (bits={a.bits}, {a.admm_outer_iters} outer iters) =====", flush=True)
    t0 = time.time()
    out["admm"] = quantize_drafter(draft, stats, a.bits, a.admm_outer_iters,
                                   a.admm_inner_iters, a.admm_reg, a.admm_scheduler, a.min_rank)
    out["admm_total_sec"] = time.time() - t0
    print(f"  ADMM total {out['admm_total_sec']:.0f}s", flush=True)

    print(f"\n===== 4) tau, 1-bit ADMM drafter =====", flush=True)
    out["tau_q"] = measure_tau(target, draft, tok, eval_prompts, tag="1bit", **ev)
    print(json.dumps({k: v for k, v in out["tau_q"].items() if k != "per_sample_tau"}, indent=2), flush=True)

    print(f"\n########## tau {out['tau_fp']['tau']:.3f} -> {out['tau_q']['tau']:.3f} ##########")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
