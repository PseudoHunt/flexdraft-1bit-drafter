"""ADMM  ->  tune_fact  on the FlexDraft drafter, mirroring NanoQuant compress_block_recon.

The drafter has no standalone block: its attention projections live inside
FlexDraft's fused `_dual_attn_layer_forward(target_layer, draft_layer, ...)`.
So the "block" here is that fused pair, evaluated in the PREFILL configuration
(L clean tokens + block_size mask tokens).  With an empty KV cache,
past_key_value=None is exactly equivalent to what generate() does, which makes
the block a pure function of hidden_states -- what tune_fact needs.

Only the mask-token positions depend on draft weights (clean queries are masked
away from mask keys), so clean positions contribute exactly zero loss/gradient.

NanoQuant's factorize_and_replace / tune_fact / AdamW are used verbatim.
"""
import argparse, copy, json, os, sys, time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SC, "NanoQuant/src"))

from fd_common import load_models, build_prompts, measure_tau
from flexdraft.draft_model import FlexDraftModel

from nanoquant.modules.linear import NanoQuantLinear   # must precede core.* (circular import)
from nanoquant.core.compress_block import factorize_and_replace, tune_fact
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils.utils import find_layers

# NanoQuant's per-block ordering for llama/qwen3, restricted to attention
PROJ_ORDER = ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.k_proj"]


def force_unfused(draft):
    """NanoQuantLinear sets .weight = None, so _build_fused_qkv (which cats raw
    weights) cannot run. Route every layer through the module path instead;
    applied to ALL measurements so FP and quantized are compared like-for-like."""
    draft._build_fused_qkv = staticmethod(lambda tl, dl: [None] * len(list(tl)))


class DualAttnBlock(nn.Module):
    """One (target_layer, draft_layer) pair as a standalone, stateless block."""

    def __init__(self, dm, target_layer, draft_layer, cand_len, attn_mask, pos_emb):
        super().__init__()
        self._dm = [dm]                       # not a submodule
        self.target_layer = target_layer
        self.draft_layer = draft_layer
        self.cand_len, self.attn_mask, self.pos_emb = cand_len, attn_mask, pos_emb

    def forward(self, hidden_states, **kw):
        out = FlexDraftModel._dual_attn_layer_forward(
            self._dm[0], target_layer=self.target_layer, draft_layer=self.draft_layer,
            hidden_states=hidden_states, candidate_len=self.cand_len,
            attention_mask=self.attn_mask, position_embeddings=self.pos_emb,
            past_key_value=None, fused_qkv=None)
        return (out,)


@torch.no_grad()
def build_calibration(target, draft, tok, n_samples, seqlen, block_size, seed):
    """128 windows of `seqlen` real chat-formatted GSM8K train tokens, each with
    block_size mask embeddings appended -- the prefill layout the drafter sees."""
    from datasets import load_dataset
    from flexdraft.utils import _MATH_FINAL_ANSWER_INSTRUCTION
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)

    stream, i = [], 0
    need = n_samples * seqlen + seqlen
    while len(stream) < need and i < len(ds):
        ex = ds[i]; i += 1
        msgs = [{"role": "user", "content": ex["question"] + _MATH_FINAL_ANSWER_INSTRUCTION}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False) + ex["answer"] + tok.eos_token
        stream.extend(tok.encode(txt))
    assert len(stream) >= need, f"corpus too short: {len(stream)} < {need}"
    stream = torch.tensor(stream[:n_samples * seqlen], dtype=torch.long).view(n_samples, seqlen)

    dev = target.device
    dtype = target.model.embed_tokens.weight.dtype
    split = draft.target_layer_ids[0]
    from transformers import DynamicCache

    Q = seqlen + block_size
    mask_emb = draft.mask_embedding.to(dtype).view(1, 1, -1).expand(1, block_size, -1)
    inputs = torch.empty(n_samples, Q, target.config.hidden_size, dtype=dtype, device=dev)
    for j in range(n_samples):
        ids = stream[j:j + 1].to(dev)
        h = target.model.embed_tokens(ids)
        pos = torch.arange(seqlen, device=dev).unsqueeze(0)
        pe = target.model.rotary_emb(h, pos)
        cache = DynamicCache()
        for lyr in target.model.layers[:split]:
            h = lyr(hidden_states=h, attention_mask=None, position_embeddings=pe,
                    past_key_value=cache, use_cache=True)
        inputs[j] = torch.cat([h, mask_emb], dim=1)[0]
        del cache

    # shared prefill mask / rope for the L+block_size layout
    min_val = torch.finfo(dtype).min
    m = torch.zeros(1, 1, Q, Q, dtype=dtype, device=dev)
    m[:, :, :seqlen, :seqlen] = torch.triu(
        torch.full((seqlen, seqlen), min_val, dtype=dtype, device=dev), diagonal=1)
    m[:, :, :seqlen, seqlen:] = min_val
    pos = torch.arange(Q, device=dev).unsqueeze(0)
    pe = target.model.rotary_emb(torch.empty(1, 1, dtype=dtype, device=dev), pos)
    return inputs, m, pe


def rank_for(in_f, out_f, bits):
    r = (in_f * out_f * bits) / (in_f + out_f) - 16
    r = (int(r) // 32) * 32
    r = 32 if r == 0 else max(r, 32)
    return min(r, min(in_f, out_f))


def quantize_pass(draft, target, calib, mask, pe, cand_len, cfg, do_tune, importance,
                  min_rank, log):
    """Sequential per-layer ADMM (+ optional tune_fact), NanoQuant compress_block_recon style."""
    dev = target.device
    split = draft.target_layer_ids[0]
    student_in = calib.clone()
    teacher_in = calib.clone()

    for i, draft_layer in enumerate(draft.layers):
        tl = target.model.layers[split + i]
        fp_attn = copy.deepcopy(draft_layer.self_attn)          # FP teacher for this layer
        teacher_layer = SimpleNamespace(self_attn=fp_attn)

        with torch.no_grad():
            t_block = DualAttnBlock(draft, tl, teacher_layer, cand_len, mask, pe)
            teacher_out = torch.empty_like(teacher_in)
            for j in range(teacher_in.shape[0]):
                teacher_out[j:j + 1] = t_block(teacher_in[j:j + 1])[0]

        block = DualAttnBlock(draft, tl, draft_layer, cand_len, mask, pe)

        for name in PROJ_ORDER:
            lin = find_layers(draft_layer)[name]
            r = rank_for(lin.in_features, lin.out_features, cfg["bits"])
            assert r > min_rank, f"rank {r} <= {min_rank}"
            lin.register_buffer("i_norm", torch.ones(lin.in_features, device=dev), persistent=False)
            lin.register_buffer("o_norm", torch.ones(lin.out_features, device=dev), persistent=False)

            t0 = time.time()
            nano_linear, _ = factorize_and_replace(draft_layer, name, r, cfg)
            admm_s = time.time() - t0

            tune_s = 0.0
            if do_tune:
                t0 = time.time()
                tune_fact(block, nano_linear, student_in, teacher_out, importance, {}, cfg)
                tune_s = time.time() - t0
            else:
                nano_linear.finalize()
            log.append({"layer": i, "proj": name, "rank": r,
                        "admm_s": admm_s, "tune_s": tune_s})
            print(f"  L{i:02d}.{name.split('.')[1]:7s} r={r:5d} admm={admm_s:.1f}s "
                  f"tune={tune_s:.1f}s", flush=True)

        with torch.no_grad():
            for j in range(student_in.shape[0]):
                student_in[j:j + 1] = block(student_in[j:j + 1])[0]
        teacher_in = teacher_out
        del fp_attn, teacher_out
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--eval-samples", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--draft-confidence-threshold", type=float, default=0.01)
    ap.add_argument("--pruning-strategy", default="cumulative_product")
    ap.add_argument("--calib-samples", type=int, default=128)
    ap.add_argument("--calib-seqlen", type=int, default=256)
    ap.add_argument("--fact-epochs", type=int, default=8)
    ap.add_argument("--fact-binary-lr", type=float, default=1e-5)
    ap.add_argument("--fact-scale-lr", type=float, default=1e-5)
    ap.add_argument("--bits", type=float, default=1.0)
    ap.add_argument("--admm-outer-iters", type=int, default=400)
    ap.add_argument("--min-rank", type=int, default=512)
    ap.add_argument("--skip-admm-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = {"args": vars(a)}
    ev = dict(block_size=a.block_size, max_new_tokens=a.max_new_tokens, temperature=0.0,
              thr=a.draft_confidence_threshold, strategy=a.pruning_strategy)

    cfg = NanoQuantConfig(bits=a.bits, seed=a.seed, num_calib_samples=a.calib_samples,
                          admm_type="nanoquant", admm_outer_iters=a.admm_outer_iters,
                          admm_inner_iters=5, admm_penalty_scheduler="linear",
                          tune_nonfact=False, tune_fact=True, tune_model=False,
                          fact_epochs=a.fact_epochs, fact_batch_size=1,
                          fact_binary_lr=a.fact_binary_lr, fact_scale_lr=a.fact_scale_lr,
                          fact_bias_lr=1e-5)

    target, draft, tok = load_models(a.target, a.draft)
    force_unfused(draft)
    for p in target.parameters():
        p.requires_grad_(False)
    dev = target.device
    eval_prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev)

    print("\n===== A) tau, released bf16 drafter (unfused qkv path) =====", flush=True)
    out["tau_fp"] = measure_tau(target, draft, tok, eval_prompts, tag="fp16", **ev)
    print(f"  tau_fp = {out['tau_fp']['tau']:.3f}", flush=True)

    print("\n===== build calibration =====", flush=True)
    calib, mask, pe = build_calibration(target, draft, tok, a.calib_samples,
                                        a.calib_seqlen, a.block_size, a.seed)
    importance = torch.ones(target.config.hidden_size, device=dev)
    print(f"  calib {tuple(calib.shape)}", flush=True)

    if not a.skip_admm_only:
        print("\n===== B) ADMM only (NanoQuantLinear, binarized) =====", flush=True)
        sd = {k: v.clone() for k, v in draft.state_dict().items()}
        log_b = []
        t0 = time.time()
        quantize_pass(draft, target, calib, mask, pe, a.calib_seqlen, cfg,
                      False, importance, a.min_rank, log_b)
        out["admm_only_sec"] = time.time() - t0
        out["tau_admm_only"] = measure_tau(target, draft, tok, eval_prompts, tag="admm", **ev)
        print(f"  tau_admm_only = {out['tau_admm_only']['tau']:.3f}", flush=True)
        # restore FP drafter for pass C
        del draft
        torch.cuda.empty_cache()
        target2, draft, tok2 = load_models(a.target, a.draft)
        del target2, tok2
        force_unfused(draft)
        torch.cuda.empty_cache()

    print("\n===== C) ADMM + tune_fact =====", flush=True)
    log_c = []
    t0 = time.time()
    quantize_pass(draft, target, calib, mask, pe, a.calib_seqlen, cfg,
                  True, importance, a.min_rank, log_c)
    out["tune_fact_sec"] = time.time() - t0
    out["log_c"] = log_c
    out["tau_tuned"] = measure_tau(target, draft, tok, eval_prompts, tag="tuned", **ev)
    print(f"  tau_tuned = {out['tau_tuned']['tau']:.3f}", flush=True)

    f = out["tau_fp"]["tau"]
    print(f"\n########## tau  fp={f:.3f}"
          + (f"  admm_only={out['tau_admm_only']['tau']:.3f}" if "tau_admm_only" in out else "")
          + f"  tuned={out['tau_tuned']['tau']:.3f} ##########")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
