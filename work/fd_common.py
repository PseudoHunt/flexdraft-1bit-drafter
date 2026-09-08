"""Shared helpers: load FlexDraft target+draft, measure tau (avg acceptance length)."""
import importlib.util, json, os, sys, time
import numpy as np
import torch

SCRATCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCRATCH, "FlexDraft"))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from flexdraft import FlexDraftBiasModel, load_dataset_prompts


def load_admm_nq():
    """Import NanoQuant's admm_nq module directly (skips core/__init__, which
    pulls in cut_cross_entropy/gemlite that we don't need for the ADMM step)."""
    p = os.path.join(SCRATCH, "NanoQuant/src/nanoquant/core/admm_nq.py")
    spec = importlib.util.spec_from_file_location("nq_admm_nq", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_models(target_path, draft_path, device="cuda:0"):
    device = torch.device(device)
    target = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=torch.bfloat16).to(device).eval()

    draft_config = AutoConfig.from_pretrained(draft_path)
    # The released checkpoint stores its FlexDraft settings under `dflash_config`,
    # but flexdraft/utils.py:get_flexdraft_config reads `flexdraft_config`.
    if not getattr(draft_config, "flexdraft_config", None):
        legacy = getattr(draft_config, "dflash_config", None)
        assert legacy, "no flexdraft_config/dflash_config in draft config"
        draft_config.flexdraft_config = legacy
    draft = FlexDraftBiasModel.from_pretrained(
        draft_path, config=draft_config, dtype=torch.bfloat16).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    return target, draft, tokenizer


def build_prompts(tokenizer, dataset, n, device, split_offset=0):
    ds = load_dataset_prompts(dataset, None)
    idxs = list(range(split_offset, min(split_offset + n, len(ds))))
    out = []
    for i in idxs:
        msgs = [{"role": "user", "content": ds[i]["turns"][0]}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append(tokenizer.encode(text, return_tensors="pt").to(device))
    return out


@torch.inference_mode()
def measure_tau(target, draft, tokenizer, prompts, block_size, max_new_tokens,
                temperature, thr, strategy, tag=""):
    """tau = mean accepted length per drafting step (inference.py's 'Avg Acceptance Length')."""
    all_acc, per_sample, tpots = [], [], []
    for i, input_ids in enumerate(prompts):
        res = draft.dual_attn_parallel_generate(
            target=target, input_ids=input_ids,
            mask_token_id=tokenizer.mask_token_id,
            max_new_tokens=max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=temperature, block_size=block_size,
            is_profiling=True,
            draft_confidence_threshold=thr, pruning_strategy=strategy)
        all_acc.extend(res.acceptance_lengths)
        per_sample.append(float(np.mean(res.acceptance_lengths)))
        tpots.append(res.time_per_output_token)
        print(f"  [{tag}] sample {i:3d}: acc_len={per_sample[-1]:.3f} "
              f"steps={len(res.acceptance_lengths)} out_tok={res.num_output_tokens}",
              flush=True)
    a = np.array(all_acc, dtype=np.float64)
    return {
        "tau": float(a.mean()),
        "tau_sem": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
        "n_steps": int(a.size),
        "per_sample_tau": per_sample,
        "tau_macro": float(np.mean(per_sample)),
        "mean_tpot_ms": float(np.mean(tpots) * 1000),
    }
