"""Per-projection sensitivity: binarize ONLY q / k / v / o (all 10 layers), FP elsewhere."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_common import load_models, build_prompts, measure_tau

from run_all import install_calibration, collect_i_norm, quantize_drafter
from flexdraft.utils import _MATH_FINAL_ANSWER_INSTRUCTION
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True)
ap.add_argument("--eval-samples", type=int, default=40); ap.add_argument("--max-new-tokens", type=int, default=256)
ap.add_argument("--calib-samples", type=int, default=128); ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
target, draft, tok = load_models(a.target, a.draft); dev = target.device
ev = dict(block_size=16, max_new_tokens=a.max_new_tokens, temperature=0.0, thr=0.01, strategy="cumulative_product")
prompts = build_prompts(tok, "gsm8k", a.eval_samples, dev)
fp_sd = {k: v.detach().clone() for k, v in draft.state_dict().items()}

install_calibration(draft)
raw = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(a.calib_samples))
cal = []
for ex in raw:
    m = [{"role": "user", "content": ex["question"] + _MATH_FINAL_ANSWER_INSTRUCTION}]
    cal.append(tok.encode(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
               enable_thinking=False), return_tensors="pt").to(dev))
stats = collect_i_norm(draft, target, tok, cal, 16, 128, 0.01, "cumulative_product", 0.4)
print("calibrated", flush=True)

out = {}
for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    draft.load_state_dict(fp_sd)
    rep = quantize_drafter(draft, stats, 1.0, 400, 5, 3e-2, "linear", 512, projs=[proj])
    r = measure_tau(target, draft, tok, prompts, tag=proj, **ev)
    r["mean_rel_err"] = sum(x["rel_err"] for x in rep) / len(rep)
    out[proj] = r
    print(f"### only {proj} binarized: tau={r['tau']:.3f} +/- {r['tau_sem']:.3f}  rel_err={r['mean_rel_err']:.3f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
print("wrote", a.out)
