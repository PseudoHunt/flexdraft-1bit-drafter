"""Budget-neutral non-uniform rank allocation across q/k/v/o (ADMM only, calibrated i_norm, fused path).
Baseline ranks q2016 k800 v800 o2016 = 41.2M bits/layer.  Bits per proj = rank*(in+out)."""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_common import load_models, build_prompts, measure_tau
from run_all import install_calibration, collect_i_norm, quantize_drafter
from flexdraft.utils import _MATH_FINAL_ANSWER_INSTRUCTION
from datasets import load_dataset

CONFIGS = {  # name: (ranks, note)
    "baseline":      ({"q_proj": 2016, "k_proj": 800, "v_proj": 800, "o_proj": 2016}, "uniform-bpw NanoQuant ranks"),
    "q->o_small":    ({"q_proj": 1760, "k_proj": 800, "v_proj": 800, "o_proj": 2272}, "all r>512"),
    "q->o_large":    ({"q_proj": 1536, "k_proj": 800, "v_proj": 800, "o_proj": 2496}, "all r>512"),
    "k->o (k<512)":  ({"q_proj": 2016, "k_proj": 416, "v_proj": 800, "o_proj": 2240}, "VIOLATES r>512 on k"),
}
DIMS = {"q_proj": (4096, 4096), "k_proj": (4096, 1024), "v_proj": (4096, 1024), "o_proj": (4096, 4096)}
def bits(r): return sum((r[p] + 16) * sum(DIMS[p]) for p in r)
BASE = bits(CONFIGS["baseline"][0])

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
cal = [tok.encode(tok.apply_chat_template([{"role": "user", "content": ex["question"] + _MATH_FINAL_ANSWER_INSTRUCTION}],
       tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").to(dev) for ex in raw]
stats = collect_i_norm(draft, target, tok, cal, 16, 128, 0.01, "cumulative_product", 0.4)
print("calibrated", flush=True)
out = {}
for name, (ranks, note) in CONFIGS.items():
    draft.load_state_dict(fp_sd)
    rep = quantize_drafter(draft, stats, 1.0, 400, 5, 3e-2, "linear", 32, ranks=ranks)
    r = measure_tau(target, draft, tok, prompts, tag=name, **ev)
    r.update(ranks=ranks, note=note, bits_vs_baseline=bits(ranks) / BASE,
             rel_err={p: sum(x["rel_err"] for x in rep if x["proj"] == p) / 10 for p in ranks})
    out[name] = r
    print(f"### {name:14s} {ranks}  bits={r['bits_vs_baseline']:.4f}x  tau={r['tau']:.3f}±{r['tau_sem']:.3f}  "
          f"rel_err={ {p: round(v,3) for p,v in r['rel_err'].items()} }", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
print("wrote", a.out)
