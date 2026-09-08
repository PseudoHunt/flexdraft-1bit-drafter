import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_common import load_models, build_prompts, measure_tau

T = os.environ["TARGET"]; D = os.environ["DRAFT"]
target, draft, tok = load_models(T, D)
print("target_layer_ids:", draft.target_layer_ids)
print("anchor bias:", draft.enable_anchor_bias_logits, "rank:", draft.anchor_bias_rank)
p = build_prompts(tok, "gsm8k", 2, target.device)
r = measure_tau(target, draft, tok, p, 16, 128, 0.0, 0.01, "cumulative_product", tag="smoke")
print(json.dumps({k:v for k,v in r.items() if k!='per_sample_tau'}, indent=2))
