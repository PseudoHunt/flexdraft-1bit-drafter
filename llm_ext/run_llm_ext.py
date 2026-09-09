#!/usr/bin/env python
"""Driver for docs/llm_extension.md: NanoQuant's diagonal ADMM vs the full-covariance ADMM on a small LLM.

One process = one arm.  Calibration statistics (i_norm, o_norm and the full input covariances) are
collected once and cached on disk, so both arms see bit-identical statistics and only `admm_type` differs.

  python llm_ext/run_llm_ext.py --model_id Qwen/Qwen3-0.6B-Base --arm nanoquant --out results/x.json
"""
import argparse, json, os, time

import torch

# NanoQuant has a circular import: modules.linear must come first.
import nanoquant.modules.linear  # noqa: F401
from nanoquant.core import compress_block
from nanoquant.core import compress_model as compress_model_mod
from nanoquant.core.compress_model import compress_block_recon, compress_model_recon
from nanoquant.core.importance import collect_stats, get_shrunk_stats, register_stats
from nanoquant.modules.quant_config import NanoQuantConfig
from nanoquant.utils.data_utils import get_calib_loader, prepare_dataset
from nanoquant.utils.eval_utils import evaluate_model
from nanoquant.utils.load_utils import load_model, load_tokenizer
from nanoquant.utils.utils import (calculate_ranks, cleanup_memory, find_layers, get_decoder_layers,
                                   get_layers_to_factorize, set_seed)


def factorized_bpw(model, quant_config):
    """bits/weight over the factorized matrices only: rank*(a+b) sign bits + 16*(a+b) scale bits."""
    layers_to_factorize = get_layers_to_factorize(model.config.model_type)
    ranks = calculate_ranks(model, layers_to_factorize, quant_config)
    bits = params = 0
    for i, layer in enumerate(get_decoder_layers(model)):
        sub = find_layers(layer)
        for name in layers_to_factorize:
            if name not in sub:
                continue
            a, b = sub[name].in_features, sub[name].out_features
            r = ranks[f"{i}.{name}"]
            bits += r * (a + b) + 16 * (a + b)
            params += a * b
    return {"bpw": bits / params, "params": params, "ranks": {k: int(v) for k, v in ranks.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--arm", default="nanoquant", choices=["nanoquant", "cov", "fp"])
    ap.add_argument("--bits", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_calib_samples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib_shrinkage", type=float, default=0.4)
    ap.add_argument("--calib_dataset", default="wikitext2")
    ap.add_argument("--admm_outer_iters", type=int, default=400)
    ap.add_argument("--tune_nonfact", action="store_true")
    ap.add_argument("--nonfact_epochs", type=int, default=8)
    ap.add_argument("--tune_fact", action="store_true")
    ap.add_argument("--fact_epochs", type=int, default=8)
    ap.add_argument("--tune_model", action="store_true")
    ap.add_argument("--model_kd_epochs", type=int, default=8)
    ap.add_argument("--stats_cache", default="")
    ap.add_argument("--cov_eig_device", default="cuda")
    ap.add_argument("--ppl_task", default="wikitext2")
    ap.add_argument("--zeroshot_task", default="")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max_blocks", type=int, default=-1, help="debug: only compress the first N blocks")
    ap.add_argument("--only_blocks", default="", help="comma-separated block indices to quantize; the rest stay FP")
    ap.add_argument("--ppl_after_block", action="store_true",
                    help="evaluate wikitext2 PPL after each block is quantized (recorded in block_stats)")
    ap.add_argument("--stats_only", action="store_true",
                    help="collect and cache the calibration statistics (incl. covariances), then exit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    quant_config = NanoQuantConfig(
        model_id=args.model_id, bits=args.bits, seed=args.seed, num_calib_samples=args.num_calib_samples,
        calib_dataset=args.calib_dataset, calib_shrinkage=args.calib_shrinkage, seqlen=args.seqlen,
        admm_type=("nanoquant" if args.arm == "fp" else args.arm), admm_outer_iters=args.admm_outer_iters,
        tune_nonfact=args.tune_nonfact, nonfact_epochs=args.nonfact_epochs, tune_fact=args.tune_fact,
        fact_epochs=args.fact_epochs, tune_model=args.tune_model, model_kd_epochs=args.model_kd_epochs,
        cov_eig_device=args.cov_eig_device, ppl_after_block=args.ppl_after_block,
    )
    if args.only_blocks:
        quant_config['block_indices'] = [int(x) for x in args.only_blocks.split(",")]

    model = load_model(args.model_id, args.seqlen, device_map="cpu")
    record = {"args": vars(args), "quant_config": {k: v for k, v in quant_config.items()}}

    if args.arm != "fp":
        record["alloc"] = factorized_bpw(model, quant_config)
        print(f"factorized bpw = {record['alloc']['bpw']:.4f}")
        fp_model = load_model(args.model_id, args.seqlen, device_map="cpu")

        data = prepare_dataset(args.model_id, quant_config)
        tokenizer = load_tokenizer(args.model_id)
        dataloader = get_calib_loader(data, tokenizer, quant_config['num_calib_samples'], quant_config['seed'],
                                      quant_config['seqlen'])

        # ---- calibration statistics (shared between arms via the cache) ----
        t = time.time()
        if args.stats_cache and os.path.exists(args.stats_cache):
            print(f"Loading cached calibration statistics from {args.stats_cache}")
            raw_stats = torch.load(args.stats_cache, map_location="cpu", weights_only=False)
        else:
            raw_stats = collect_stats(model, dataloader, "cuda", strategy=quant_config['calib_strategy'],
                                      collect_cov=True, cov_device="cpu")
            raw_stats = {k: ({kk: vv.cpu() for kk, vv in v.items()} if isinstance(v, dict) else v)
                         for k, v in raw_stats.items()}
            raw_stats['stats_device'] = 'cpu'
            if args.stats_cache:
                torch.save(raw_stats, args.stats_cache)
                print(f"Saved calibration statistics to {args.stats_cache}")
        record["calib_time"] = time.time() - t
        model.cpu(); cleanup_memory()

        if args.stats_only:
            print(f"stats_only: calibration statistics ready in {args.stats_cache} "
                  f"({record['calib_time']:.0f}s); exiting before compression.")
            return

        shrunk_stats = get_shrunk_stats(raw_stats, shrinkage=quant_config['calib_shrinkage'])
        model = register_stats(model, shrunk_stats)

        if args.max_blocks > 0:
            blocks = get_decoder_layers(model)
            fp_blocks = get_decoder_layers(fp_model)
            keep = min(args.max_blocks, len(blocks))
            model.model.layers = torch.nn.ModuleList(list(blocks)[:keep])
            fp_model.model.layers = torch.nn.ModuleList(list(fp_blocks)[:keep])
            print(f"[debug] truncated to {keep} blocks")

        t = time.time()
        model = compress_block_recon(model, fp_model, dataloader, quant_config)
        record["compress_time"] = time.time() - t
        record["layer_stats"] = list(compress_block.LAYER_STATS)
        record["block_stats"] = list(compress_model_mod.BLOCK_STATS)
        n = max(1, len(record["layer_stats"]))
        record["mean_weight_err"] = sum(s["weight_err"] for s in record["layer_stats"]) / n
        oes = [s["out_err"] for s in record["layer_stats"] if s["out_err"] is not None]
        record["mean_out_err"] = (sum(oes) / len(oes)) if oes else None
        print(f"mean weight err {record['mean_weight_err']:.4f} | mean out err "
              f"{record['mean_out_err'] if record['mean_out_err'] is None else round(record['mean_out_err'], 4)}")

        if args.tune_model:
            t = time.time()
            model = compress_model_recon(model, fp_model, dataloader, quant_config)
            record["kd_time"] = time.time() - t
        del fp_model
        cleanup_memory()
    else:
        tokenizer = load_tokenizer(args.model_id)

    model.eval().cuda()
    t = time.time()
    record["eval"] = evaluate_model(model=model, tokenizer=tokenizer, tasks_str=args.zeroshot_task,
                                    eval_ppl=args.ppl_task, num_fewshot=0, limit=args.limit, batch_size="auto")
    record["eval_time"] = time.time() - t
    record["total_time"] = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({"arm": args.arm, "eval": record["eval"], "mean_out_err": record.get("mean_out_err"),
                      "mean_weight_err": record.get("mean_weight_err"), "total_time": record["total_time"]}, indent=1))


if __name__ == "__main__":
    main()
