#!/usr/bin/env bash
# Screen for the block-3 PPL step (RESULTS.md §3): covariance-arm variants on blocks 0..9 only, no model KD,
# PPL after every block.  Blocks 0..9 of the full runs (q06_tuned_*) are exact baselines because block i
# depends only on blocks < i.  All variants share the same statistics cache.  Runs all variants concurrently.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_b3}
CACHE=${CACHE:-stats_q06.pt}
BLOCKS=${BLOCKS:-0,1,2,3,4,5,6,7,8,9}
EPOCHS=${EPOCHS:-8}
[ -f "$CACHE" ] || { echo "missing $CACHE (run run_full_tuned_parallel.sh first)"; exit 1; }
ATTN=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj

declare -A V=(
  [diag_rep]="--arm nanoquant"
  [cov_b0]="--arm cov"
  [cov_b25]="--arm cov --cov_corr_shrink 0.25"
  [cov_b50]="--arm cov --cov_corr_shrink 0.5"
  [cov_b75]="--arm cov --cov_corr_shrink 0.75"
  [cov_attn]="--arm cov --cov_layers $ATTN"
  [cov_oproj]="--arm cov --cov_layers self_attn.o_proj"
  [cov_mlp]="--arm cov --cov_layers mlp.gate_proj,mlp.up_proj,mlp.down_proj"
)
for v in "${!V[@]}"; do
  out=llm_ext/results/${TAG}_${v}.json
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) launching $v: ${V[$v]} ==="
  venv/bin/python llm_ext/run_llm_ext.py ${V[$v]} --num_calib_samples 128 --stats_cache "$CACHE" \
      --tune_nonfact --nonfact_epochs $EPOCHS --tune_fact --fact_epochs $EPOCHS \
      --ppl_after_block --only_blocks "$BLOCKS" \
      --out "$out" > llm_ext/logs/${TAG}_${v}.log 2>&1 &
done
wait
echo "=== $(date +%T) screen finished ==="
echo SCREENDONE
