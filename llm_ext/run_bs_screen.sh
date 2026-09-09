#!/usr/bin/env bash
# Is the block-3 lottery optimizer noise?  Diagonal arm, 128 samples, 8 epochs (same sample-passes as the baseline),
# but 4x gradient accumulation in Step 3 (nonfact 4->16, fact 1->8).  Two runs, blocks 0..9, PPL after every block.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_bs}; CACHE=${CACHE:-stats_q06.pt}; BLOCKS=${BLOCKS:-0,1,2,3,4,5,6,7,8,9}
NB=${NB:-16}; FB=${FB:-8}; ARM=${ARM:-nanoquant}
for r in 1 2; do
  out=llm_ext/results/${TAG}_${ARM}_r${r}.json; [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) launching $ARM run $r (nonfact_bs=$NB fact_bs=$FB) ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm $ARM --num_calib_samples 128 --stats_cache "$CACHE" \
      --tune_nonfact --nonfact_epochs 8 --nonfact_batch_size $NB --tune_fact --fact_epochs 8 --fact_batch_size $FB \
      --ppl_after_block --only_blocks "$BLOCKS" --out "$out" > llm_ext/logs/${TAG}_${ARM}_r${r}.log 2>&1 &
done
wait; echo "=== $(date +%T) done ==="; echo BSDONE
