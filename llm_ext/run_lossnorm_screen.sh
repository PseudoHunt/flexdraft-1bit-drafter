#!/usr/bin/env bash
# Idea A: per-channel-normalised block loss (loss_norm=inv_var) vs NanoQuant's o_norm weighting.
# Diagonal arm, 512 samples, EPOCHS=2 (matched steps), blocks 0..9, PPL after every block, two runs.
# Reference: q06_n512_nanoquant_r{1,2} (same settings, o_norm loss).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_ln}; CACHE=${CACHE:-stats_q06_n512.pt}; N=512; EPOCHS=${EPOCHS:-2}; BLOCKS=${BLOCKS:-0,1,2,3,4,5,6,7,8,9}; LN=${LN:-inv_var}
for r in 1 2; do
  out=llm_ext/results/${TAG}_${LN}_r${r}.json; [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) launching loss_norm=$LN run $r ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm nanoquant --num_calib_samples $N --stats_cache "$CACHE" \
      --tune_nonfact --nonfact_epochs $EPOCHS --tune_fact --fact_epochs $EPOCHS --loss_norm $LN \
      --ppl_after_block --only_blocks "$BLOCKS" --out "$out" > llm_ext/logs/${TAG}_${LN}_r${r}.log 2>&1 &
done
wait; echo "=== $(date +%T) done ==="; echo LNDONE
