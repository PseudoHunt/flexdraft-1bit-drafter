#!/usr/bin/env bash
# Idea H: sensitivity-aware rank allocation at equal bpw.  Full 28 blocks, 512 calibration samples, diagonal arm,
# EPOCHS=2 for the block loop (matched steps); model KD = NanoQuant default (128 samples x 8 epochs; the 512-sample
# runs with the per-block multipliers from block_bits.py (--gamma 0.15 --smooth 3, reference: diag 128 run 1).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_ra}; CACHE=${CACHE:-stats_q06_n512.pt}; N=512; EPOCHS=${EPOCHS:-2}
MULTS="1.2543,1.1569,0.9645,0.8188,0.7585,0.7790,0.7988,0.8139,0.8277,0.8456,0.8668,0.8861,0.9017,0.9121,0.9284,0.9578,0.9977,1.0398,1.0774,1.1097,1.1363,1.1475,1.1512,1.1458,1.1416,1.1352,1.1506,1.1575"
[ -f "$CACHE" ] || { echo "missing $CACHE"; exit 1; }
for v in uniform alloc; do for r in 1 2; do
  out=llm_ext/results/${TAG}_${v}_r${r}.json; [ -f "$out" ] && { echo "skip $out"; continue; }
  extra=""; [ "$v" = alloc ] && extra="--block_bits $MULTS"
  echo "=== $(date +%T) launching $v run $r ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm nanoquant --num_calib_samples $N --stats_cache "$CACHE" \
      --tune_nonfact --nonfact_epochs $EPOCHS --tune_fact --fact_epochs $EPOCHS --tune_model --model_kd_epochs 8 --kd_samples 128 \
      --ppl_after_block $extra --out "$out" > llm_ext/logs/${TAG}_${v}_r${r}.log 2>&1 &
done; done
wait; echo "=== $(date +%T) done ==="; echo RADONE
