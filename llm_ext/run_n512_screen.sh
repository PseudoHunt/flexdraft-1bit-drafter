#!/usr/bin/env bash
# 512 calibration samples (RESULTS.md 4.5): does more calibration data stop the block-3 collapse?
# Builds a 512-sample statistics cache, then runs both arms TWICE on blocks 0..9 (no KD), PPL after every block.
# EPOCHS=2 keeps the optimizer-step budget equal to the 128-sample/8-epoch baseline (1024 sample-passes per phase).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_n512}
N=${N:-512}
CACHE=${CACHE:-stats_q06_n${N}.pt}
BLOCKS=${BLOCKS:-0,1,2,3,4,5,6,7,8,9}
EPOCHS=${EPOCHS:-2}
if [ ! -f "$CACHE" ]; then
  echo "=== $(date +%T) collecting $N-sample calibration statistics -> $CACHE ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm nanoquant --num_calib_samples $N \
      --stats_cache "$CACHE" --stats_only --out /dev/null > llm_ext/logs/${TAG}_stats.log 2>&1 \
      || { echo "stats collection FAILED"; tail -20 llm_ext/logs/${TAG}_stats.log; exit 1; }
fi
echo "stats cache: $(ls -la $CACHE | awk '{print $5}') bytes"
for arm in nanoquant cov; do for r in 1 2; do
  out=llm_ext/results/${TAG}_${arm}_r${r}.json
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) launching $arm run $r ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm $arm --num_calib_samples $N --stats_cache "$CACHE" \
      --tune_nonfact --nonfact_epochs $EPOCHS --tune_fact --fact_epochs $EPOCHS \
      --ppl_after_block --only_blocks "$BLOCKS" \
      --out "$out" > llm_ext/logs/${TAG}_${arm}_r${r}.log 2>&1 &
done; done
wait
echo "=== $(date +%T) n$N screen finished ==="; echo N512DONE
