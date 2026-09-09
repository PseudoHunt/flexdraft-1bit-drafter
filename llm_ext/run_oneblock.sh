#!/usr/bin/env bash
# Single-block verification: quantize + tune ONE mid-depth block, leave the other 27 in FP,
# so full-model wikitext2 PPL stays meaningful.  NanoQuant's default tuning budget (8/8 epochs).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
PY=venv/bin/python
BLOCK=${BLOCK:-13}
TAG=${TAG:-q06_blk${BLOCK}}
for arm in nanoquant cov; do
  out=llm_ext/results/${TAG}_${arm}.json
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) block=$BLOCK arm=$arm ==="
  $PY llm_ext/run_llm_ext.py --arm $arm --num_calib_samples 128 --stats_cache stats_q06_admmonly.pt \
      --only_blocks "$BLOCK" --tune_nonfact --nonfact_epochs 8 --tune_fact --fact_epochs 8 \
      --out "$out" > llm_ext/logs/${TAG}_${arm}.log 2>&1
  echo "exit=$?  $(date +%T)"
  grep -a "relative block output error" llm_ext/logs/${TAG}_${arm}.log | tail -1
done
echo ALLDONE
