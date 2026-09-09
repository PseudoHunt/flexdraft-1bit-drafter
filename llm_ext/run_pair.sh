#!/usr/bin/env bash
# Runs the two arms (and the FP reference) of the covariance-vs-diagonal comparison.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
PY=venv/bin/python
MODEL=${MODEL:-Qwen/Qwen3-0.6B-Base}
TAG=${TAG:-q06}
CALIB=${CALIB:-128}
EXTRA=${EXTRA:-}
CACHE=${CACHE:-stats_${TAG}.pt}
for arm in fp nanoquant cov; do
  case $arm in fp) [ "${DO_FP:-1}" = 1 ] || continue;; esac
  out=llm_ext/results/${TAG}_${arm}.json
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) arm=$arm ==="
  $PY llm_ext/run_llm_ext.py --model_id "$MODEL" --arm $arm --num_calib_samples $CALIB \
      --stats_cache "$CACHE" $EXTRA --out "$out" > llm_ext/logs/${TAG}_${arm}.log 2>&1
  echo "exit=$? $(grep -c . llm_ext/logs/${TAG}_${arm}.log) lines"
  tail -3 llm_ext/logs/${TAG}_${arm}.log | tr -d '\r'
done
