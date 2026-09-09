#!/usr/bin/env bash
# Full-model comparison with the two arms running CONCURRENTLY (for a >=80GB GPU).
# Each arm peaks well under 10 GB and the tuning loop is batch-1, so they overlap well.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_tuned}
CACHE=${CACHE:-stats_q06.pt}
EPOCHS=${EPOCHS:-8}

# 1) build the shared statistics cache once (both arms must read the SAME file)
if [ ! -f "$CACHE" ]; then
  echo "=== $(date +%T) collecting calibration statistics -> $CACHE ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm nanoquant --num_calib_samples 128 \
      --stats_cache "$CACHE" --stats_only --out /dev/null > llm_ext/logs/${TAG}_stats.log 2>&1 \
      || { echo "stats collection FAILED"; tail -20 llm_ext/logs/${TAG}_stats.log; exit 1; }
fi
echo "stats cache: $(ls -la $CACHE | awk '{print $5}') bytes"

# 2) both arms at once
for arm in nanoquant cov; do
  out=llm_ext/results/${TAG}_${arm}.json
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) launching arm=$arm ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm $arm --num_calib_samples 128 --stats_cache "$CACHE" \
      --tune_nonfact --nonfact_epochs $EPOCHS --tune_fact --fact_epochs $EPOCHS \
      --tune_model --model_kd_epochs $EPOCHS \
      --out "$out" > llm_ext/logs/${TAG}_${arm}.log 2>&1 &
done
wait
echo "=== $(date +%T) both arms finished ==="
python3 llm_ext/compare.py
echo FULLDONE
