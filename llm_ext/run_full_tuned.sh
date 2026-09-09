#!/usr/bin/env bash
# Definitive comparison: full model, all 28 blocks, NanoQuant's default Step 3 (8/8/8 epochs, 128 samples).
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_tuned}
for arm in nanoquant cov; do
  out=llm_ext/results/${TAG}_${arm}.json
  [ -f "$out" ] && { echo "skip $out"; continue; }
  echo "=== $(date +%T) full tuned arm=$arm ==="
  venv/bin/python llm_ext/run_llm_ext.py --arm $arm --num_calib_samples 128 --stats_cache stats_q06_admmonly.pt \
      --tune_nonfact --nonfact_epochs 8 --tune_fact --fact_epochs 8 --tune_model --model_kd_epochs 8 \
      --zeroshot_task "" --out "$out" > llm_ext/logs/${TAG}_${arm}.log 2>&1
  echo "exit=$?  $(date +%T)"
  python3 -c "import json;d=json.load(open('$out'));print('PPL',d['eval'],'out_err',round(d['mean_out_err'],4))" 2>/dev/null
done
echo FULLDONE
