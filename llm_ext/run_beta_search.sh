#!/usr/bin/env bash
# Per-block beta search: for each block, run Step 3 once per beta in BETAS (corr-shrink; 1 = NanoQuant diagonal),
# keep the candidate with the lowest wikitext2 VALIDATION PPL, report test PPL.  Blocks BLOCKS only, no KD.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src
TAG=${TAG:-q06_b3}
CACHE=${CACHE:-stats_q06.pt}
BLOCKS=${BLOCKS:-0,1,2,3,4,5,6,7,8,9}
BETAS=${BETAS:-0,0.25,0.5,0.75,1}
EPOCHS=${EPOCHS:-8}
out=llm_ext/results/${TAG}_betasearch.json
[ -f "$out" ] && { echo "skip $out"; exit 0; }
echo "=== $(date +%T) beta search over {$BETAS} on blocks $BLOCKS ==="
venv/bin/python llm_ext/run_llm_ext.py --arm cov --cov_beta_search "$BETAS" --num_calib_samples 128 \
    --stats_cache "$CACHE" --tune_nonfact --nonfact_epochs $EPOCHS --tune_fact --fact_epochs $EPOCHS \
    --ppl_after_block --only_blocks "$BLOCKS" --out "$out" > llm_ext/logs/${TAG}_betasearch.log 2>&1
echo "=== $(date +%T) done (exit $?) ==="; echo BETADONE
