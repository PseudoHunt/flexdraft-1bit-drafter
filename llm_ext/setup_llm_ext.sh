#!/usr/bin/env bash
# Standalone setup for the NanoQuant covariance-ADMM experiment (no FlexDraft / Qwen3-8B needed).
# Usage:  bash llm_ext/setup_llm_ext.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d NanoQuant ]; then
  git clone https://github.com/SamsungLabs/NanoQuant.git
  (cd NanoQuant && git checkout a9e0a430881ff80d83b622c3129e330dc33c04f5)
fi
# apply the covariance-objective patch (idempotent)
if ! grep -q "admm_cov" NanoQuant/src/nanoquant/core/compress_block.py; then
  (cd NanoQuant && git apply ../llm_ext/nanoquant_cov.patch)
  echo "patch applied"
else
  echo "patch already applied"
fi

if [ ! -d venv ]; then
  python3 -m venv --system-site-packages venv       # reuses a system torch if present
  venv/bin/pip install -q "transformers==4.57.3" "datasets==4.4.1" "accelerate==1.11.0" \
      loguru cut_cross_entropy lm_eval
fi
venv/bin/python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"

export HF_HOME="$PWD/hf"
venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-0.6B-Base"))
PY
echo
echo "done.  Now:  export HF_HOME=\$PWD/hf PYTHONPATH=\$PWD/NanoQuant/src"
