#!/usr/bin/env bash
# Reproduces the environment used for all runs. Needs: CUDA GPU (A100-80GB used), python3.10+, git.
set -euo pipefail
cd "$(dirname "$0")"
git clone https://github.com/SamsungLabs/NanoQuant.git && (cd NanoQuant && git checkout a9e0a430881ff80d83b622c3129e330dc33c04f5)
git clone https://github.com/Yaro1214/FlexDraft.git && (cd FlexDraft && git checkout 07b869b14dd4880741d40a9a07fa455ba2f67a8f)
python3 -m venv --system-site-packages venv          # reuses a system torch if present (torch 2.11+cu130 was used)
venv/bin/pip install "transformers==4.57.3" "datasets==4.4.1" "accelerate==1.11.0" gdown loguru cut_cross_entropy lm_eval
# models
mkdir -p ckpt && (cd ckpt && ../venv/bin/python -m gdown --folder "https://drive.google.com/drive/folders/1gXbBOgr8SUS9Co7wrA0CnNTg96hVDSB6")
export HF_HOME="$PWD/hf"; venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))"
echo "done. export HF_HOME=$PWD/hf PYTHONPATH=$PWD/NanoQuant/src before running work/*.py"
