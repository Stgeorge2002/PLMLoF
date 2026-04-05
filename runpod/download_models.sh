#!/usr/bin/env bash
# Pre-download and cache ESM2 model weights to persistent storage.
# Run this to ensure models are available offline.
#
# Usage:
#   bash runpod/download_models.sh              # Download ESM2-650M (production)
#   bash runpod/download_models.sh --all        # Download all model sizes
#   bash runpod/download_models.sh --tiny       # Download ESM2-8M only (testing)

set -euo pipefail

DEFAULT_CACHE_DIR="${WORKSPACE_DIR:-$HOME}/.cache"
export HF_HOME="${HF_HOME:-$DEFAULT_CACHE_DIR/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

MODE="${1:---production}"

echo "Downloading ESM2 models to $HF_HOME ..."

python -c "
import sys
from transformers import AutoModel, AutoTokenizer

mode = '$MODE'

models = []
if mode == '--tiny':
    models = [('facebook/esm2_t6_8M_UR50D', '8M')]
elif mode == '--all':
    models = [
        ('facebook/esm2_t6_8M_UR50D', '8M'),
        ('facebook/esm2_t12_35M_UR50D', '35M'),
        ('facebook/esm2_t30_150M_UR50D', '150M'),
        ('facebook/esm2_t33_650M_UR50D', '650M'),
    ]
else:
    models = [
        ('facebook/esm2_t6_8M_UR50D', '8M'),
        ('facebook/esm2_t33_650M_UR50D', '650M'),
    ]

for name, size in models:
    print(f'  Downloading ESM2-{size}: {name}')
    try:
        AutoTokenizer.from_pretrained(name)
        AutoModel.from_pretrained(name)
        print(f'  ESM2-{size}: OK')
    except Exception as e:
        print(f'  ESM2-{size}: FAILED ({e})', file=sys.stderr)

print('Done.')
"
