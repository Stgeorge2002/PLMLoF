#!/usr/bin/env bash
# RunPod setup script — run this ONCE when your pod starts.
# Usage: bash runpod/setup.sh
#
# This script:
#   1. Installs PLMLoF and all dependencies
#   2. Pre-downloads ESM2 model weights (cached to /workspace)
#   3. Generates synthetic training data
#   4. Runs a quick smoke test to verify GPU works
#
# RunPod pods typically mount /workspace as persistent storage.
# Model weights are cached there so they survive pod restarts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=============================================="
echo " PLMLoF RunPod Setup"
echo "=============================================="
echo "Project dir: $PROJECT_DIR"
echo "Python:      $(python --version 2>&1)"
echo "CUDA:        $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not available')"
echo ""

# ── 1. Set cache directories to persistent storage ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRANSFORMERS_CACHE"
echo "Cache dirs:"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo ""

# ── 2. Install dependencies ──
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet -e ".[tracking,dev]"
echo "Dependencies installed."
echo ""

# ── 3. Pre-download ESM2 models ──
echo "Pre-downloading ESM2 models (cached to $HF_HOME)..."
python -c "
from transformers import AutoModel, AutoTokenizer
import os
print('  Downloading ESM2-8M (tiny, for testing)...')
AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
AutoModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
print('  Downloading ESM2-650M (production)...')
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
print('  Models cached successfully.')
"
echo ""

echo ""

# ── 5. GPU smoke test ──
echo "Running GPU smoke test..."
python -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name} ({mem:.1f} GB)')
    print(f'  CUDA devices: {torch.cuda.device_count()}')
    # Quick tensor test
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    print(f'  Tensor test: OK (matmul on GPU)')
else:
    print('  WARNING: No GPU detected! Training will be slow on CPU.')
"
echo ""

# ── 6. Quick model forward pass test ──
echo "Running model forward pass test..."
python -c "
import torch
from plmlof.models.plmlof_model import PLMLoFModel
from plmlof.data.dataset import SyntheticPLMLoFDataset
from plmlof.data.collator import PLMLoFCollator
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  Device: {device}')

model = PLMLoFModel(
    esm2_model_name='facebook/esm2_t6_8M_UR50D',
    freeze_esm2=True,
).to(device)

dataset = SyntheticPLMLoFDataset(num_samples=4)
collator = PLMLoFCollator(tokenizer_name='facebook/esm2_t6_8M_UR50D')
loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

batch = next(iter(loader))
batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.no_grad():
    logits = model(
        ref_input_ids=batch['ref_input_ids'],
        ref_attention_mask=batch['ref_attention_mask'],
        var_input_ids=batch['var_input_ids'],
        var_attention_mask=batch['var_attention_mask'],
        nucleotide_features=batch['nucleotide_features'],
    )
print(f'  Output shape: {logits.shape} (expected [2, 3])')
print(f'  Predictions: {logits.argmax(dim=-1).tolist()}')
print('  Forward pass: OK')
"
echo ""

echo "=============================================="
echo " Setup complete! Ready to train."
echo ""
echo " Quick start:"
echo "   bash runpod/run_all.sh          # Full pipeline"
echo "   bash runpod/run_all.sh --test   # Quick test run"
echo ""
echo " Or run individual steps:"
echo "   python scripts/train.py --device cuda --config configs/training.yaml"
echo "   python scripts/predict.py --reference ref.fasta --variants var.fasta --device cuda"
echo "=============================================="
