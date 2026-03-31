# RunPod environment configuration
# Source this file or set these variables in your RunPod pod template.
#
# Usage: source runpod/env.sh

# ── Persistent cache directories (survive pod restarts) ──
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

# ── CUDA settings ──
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true

# ── Weights & Biases (optional — set your key) ──
# export WANDB_API_KEY="your-key-here"
# export WANDB_PROJECT="plmlof"
# export WANDB_ENTITY="your-entity"

# ── Create cache dirs ──
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRANSFORMERS_CACHE"
