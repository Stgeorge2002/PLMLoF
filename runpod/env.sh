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

# ── AWS S3 sync (optional — set to auto-upload outputs after training) ──
# export AWS_ACCESS_KEY_ID="your-access-key-id"
# export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
# export AWS_DEFAULT_REGION="us-east-1"
# export S3_BUCKET="your-bucket-name"   # triggers auto-sync at end of run_all.sh

# ── Weights & Biases (optional — set your key) ──
# export WANDB_API_KEY="your-key-here"
# export WANDB_PROJECT="plmlof"
# export WANDB_ENTITY="your-entity"

# ── Create cache dirs ──
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRANSFORMERS_CACHE"
