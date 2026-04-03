# Running PLMLoF on RunPod

## Quick Start

### Option A: GPU Pod (Recommended)

1. **Create a RunPod GPU pod**
   - Template: `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel`
   - GPU: A100 (80GB) recommended, A6000 (48GB) or RTX 4090 (24GB) also work
   - Disk: 50GB+ for model weights + data
   - Volume: Mount persistent storage at `/workspace`

2. **Clone and set up**
   ```bash
   cd /workspace
   git clone <your-repo-url> PLMLoF
   cd PLMLoF
   bash runpod/setup.sh
   ```

3. **Run the full pipeline**
   ```bash
   # Quick test (ESM2-8M, synthetic data, ~2 min)
   bash runpod/run_all.sh --test

   # Full production training (ESM2-650M, real data)
   bash runpod/run_all.sh
   ```

### Option B: Docker (Custom Template)

1. **Build the Docker image**
   ```bash
   docker build -t plmlof .
   ```

2. **Push to Docker Hub / RunPod registry**, then use as a custom RunPod template.

---

## File Structure for RunPod

```
runpod/
├── setup.sh              # One-time setup: install deps, download models, smoke test
├── run_all.sh            # Full pipeline: data → train → evaluate
├── download_models.sh    # Pre-download ESM2 weights to persistent cache
└── env.sh                # Environment variables for persistent cache paths

configs/
├── runpod_model.yaml     # Production model config (ESM2-650M, larger head)
└── runpod_training.yaml  # Production training config (larger batch, more epochs)
```

---

## Individual Steps

### 1. Download & prepare data
```bash
# Download and curate ProteinGym data
bash runpod/run_all.sh --data-only

# Or run individual scripts
python data/scripts/download_proteingym.py
python data/scripts/curate_dataset.py
```

### 2. Train
```bash
# Production training with auto GPU detection
python scripts/train.py \
    --config configs/runpod_training.yaml \
    --model-config configs/runpod_model.yaml \
    --train-data data/processed/train.parquet \
    --val-data data/processed/val.parquet \
    --output-dir outputs/production/

# With specific mixed precision mode
python scripts/train.py \
    --config configs/runpod_training.yaml \
    --model-config configs/runpod_model.yaml \
    --train-data data/processed/train.parquet \
    --val-data data/processed/val.parquet \
    --mixed-precision bf16 \
    --output-dir outputs/production/
```

### 3. Evaluate
```bash
python scripts/evaluate.py \
    --model outputs/production/checkpoints/model_best.pt \
    --test-data data/processed/test.parquet \
    --device cuda
```

### 4. Predict on your data
```bash
# From paired FASTA files (reference + variant genes)
python scripts/predict.py \
    --reference your_strain_reference.fasta \
    --variants your_variant_genes.fasta \
    --model outputs/production/checkpoints/model_best.pt \
    --output results.tsv

# From VCF + reference
python scripts/predict.py \
    --reference your_strain_reference.fasta \
    --vcf your_variants.vcf \
    --model outputs/production/checkpoints/model_best.pt \
    --output results.tsv
```

---

## GPU Recommendations

| GPU | VRAM | Batch Size | ESM2 Model | Notes |
|-----|------|------------|------------|-------|
| RTX 4090 | 24 GB | 8-16 | ESM2-650M | Good for fine-tuning |
| A6000 | 48 GB | 16-32 | ESM2-650M | Comfortable for training |
| A100 (40GB) | 40 GB | 16-32 | ESM2-650M | Best price/performance |
| A100 (80GB) | 80 GB | 32-64 | ESM2-650M | Fastest training |
| H100 | 80 GB | 32-64 | ESM2-650M | Best throughput, use bf16 |

For H100 pods, switch to `bf16` mixed precision:
```bash
python scripts/train.py --mixed-precision bf16 ...
```

---

## Persistent Storage

RunPod mounts `/workspace` as persistent storage. The setup script caches:
- **ESM2 model weights**: `/workspace/.cache/huggingface/` (~2.6 GB for ESM2-650M)
- **Training outputs**: `outputs/` in the project directory

These survive pod restarts so you don't re-download models each time.

---

## Multi-GPU Training

If your pod has multiple GPUs, `run_all.sh` automatically uses `accelerate` for data-parallel training:

```bash
# Automatic detection
bash runpod/run_all.sh

# Manual multi-GPU with accelerate
accelerate launch --num_processes 4 --mixed_precision fp16 \
    scripts/train.py \
    --config configs/runpod_training.yaml \
    --model-config configs/runpod_model.yaml \
    --train-data data/processed/train.parquet \
    --val-data data/processed/val.parquet
```

---

## Troubleshooting

**OOM (Out of Memory)**:
- Reduce batch size: `--batch-size 4`
- Use gradient accumulation (already configured in `runpod_training.yaml`)
- Try fp16 if using bf16, or vice versa

**Slow data loading**:
- Increase `--num-workers 8`
- Ensure data is on the local SSD, not network storage

**Model download fails**:
- Set `HF_HOME` to a writable directory
- Use `bash runpod/download_models.sh` to pre-cache models
- Check network connectivity from the pod

**CUDA errors**:
- Verify CUDA version matches PyTorch: `python -c "import torch; print(torch.version.cuda)"`
- Try `CUDA_LAUNCH_BLOCKING=1` for better error messages
