# PLMLoF — PLM-based Loss/Gain-of-Function Variant Classifier

A PyTorch tool that classifies bacterial gene variants as **Loss-of-Function (LoF)**, **Wildtype (WT)**, or **Gain-of-Function (GoF)** using protein language model embeddings from ESM2. Designed to replace SNPEff in bacterial genomics pipelines.

## Architecture

```
Reference protein ──► ┌─────────────┐
                      │  Shared      │──► Comparison ──► ┌────────────┐
Variant protein  ──►  │  ESM2        │    Module         │ Classifier │──► LoF / WT / GoF
                      │  Encoder     │                   │ Head       │
                      │  (+ LoRA)    │         12-dim ──►│            │
                      └─────────────┘    nucleotide      └────────────┘
                                         features
```

- **ESM2 Encoder**: Shared encoder embeds both reference and variant protein sequences
- **Comparison Module**: Computes element-wise diff, product, and pooled features
- **Nucleotide Features**: 12-dim engineered features (frameshift, premature stop, start codon loss, mutation counts, truncation fraction, etc.)
- **Classifier Head**: MLP producing 3-class logits

Training uses a two-stage approach:
1. **Stage 1**: Freeze ESM2, train classifier head only
2. **Stage 2**: Enable LoRA adapters on ESM2 query/value projections for fine-tuning

## Installation

```bash
# Clone and install
git clone <repo-url> && cd PLMLoF
pip install -e ".[dev]"
```

Requirements: Python 3.10+, PyTorch 2.0+

## Quick Start

### Predict variants (no training needed)

```bash
# Using paired FASTA files
python scripts/predict.py \
  --reference ref_genes.fasta \
  --variants var_genes.fasta \
  --tiny  # uses 8M-param model, runs on CPU

# Using VCF
python scripts/predict.py \
  --reference ref_genes.fasta \
  --vcf variants.vcf \
  --model outputs/checkpoints/model_best.pt
```

### Train a model

```bash
# Quick smoke test (synthetic data, tiny model, CPU)
python scripts/train.py --tiny --max-epochs 2 --device cpu

# Full training
python scripts/train.py \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --config configs/training.yaml \
  --model-config configs/model.yaml \
  --device cuda
```

### Evaluate

```bash
python scripts/evaluate.py \
  --test-data data/processed/test.parquet \
  --model outputs/checkpoints/model_best.pt \
  --device cpu
```

### Benchmark against SNPEff

```bash
python scripts/benchmark_vs_snpeff.py \
  --snpeff-vcf annotated.vcf \
  --reference ref.fasta \
  --model outputs/checkpoints/model_best.pt
```

## Data Preparation

Download and curate training data from public databases:

```bash
# Download from CARD (GoF/AMR mutations)
python data/scripts/download_card.py --output-dir data/raw/card

# Download from DEG (essential gene disruptions → LoF)
python data/scripts/download_deg.py --output-dir data/raw/deg

# Download from ProteinGym (DMS fitness data)
python data/scripts/download_proteingym.py --output-dir data/raw/proteingym

# Generate synthetic variants
python data/scripts/generate_synthetic.py --output-dir data/raw/synthetic

# Merge and split
python data/scripts/curate_dataset.py \
  --input-dir data/raw \
  --output-dir data/processed
```

## Project Structure

```
PLMLoF/
├── configs/               # YAML configs for model, training, inference
├── plmlof/
│   ├── models/            # ESM2 encoder, comparison, classifier, full model
│   ├── data/              # Dataset, collator, features, preprocessing
│   ├── training/          # Trainer, metrics
│   ├── inference/         # Predictor, VCF handler, attribution, cache
│   └── utils/             # Sequence utilities
├── data/scripts/          # Data download & curation
├── scripts/               # CLI entry points (train, predict, evaluate)
└── tests/                 # Test suite (CPU-compatible)
```

## Testing

```bash
# Run all tests (uses tiny 8M-param ESM2, CPU only)
pytest tests/ -v

# Run specific test module
pytest tests/test_features.py -v
```

## Configuration

See `configs/` for full options:
- [configs/model.yaml](configs/model.yaml) — Model architecture, LoRA parameters
- [configs/training.yaml](configs/training.yaml) — Stage 1/2 training hyperparameters
- [configs/inference.yaml](configs/inference.yaml) — Batch size, caching, attribution

## License

MIT
