#!/bin/bash
# Complete pipeline to re-curate PLMLoF dataset with ALL bacterial data
# This includes:
#   - ALL E. coli variants (40K+ instead of 18K)
#   - ALL priority species (100%)
#   - SINGLE AND MULTIPLE mutants (epistatic effects)
#   - 900K total samples (3x previous size)

set -e  # Exit on error

echo "============================================================"
echo "PLMLoF DATA RE-CURATION: FULL BACTERIAL DATASET"
echo "============================================================"
echo ""
echo "Changes from previous pipeline:"
echo "  - E. coli sampling: 45% → 100% (+22K variants)"
echo "  - Total samples: 300K → 900K (3x increase)"
echo "  - Multiple mutants: NOW INCLUDED"
echo "  - All bacterial species: 100% retention"
echo ""
echo "Expected outcomes:"
echo "  - Better per-species performance (E. coli, Klebsiella, etc.)"
echo "  - +10-20% improvement on priority species test sets"
echo "  - ~6 hours embedding precomputation on RTX 6000 Ada"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Step 1: Download ProteinGym data (if not already done)
echo "============================================================"
echo "STEP 1: Download ProteinGym bacterial data"
echo "============================================================"
if [ ! -f "data/raw/proteingym/DMS_substitutions.csv" ]; then
    echo "Downloading ProteinGym reference and DMS scores..."
    python data/scripts/download_proteingym.py
    echo "✓ Download complete"
else
    echo "✓ ProteinGym data already downloaded"
fi
echo ""

# Step 2: Curate dataset with new parameters
echo "============================================================"
echo "STEP 2: Curate balanced dataset (900K samples)"
echo "============================================================"
echo "This will create:"
echo "  - data/processed/train.parquet (720K samples)"
echo "  - data/processed/val.parquet (90K samples)"
echo "  - data/processed/test.parquet (90K samples)"
echo "  - data/processed/test_ecoli.parquet (per-species test)"
echo "  - data/processed/test_klepn.parquet"
echo "  - data/processed/test_myctu.parquet"
echo "  - data/processed/test_stau.parquet"
echo "  - data/processed/test_strsp.parquet (Streptococcus sp.)"
echo ""

# Backup old data if exists
if [ -f "data/processed/train.parquet" ]; then
    BACKUP_DIR="data/processed/backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old data to $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    mv data/processed/*.parquet "$BACKUP_DIR/" 2>/dev/null || true
    echo "✓ Backup complete"
fi

echo "Running curation (this may take 5-10 minutes)..."
python data/scripts/curate_dataset.py --total-samples 900000

echo ""
echo "✓ Dataset curation complete!"
echo ""

# Step 3: Verify new dataset
echo "============================================================"
echo "STEP 3: Verify new dataset composition"
echo "============================================================"
python -c "
import pandas as pd

train = pd.read_parquet('data/processed/train.parquet')
val = pd.read_parquet('data/processed/val.parquet')
test = pd.read_parquet('data/processed/test.parquet')

print(f'Train: {len(train):,} samples')
print(f'Val:   {len(val):,} samples')
print(f'Test:  {len(test):,} samples')
print(f'Total: {len(train)+len(val)+len(test):,} samples')
print()

# Label distribution
print('TRAIN LABEL DISTRIBUTION:')
print(train['label'].value_counts().sort_index())
print()

# Species distribution
print('TRAIN SPECIES DISTRIBUTION (top 10):')
species_counts = train['species'].value_counts().head(10)
for species, count in species_counts.items():
    pct = count / len(train) * 100
    print(f'  {species:<40} {count:>8,} ({pct:>5.1f}%)')
print()

# Count mutations per variant
from plmlof.utils.sequence_utils import find_mutations
train['num_mutations'] = train.apply(
    lambda r: len(find_mutations(r['ref_protein'], r['var_protein'])), 
    axis=1
)
print('MUTATION COUNT DISTRIBUTION:')
mut_dist = train['num_mutations'].value_counts().sort_index()
for n_mut, count in mut_dist.head(10).items():
    pct = count / len(train) * 100
    print(f'  {n_mut} mutation(s): {count:>8,} ({pct:>5.1f}%)')
print()

# Per-species test sets
import glob
species_tests = glob.glob('data/processed/test_*.parquet')
print('PER-SPECIES TEST SETS:')
for test_file in sorted(species_tests):
    df = pd.read_parquet(test_file)
    species_name = test_file.split('_')[-1].replace('.parquet', '')
    print(f'  test_{species_name}: {len(df):,} samples')
"
echo ""

# Step 4: Precompute embeddings
echo "============================================================"
echo "STEP 4: Precompute ESM2 embeddings"
echo "============================================================"
echo "WARNING: This will take ~6 hours on NVIDIA RTX 6000 Ada"
echo "         Requires ~51GB VRAM and 20GB disk space"
echo ""
echo "Files to be created:"
echo "  - data/embeddings/train_embeddings.pt (~14GB)"
echo "  - data/embeddings/val_embeddings.pt (~1.8GB)"
echo "  - data/embeddings/test_embeddings.pt (~1.8GB)"
echo "  - data/embeddings/test_ecoli_embeddings.pt"
echo "  - data/embeddings/test_klepn_embeddings.pt"
echo "  - data/embeddings/test_myctu_embeddings.pt"
echo "  - data/embeddings/test_stau_embeddings.pt"
echo "  - data/embeddings/test_strsp_embeddings.pt"
echo ""
read -p "Run embedding precomputation now? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Backup old embeddings
    if [ -d "data/embeddings" ]; then
        EMBED_BACKUP="data/embeddings_backup_$(date +%Y%m%d_%H%M%S)"
        echo "Backing up old embeddings to $EMBED_BACKUP"
        mv data/embeddings "$EMBED_BACKUP"
    fi
    
    echo "Starting embedding precomputation..."
    echo "Monitor with: watch -n 1 'ls -lh data/embeddings/*.pt'"
    python scripts/precompute_embeddings.py \\
        --train-data data/processed/train.parquet \\
        --val-data data/processed/val.parquet \\
        --test-data data/processed/test.parquet \\
        --output-dir data/embeddings \\
        --batch-size 8 \\
        --device cuda
    
    echo ""
    echo "✓ Embedding precomputation complete!"
else
    echo "Skipping embedding precomputation."
    echo "Run manually with:"
    echo "  python scripts/precompute_embeddings.py \\"
    echo "    --train-data data/processed/train.parquet \\"
    echo "    --val-data data/processed/val.parquet \\"
    echo "    --test-data data/processed/test.parquet \\"
    echo "    --output-dir data/embeddings \\"
    echo "    --batch-size 8 \\"
    echo "    --device cuda"
fi
echo ""

# Step 5: Summary
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Train new model with expanded dataset:"
echo "     python scripts/train.py \\"
echo "       --config configs/runpod_training.yaml \\"
echo "       --device cuda"
echo ""
echo "  2. Expected improvements:"
echo "     - E. coli test accuracy: 65% → 75-80%"
echo "     - Klebsiella test accuracy: 58% → 70-75%"
echo "     - M. tuberculosis: 65% → 75-80%"
echo "     - S. aureus: 72% → 80-85%"
echo ""
echo "  3. Training time: ~2-3 hours on RTX 6000 Ada"
echo "     (3x more data but better batching efficiency)"
echo ""
echo "============================================================"
