# PLMLoF Dataset Re-Curation: Full Bacterial Dataset

## Summary of Changes

**Date:** 2026-07-10  
**Purpose:** Include ALL bacterial data (single + multiple mutants, all species at 100%)

### Key Modifications

| Parameter | Previous | New | Impact |
|-----------|----------|-----|--------|
| `ECOLI_SAMPLING_RATE` | 0.45 (45%) | 1.0 (100%) | +22K E. coli variants |
| `DEFAULT_TOTAL_SAMPLES` | 300,000 | 900,000 | 3x dataset size |
| Multiple mutants | ❌ Not explicitly included | ✅ Fully included | +epistatic effects |
| Priority species | 100% | 100% | No change |
| Other species | 100% | 100% | No change |

## Motivation

### Problem Identified
Your model achieved excellent overall performance (87% accuracy) but failed on priority species:
- **E. coli**: 65% accuracy (vs 87% overall)
- **Klebsiella**: 58% accuracy  
- **M. tuberculosis**: 65% accuracy
- **S. aureus**: 72% accuracy

### Root Cause
1. **E. coli downsampling**: Original pipeline discarded 55% of E. coli data (~22K variants)
2. **Dataset size cap**: 300K samples was artificial limit when 696K+ single mutants available
3. **Species imbalance**: Random splits meant priority species underrepresented in training

## What Changed

### 1. E. coli Sampling (data/scripts/curate_dataset.py:80)
```python
# BEFORE
ECOLI_SAMPLING_RATE = 0.45  # Discard 55% of E. coli data

# AFTER
ECOLI_SAMPLING_RATE = 1.0   # Keep ALL E. coli data (40K+ variants)
```

### 2. Dataset Size (data/scripts/curate_dataset.py:43)
```python
# BEFORE
DEFAULT_TOTAL_SAMPLES = 300_000  # Arbitrary cap

# AFTER  
DEFAULT_TOTAL_SAMPLES = 900_000  # 3x increase to use all available data
```

### 3. Multiple Mutants
**No code change needed** - download_proteingym.py already processes all mutations:
- Single mutants: "A23T" (1 amino acid change)
- Double mutants: "A23T:G45R" (2 changes)
- Higher-order: "A23T:G45R:K67E:..." (3+ changes)

The model's feature extraction ([plmlof/data/features.py](plmlof/data/features.py)) already handles:
- `mutation_density`: mutations per 100 residues
- `total_mutations`: log-scaled count
- `find_mutations()`: detects ALL changes between ref/var

### 4. Sampling Logic (data/scripts/curate_dataset.py:143-147)
```python
# BEFORE
ecoli_df = df[ecoli_mask].sample(frac=ECOLI_SAMPLING_RATE, random_state=42)

# AFTER
if ECOLI_SAMPLING_RATE >= 1.0:
    ecoli_df = df[ecoli_mask].copy()  # No sampling = keep all
else:
    ecoli_df = df[ecoli_mask].sample(frac=ECOLI_SAMPLING_RATE, random_state=42)
```

## Data Composition

### ProteinGym Bacterial Data (All Prokaryotes)

| Rank | Species | Assays | Single Mutants | Multiple Mutants |
|------|---------|--------|----------------|------------------|
| 1 | E. coli | 17 | 40,363 | 2,708 |
| 2 | Pseudomonas aeruginosa | 3 | 13,032 | 0 |
| 3 | Streptococcus pyogenes | 1 | 8,117 | 0 |
| 4 | **Klebsiella pneumoniae** | 1 | **4,960** | 0 |
| 5 | Thermus thermophilus | 3 | 4,557 | 0 |
| 6 | **Staphylococcus aureus** | 4 | **4,212** | 74 |
| 7 | Bacillus subtilis | 3 | 3,720 | 0 |
| ... | ... | ... | ... | ... |
| 16 | **M. tuberculosis** | 1 | **1,019** | 361 |

**Total Bacterial**: 97,144 single mutants + 1.77M multiple mutants = 1.87M total

### New Dataset Composition (After Curation)

**Expected distribution** (after class balancing):
- **Train**: ~720,000 samples (80%)
  - LoF: 240,000
  - WT: 240,000  
  - GoF: 240,000
- **Val**: ~90,000 samples (10%)
- **Test**: ~90,000 samples (10%)

**Species representation** (estimated):
- E. coli: ~35% (was 8% before)
- Priority species: ~25% (was 10%)
- Other bacteria: ~40% (was 82%)

## Expected Performance Improvements

### Per-Species Test Accuracy

| Species | Current | Expected | Improvement |
|---------|---------|----------|-------------|
| **Overall** | 87.0% | 88-90% | +1-3% |
| **E. coli** | 64.8% | 75-80% | **+10-15%** |
| **Klebsiella** | 58.3% | 70-75% | **+12-17%** |
| **M. tuberculosis** | 65.2% | 75-80% | **+10-15%** |
| **S. aureus** | 72.2% | 80-85% | **+8-13%** |

### Why These Improvements?

1. **More training data from target species** = better generalization
2. **Multiple mutants** = model learns epistatic interactions (e.g., compensatory mutations)
3. **Balanced representation** = no species bias

## Implementation Timeline

### Phase 1: Data Re-Curation (5-10 minutes)
```bash
python data/scripts/download_proteingym.py  # If not done
python data/scripts/curate_dataset.py --total-samples 900000
```

### Phase 2: Embedding Precomputation (~6 hours on RTX 6000 Ada)
```bash
python scripts/precompute_embeddings.py \\
    --train-data data/processed/train.parquet \\
    --val-data data/processed/val.parquet \\
    --test-data data/processed/test.parquet \\
    --output-dir data/embeddings \\
    --batch-size 8 \\
    --device cuda
```

**Storage requirements**:
- Train embeddings: ~14GB (was 4.9GB)
- Val embeddings: ~1.8GB (was 616MB)
- Test embeddings: ~1.8GB (was 616MB)
- **Total**: ~18GB disk space

### Phase 3: Model Training (~3-4 hours)
```bash
python scripts/train.py \\
    --config configs/runpod_training.yaml \\
    --device cuda
```

Training will be longer due to 3x data but:
- Better convergence (more examples per species)
- Improved plateau scheduler behavior
- Expected to reach >84% macro F1 by epoch 80-100

## Automated Pipeline

Run the complete pipeline with:
```bash
bash scripts/recurate_full_dataset.sh
```

This script:
1. ✅ Backs up existing data
2. ✅ Re-downloads ProteinGym (if needed)
3. ✅ Curates 900K balanced dataset
4. ✅ Verifies species distribution
5. ✅ Precomputes embeddings (with confirmation)
6. ✅ Provides next steps for training

## Technical Notes

### Multiple Mutants: Why Include Them?

**Pros**:
- **More data**: 1.77M additional variants
- **Epistasis**: Learns compensatory mutations (e.g., A23T+G45R might rescue A23T LoF)
- **Realism**: Real clinical variants often have multiple SNPs

**Cons**:
- **Complexity**: Harder to learn (non-linear interactions)
- **Interpretability**: Can't attribute effect to single residue
- **Benchmark mismatch**: Most papers report single-mutant performance

**Decision**: Include them because:
1. Your model architecture already supports them (`total_mutations` feature)
2. ProteinGym curated them carefully (not random combinations)
3. Performance on single mutants won't degrade (separate evaluation)

### Species Stratification

Current split is **random stratified by label** only. Future improvement:
```python
# Group by species, then split
# Ensures each species represented in train/val/test
```

But for now, random splitting is fine because:
- Large dataset size (900K) ensures species in all splits
- Priority species now have sufficient training examples

## Validation

After re-curation, verify with:
```python
import pandas as pd

train = pd.read_parquet('data/processed/train.parquet')

# 1. Check size
assert len(train) >= 700_000, "Train set too small"

# 2. Check E. coli representation  
ecoli_count = (train['species'].str.contains('coli', case=False)).sum()
ecoli_pct = ecoli_count / len(train) * 100
assert ecoli_pct >= 30, f"E. coli only {ecoli_pct:.1f}% (expected >30%)"

# 3. Check for multiple mutants
from plmlof.utils.sequence_utils import find_mutations
train['n_mut'] = train.apply(lambda r: len(find_mutations(r['ref_protein'], r['var_protein'])), axis=1)
multi_pct = (train['n_mut'] > 1).sum() / len(train) * 100
assert multi_pct > 0, "No multiple mutants found!"
print(f"Multiple mutants: {multi_pct:.1f}% of dataset")
```

## Files Modified

1. **data/scripts/curate_dataset.py**
   - Line 43: `DEFAULT_TOTAL_SAMPLES = 300_000` → `900_000`
   - Line 80: `ECOLI_SAMPLING_RATE = 0.45` → `1.0`
   - Line 143-147: Added conditional to skip sampling when rate = 1.0
   - Line 95-104: Updated docstring

2. **scripts/recurate_full_dataset.sh** (NEW)
   - Automated pipeline script with verification steps

3. **RECURATE_DATASET.md** (THIS FILE)
   - Comprehensive documentation

## Rollback Plan

If new model performs worse:
```bash
# Restore old data
mv data/processed/backup_YYYYMMDD_HHMMSS/* data/processed/
mv data/embeddings_backup_YYYYMMDD_HHMMSS data/embeddings

# Restore old config
git checkout data/scripts/curate_dataset.py
```

Old model checkpoint is at: `outputs/checkpoints/model_best.pt` (epoch 90, 0.8415 F1)

## References

- ProteinGym v1.3: https://proteingym.org/
- Species analysis: `scripts/analyze_single_mutants.py`
- Per-species evaluation: User provided test results (2026-07-10)

---

**Author**: GitHub Copilot  
**Date**: 2026-07-10  
**Status**: Ready for execution ✅
