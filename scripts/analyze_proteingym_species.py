"""Analyze ProteinGym bacterial species distribution by label class."""

import pandas as pd
from pathlib import Path

# Load ProteinGym reference file
ref_file = Path("data/raw/proteingym/DMS_substitutions.csv")

if not ref_file.exists():
    print(f"Reference file not found: {ref_file}")
    print("Run: python data/scripts/download_proteingym.py")
    exit(1)

# Load reference metadata
ref_df = pd.read_csv(ref_file)
print(f"Total assays in ProteinGym v1.3: {len(ref_df)}")

# Filter bacterial assays - use 'taxon' column which contains "Prokaryote"
bacterial_df = ref_df[ref_df['taxon'] == 'Prokaryote'].copy()

print(f"Prokaryote assays: {len(bacterial_df)}\n")

# Group by organism and count
print("=" * 100)
print("TOP 20 BACTERIAL SPECIES IN PROTEINGYM (by total variants)")
print("=" * 100)
print(f"\n{'Rank':<6} {'Organism':<40} {'Assays':>8} {'Total Variants':>16}")
print("-" * 100)

# Aggregate by source_organism
organism_stats = bacterial_df.groupby('source_organism').agg({
    'DMS_id': 'count',
    'DMS_total_number_mutants': 'sum'
}).reset_index()
organism_stats.columns = ['organism', 'assays', 'total_variants']
organism_stats = organism_stats.sort_values('total_variants', ascending=False)

for i, row in enumerate(organism_stats.head(20).itertuples(), 1):
    print(f"{i:<6} {row.organism:<40} {row.assays:>8} {row.total_variants:>16,}")

print("\n" + "=" * 100)
print("IMPORTANT NOTES:")
print("=" * 100)
print("1. ProteinGym uses BINARY fitness labels: 0 (not fit) vs 1 (fit)")
print("2. Your PLMLoF model uses 3-class labels: LoF / WT / GoF")
print("3. The curate_dataset.py script applies custom thresholding to create 3 classes")
print("4. These totals include single AND multiple mutants")
print("\nTo see species breakdown in YOUR dataset, run:")
print("  python -c \"import pandas as pd; df=pd.read_parquet('data/processed/train.parquet'); \\")
print("             print(df.groupby('species')['label'].value_counts().unstack(fill_value=0))\"")
