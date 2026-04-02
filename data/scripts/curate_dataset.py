"""Curate and merge all data sources into a unified training dataset.

Merges:
1. CARD GoF variants (download_card.py)
2. DEG LoF variants (download_deg.py)
3. ProteinGym DMS bacterial variants (download_proteingym.py)
4. Synthetic variants (generate_synthetic.py)

Produces stratified train/val/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed/")
OUTPUT_DIR = Path("data/processed/")

REQUIRED_COLUMNS = [
    "ref_protein", "var_protein", "label", "gene", "species", "source",
]


def load_source(path: Path, source_name: str) -> pd.DataFrame:
    """Load a processed data source, handling missing files gracefully."""
    if not path.exists():
        logger.warning(f"{source_name} not found at {path}. Skipping.")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} records from {source_name}")
    return df


def merge_datasets() -> pd.DataFrame:
    """Load and merge all data sources into a unified dataset."""
    sources = {
        "CARD GoF": PROCESSED_DIR / "card_gof.parquet",
        "DEG LoF": PROCESSED_DIR / "deg_lof.parquet",
        "ProteinGym": PROCESSED_DIR / "proteingym_bacterial.parquet",
        "Synthetic": PROCESSED_DIR / "synthetic_variants.parquet",
    }

    dfs = []
    for name, path in sources.items():
        df = load_source(path, name)
        if not df.empty:
            # Ensure required columns exist
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            # Ensure label is int
            if "label" in df.columns:
                df["label"] = df["label"].astype(int)
            dfs.append(df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]])

    if not dfs:
        logger.error("No data sources available. Run download scripts first.")
        raise RuntimeError("No data sources found")

    merged = pd.concat(dfs, ignore_index=True)

    # Strip stop codon '*' characters — ESM2 cannot tokenize them
    merged["ref_protein"] = merged["ref_protein"].str.replace("*", "", regex=False)
    merged["var_protein"] = merged["var_protein"].str.replace("*", "", regex=False)

    # Drop rows with empty proteins
    merged = merged[merged["ref_protein"].str.len() > 0]
    merged = merged[merged["var_protein"].str.len() > 0]

    # Drop exact duplicates
    merged = merged.drop_duplicates(subset=["ref_protein", "var_protein", "label"])

    logger.info(f"Merged dataset: {len(merged)} records")
    label_counts = merged["label"].value_counts()
    logger.info(f"Label distribution: LoF={label_counts.get(0, 0)}, WT={label_counts.get(1, 0)}, GoF={label_counts.get(2, 0)}")

    return merged


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
    holdout_species: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test with stratification.

    Optionally holds out entire species for cross-species generalization testing.

    Args:
        df: Full dataset.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        seed: Random seed.
        holdout_species: Species to exclude from train/val and put entirely in test.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Holdout species
    if holdout_species:
        holdout_mask = df["species"].str.lower().apply(
            lambda s: any(h.lower() in s for h in holdout_species) if isinstance(s, str) else False
        )
        holdout_df = df[holdout_mask]
        remaining_df = df[~holdout_mask]
        logger.info(f"Held out {len(holdout_df)} records from species: {holdout_species}")
    else:
        holdout_df = pd.DataFrame()
        remaining_df = df

    if remaining_df.empty:
        logger.warning("No data remaining after holdout. Using all data.")
        remaining_df = df
        holdout_df = pd.DataFrame()

    # Stratified split on label
    stratify_col = remaining_df["label"]

    # Check if stratification is possible (need ≥2 samples per class per split)
    min_class_count = stratify_col.value_counts().min()
    can_stratify = min_class_count >= 3  # Need at least 3 to split into train/val/test

    # First split: separate test set
    try:
        train_val_df, test_df = train_test_split(
            remaining_df,
            test_size=test_size,
            stratify=stratify_col if can_stratify else None,
            random_state=seed,
        )
    except ValueError:
        # Fallback: no stratification if classes are too small
        train_val_df, test_df = train_test_split(
            remaining_df,
            test_size=test_size,
            random_state=seed,
        )

    # Second split: separate validation from training
    val_fraction = val_size / (1 - test_size)
    try:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction,
            stratify=train_val_df["label"] if can_stratify else None,
            random_state=seed,
        )
    except ValueError:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction,
            random_state=seed,
        )

    # Add holdout species to test set
    if not holdout_df.empty:
        test_df = pd.concat([test_df, holdout_df], ignore_index=True)

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def main():
    logging.basicConfig(level=logging.INFO)

    merged = merge_datasets()

    train_df, val_df, test_df = stratified_split(
        merged,
        holdout_species=["Pseudomonas aeruginosa", "Salmonella"],
    )

    # Save splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = OUTPUT_DIR / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info(f"Saved {split_name}: {len(split_df)} records → {path}")

    # Save full merged dataset
    merged_path = OUTPUT_DIR / "merged_all.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.info(f"Saved merged dataset: {len(merged)} records → {merged_path}")


if __name__ == "__main__":
    main()
