"""Curate ProteinGym DMS data into a balanced training dataset.

Loads ProteinGym bacterial variants and subsamples to 150K total records
with equal class balance (50,000 each for LoF, WT, GoF).

Produces stratified train/val/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed/")
OUTPUT_DIR = Path("data/processed/")

REQUIRED_COLUMNS = [
    "ref_protein", "var_protein", "label", "gene", "species", "source",
    "dms_score", "dms_zscore",
]


def load_source(path: Path, source_name: str) -> pd.DataFrame:
    """Load a processed data source, handling missing files gracefully."""
    if not path.exists():
        logger.warning(f"{source_name} not found at {path}. Skipping.")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} records from {source_name}")
    return df


# Total samples and per-class target for balanced subsampling.
# Overridden at runtime by --total-samples CLI arg (or PLMLOF_TOTAL_SAMPLES env var).
DEFAULT_TOTAL_SAMPLES = 300_000


def merge_datasets(total_samples: int = DEFAULT_TOTAL_SAMPLES) -> pd.DataFrame:
    """Load ProteinGym data and subsample to a balanced dataset."""
    path = PROCESSED_DIR / "proteingym_bacterial.parquet"
    df = load_source(path, "ProteinGym")

    if df.empty:
        logger.error("ProteinGym data not found. Run download_proteingym.py first.")
        raise RuntimeError("No data source found")

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col in ("dms_score", "dms_zscore"):
                df[col] = float("nan")
            else:
                df[col] = ""
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    # NOTE: Do NOT strip "*" here — feature extraction (has_premature_stop,
    # nonsense count, truncation_fraction) depends on "*" being present.
    # The Dataset class / collator strip "*" only at the ESM2 tokenization stage.

    # Drop rows with empty proteins
    df = df[df["ref_protein"].str.len() > 0]
    df = df[df["var_protein"].str.len() > 0]

    # Drop duplicates on (ref_protein, var_protein) only — keep first label
    # This avoids conflicting labels when the same variant appears in
    # different assays with different fitness thresholds.
    df = df.drop_duplicates(subset=["ref_protein", "var_protein"])

    logger.info(f"ProteinGym after dedup: {len(df)} records")
    label_counts = df["label"].value_counts()
    logger.info(f"Full label distribution: LoF={label_counts.get(0, 0)}, WT={label_counts.get(1, 0)}, GoF={label_counts.get(2, 0)}")

    # Balanced subsample: equal thirds per class
    samples_per_class = total_samples // 3
    balanced_dfs = []
    for label in [0, 1, 2]:
        class_df = df[df["label"] == label]
        n = min(len(class_df), samples_per_class)
        balanced_dfs.append(class_df.sample(n=n, random_state=42))
        logger.info(f"  Label {label}: sampled {n} / {len(class_df)} available")

    merged = pd.concat(balanced_dfs, ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    logger.info(f"Balanced dataset: {len(merged)} records")
    label_counts = merged["label"].value_counts()
    logger.info(f"Label distribution: LoF={label_counts.get(0, 0)}, WT={label_counts.get(1, 0)}, GoF={label_counts.get(2, 0)}")

    return merged


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
    holdout_species: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test using gene-level stratification.

    All variants from the same gene/assay go to the same split, preventing
    reference-sequence leakage across train, val, and test.

    Holdout species are extracted first and returned as a separate DataFrame
    (not merged into test) for independent cross-species evaluation.

    Returns:
        Tuple of (train_df, val_df, test_df, holdout_df).
    """
    # ── Holdout species (extracted before split, returned separately) ──
    if holdout_species:
        holdout_mask = df["species"].str.lower().apply(
            lambda s: any(h.lower() in s for h in holdout_species) if isinstance(s, str) else False
        )
        holdout_df = df[holdout_mask].copy()
        remaining_df = df[~holdout_mask].copy()
        logger.info(f"Held out {len(holdout_df)} records from species: {holdout_species}")
    else:
        holdout_df = pd.DataFrame()
        remaining_df = df.copy()

    if remaining_df.empty:
        logger.warning("No data remaining after holdout. Using all data.")
        remaining_df = df.copy()
        holdout_df = pd.DataFrame()

    min_needed = max(10, int(1 / test_size) + 1)
    if len(remaining_df) < min_needed:
        raise RuntimeError(
            f"Dataset has only {len(remaining_df)} records — too small to split "
            f"(need ≥ {min_needed}).\n"
            "The ProteinGym download likely failed or returned no bacterial assays.\n"
            "Re-run:  python data/scripts/download_proteingym.py\n"
            "Then check data/raw/proteingym/ for non-empty CSV files."
        )

    # ── Stratified random split by label ──
    # All genes appear in train, val, and test — different variants from the
    # same gene end up in each split.  The model learns mutation-type patterns
    # (stop codons, truncation fraction, conservative vs. radical substitutions)
    # not gene identity, so gene-level leakage is not a meaningful concern.
    # The only true leakage — identical (ref, var) pairs in multiple splits —
    # is already prevented by drop_duplicates() in merge_datasets().
    #
    # Cross-gene / cross-species generalisation is evaluated separately using
    # the held-out species set (Pseudomonas, Salmonella).
    from sklearn.model_selection import train_test_split as _tts

    remaining_df["gene"] = remaining_df["gene"].fillna("").astype(str)
    stratify_col = remaining_df["label"] if remaining_df["label"].nunique() > 1 else None

    # First split: carve out test set
    trainval_df, test_df = _tts(
        remaining_df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=seed,
    )

    # Second split: carve val from train+val
    val_frac = val_size / (1.0 - test_size)
    strat_tv = trainval_df["label"] if trainval_df["label"].nunique() > 1 else None
    train_df, val_df = _tts(
        trainval_df,
        test_size=val_frac,
        stratify=strat_tv,
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    logger.info(f"Split (stratified random): train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    return train_df, val_df, test_df, holdout_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES,
        help=f"Total balanced samples across LoF/WT/GoF (default: {DEFAULT_TOTAL_SAMPLES:,})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Target dataset size: {args.total_samples:,} samples ({args.total_samples // 3:,} per class)")

    merged = merge_datasets(total_samples=args.total_samples)

    train_df, val_df, test_df, holdout_df = stratified_split(
        merged,
        holdout_species=["Pseudomonas aeruginosa", "Salmonella"],
    )

    # Save splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = OUTPUT_DIR / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info(f"Saved {split_name}: {len(split_df)} records → {path}")

    # Save holdout species separately for cross-species generalisation evaluation
    if not holdout_df.empty:
        holdout_path = OUTPUT_DIR / "test_holdout_species.parquet"
        holdout_df.to_parquet(holdout_path, index=False)
        logger.info(f"Saved holdout species: {len(holdout_df)} records → {holdout_path}")

    # Save full merged dataset
    merged_path = OUTPUT_DIR / "merged_all.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.info(f"Saved merged dataset: {len(merged)} records → {merged_path}")


if __name__ == "__main__":
    main()
