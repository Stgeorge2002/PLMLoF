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

# Species used for per-species evaluation.  All of their variants are prioritised
# during subsampling so the model has been trained on as many of their sequences as
# possible.  The file-system tag (used for parquet filenames) maps to a substring
# that is matched against the `species` column.
EVALUATION_SPECIES: list[tuple[str, str]] = [
    ("ecoli",   "Escherichia coli"),
    ("myctu",   "Mycobacterium tuberculosis"),
    ("stau",    "Staphylococcus aureus"),
    ("klepn",   "Klebsiella pneumoniae"),
    ("strpn",   "Streptococcus pneumoniae"),
]

# Species whose variants are included at 100 % in the subsampled training pool.
# E. coli is NOT in this list — it provides the bulk signal but is rate-limited
# so that minority species are not crowded out.
PRIORITY_SPECIES_SUBSTRINGS = [
    "Mycobacterium",
    "Staphylococcus",
    "Klebsiella",
    "Streptococcus",
    "Neisseria",
    "Bacillus",
    "Pseudomonas",
    "Salmonella",
    "Vibrio",
    "Acinetobacter",
    "Helicobacter",
    "Campylobacter",
    "Clostridium",
    "Listeria",
    "Legionella",
    "Corynebacterium",
]

# Fraction of E. coli variants to include (before class-balancing)
# — keeps E. coli as majority while letting other species breathe.
ECOLI_SAMPLING_RATE = 0.45


def _is_ecoli(species: str) -> bool:
    return "escherichia" in species.lower() or "ecolx" in species.lower()


def _is_priority(species: str) -> bool:
    return any(sub.lower() in species.lower() for sub in PRIORITY_SPECIES_SUBSTRINGS)


def merge_datasets(total_samples: int = DEFAULT_TOTAL_SAMPLES) -> pd.DataFrame:
    """Load ProteinGym data and subsample using species-aware sampling.

    E. coli is the largest assay by far but is capped at ECOLI_SAMPLING_RATE
    so that all other bacterial species contribute proportionally more.
    Priority test species (M. tuberculosis, S. aureus, Klebsiella, etc.) are
    included at 100 %.
    A final class-balance step ensures equal LoF / WT / GoF counts.
    """
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

    # Drop rows with empty proteins
    df = df[df["ref_protein"].str.len() > 0]
    df = df[df["var_protein"].str.len() > 0]

    # Drop duplicates — keep first label
    df = df.drop_duplicates(subset=["ref_protein", "var_protein"])

    logger.info(f"ProteinGym after dedup: {len(df):,} records")
    label_counts = df["label"].value_counts()
    logger.info(f"Full label distribution: LoF={label_counts.get(0, 0)}, WT={label_counts.get(1, 0)}, GoF={label_counts.get(2, 0)}")

    # ── Species-aware sampling ────────────────────────────────────────────────
    # Log available data per species before sampling
    species_col = df["species"].fillna("").astype(str)
    logger.info("Species breakdown (pre-sampling):")
    for sp, cnt in species_col.value_counts().items():
        logger.info(f"  {sp or '(unknown)'}: {cnt:,}")

    ecoli_mask    = species_col.apply(_is_ecoli)
    priority_mask = species_col.apply(_is_priority)
    other_mask    = ~ecoli_mask & ~priority_mask

    # Priority species: include ALL available variants
    priority_df = df[priority_mask].copy()

    # E. coli: cap at ECOLI_SAMPLING_RATE of its available variants
    ecoli_df = df[ecoli_mask].sample(frac=ECOLI_SAMPLING_RATE, random_state=42)

    # Other (unknown species): include all
    other_df = df[other_mask].copy()

    combined = pd.concat([ecoli_df, priority_df, other_df], ignore_index=True)
    logger.info(
        f"Pre-balance pool: {len(ecoli_df):,} E. coli + "
        f"{len(priority_df):,} priority species + "
        f"{len(other_df):,} other = {len(combined):,} total"
    )

    # ── Class balance ─────────────────────────────────────────────────────────
    samples_per_class = total_samples // 3
    balanced_dfs = []
    for label in [0, 1, 2]:
        class_df = combined[combined["label"] == label]
        n = min(len(class_df), samples_per_class)
        balanced_dfs.append(class_df.sample(n=n, random_state=42))
        logger.info(f"  Label {label}: sampled {n:,} / {len(class_df):,} available")

    merged = pd.concat(balanced_dfs, ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Balanced dataset: {len(merged):,} records")
    label_counts = merged["label"].value_counts()
    logger.info(f"Label distribution: LoF={label_counts.get(0, 0)}, WT={label_counts.get(1, 0)}, GoF={label_counts.get(2, 0)}")

    # Log final species breakdown
    logger.info("Species breakdown (post-sampling):")
    for sp, cnt in merged["species"].fillna("").value_counts().items():
        logger.info(f"  {sp or '(unknown)'}: {cnt:,}")

    return merged


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split by label. All species in all splits.

    Per-species test subsets are created in main() from the test split.
    """
    from sklearn.model_selection import train_test_split as _tts

    df["gene"] = df["gene"].fillna("").astype(str)
    stratify_col = df["label"] if df["label"].nunique() > 1 else None

    min_needed = max(10, int(1 / test_size) + 1)
    if len(df) < min_needed:
        raise RuntimeError(
            f"Dataset has only {len(df)} records — too small to split "
            f"(need >= {min_needed}).\n"
            "Re-run:  python data/scripts/download_proteingym.py"
        )

    trainval_df, test_df = _tts(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=seed,
    )

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
    return train_df, val_df, test_df


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

    train_df, val_df, test_df = stratified_split(merged)

    # Save main splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = OUTPUT_DIR / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info(f"Saved {split_name}: {len(split_df):,} records -> {path}")

    # Save per-species test subsets for targeted evaluation
    species_col = test_df["species"].fillna("").astype(str)
    for tag, species_name in EVALUATION_SPECIES:
        mask = species_col.str.contains(species_name.split()[0], case=False, na=False)
        species_test = test_df[mask].reset_index(drop=True)
        if not species_test.empty:
            out_path = OUTPUT_DIR / f"test_{tag}.parquet"
            species_test.to_parquet(out_path, index=False)
            logger.info(f"Saved test_{tag}: {len(species_test):,} records -> {out_path}")
        else:
            logger.warning(f"No test records found for {species_name} (tag={tag}) -- skipping.")

    # Save full merged dataset
    merged_path = OUTPUT_DIR / "merged_all.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.info(f"Saved merged dataset: {len(merged):,} records -> {merged_path}")


if __name__ == "__main__":
    main()
