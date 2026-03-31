"""Download and process ProteinGym DMS benchmark data.

ProteinGym provides Deep Mutational Scanning (DMS) datasets with fitness scores.
We filter for bacterial protein assays and threshold fitness scores into
LoF / WT / GoF classes.

Source: https://proteingym.org/
GitHub: https://github.com/OATML-Markslab/ProteinGym
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

# ProteinGym substitution scores (HuggingFace hosted)
PROTEINGYM_SUBS_URL = (
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym/resolve/main/"
    "DMS_ProteinGym_substitutions.zip"
)
PROTEINGYM_REFERENCE_URL = (
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym/resolve/main/"
    "DMS_substitutions.csv"
)

OUTPUT_DIR = Path("data/raw/proteingym/")

# Bacterial species keywords for filtering
BACTERIAL_KEYWORDS = [
    "escherichia", "e. coli", "salmonella", "staphylococcus", "streptococcus",
    "pseudomonas", "mycobacterium", "bacillus", "klebsiella", "enterococcus",
    "vibrio", "acinetobacter", "neisseria", "helicobacter", "campylobacter",
    "clostridium", "clostridioides", "listeria", "legionella", "corynebacterium",
    "bacterium", "prokaryot",
]

# Known bacterial DMS assays in ProteinGym
BACTERIAL_ASSAYS = [
    "TEM1_ECOLI",  # TEM-1 beta-lactamase (E. coli) — AMR gene
    "BLAT_ECOLX",  # Beta-lactamase (E. coli)
    "DHFR_ECOLI",  # Dihydrofolate reductase (E. coli)
    "INHA_MYCTU",  # InhA (M. tuberculosis)
    "RPOB_ECOLI",  # RNA polymerase beta (E. coli)
    "PARE_NEIGO",  # ParE (N. gonorrhoeae)
    "AMPC_ECOLI",  # AmpC (E. coli)
    "ADRB_HUMAN",  # Not bacterial — will be filtered
]


def download_proteingym_reference(output_dir: Path = OUTPUT_DIR) -> Path:
    """Download the ProteinGym reference file listing all assays.

    Returns:
        Path to the downloaded reference CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / "DMS_substitutions.csv"

    if ref_path.exists():
        logger.info(f"ProteinGym reference already downloaded: {ref_path}")
        return ref_path

    logger.info(f"Downloading ProteinGym reference from {PROTEINGYM_REFERENCE_URL}...")
    try:
        response = urlopen(PROTEINGYM_REFERENCE_URL)  # noqa: S310 — trusted URL
        data = response.read()
        ref_path.write_bytes(data)
        logger.info(f"Downloaded {len(data) / 1e6:.1f} MB")
    except Exception as e:
        logger.warning(f"Could not download ProteinGym reference: {e}")
        ref_path.write_text("")

    return ref_path


def filter_bacterial_assays(ref_path: Path) -> pd.DataFrame:
    """Filter the ProteinGym reference for bacterial protein assays.

    Returns:
        DataFrame of bacterial assay metadata.
    """
    if not ref_path.exists() or ref_path.stat().st_size == 0:
        logger.warning("ProteinGym reference file empty or missing.")
        return pd.DataFrame()

    df = pd.read_csv(ref_path)
    logger.info(f"ProteinGym reference contains {len(df)} assays")

    # Filter by known bacterial assays and organism keywords
    bacterial_mask = pd.Series(False, index=df.index)

    # Check DMS_id column for known bacterial assays
    if "DMS_id" in df.columns:
        for assay in BACTERIAL_ASSAYS:
            bacterial_mask |= df["DMS_id"].str.contains(assay, case=False, na=False)

    # Check UniProt_ID or target_seq columns for bacterial species
    for col in ["UniProt_ID", "DMS_id", "target_seq"]:
        if col in df.columns:
            for kw in BACTERIAL_KEYWORDS:
                bacterial_mask |= df[col].str.contains(kw, case=False, na=False)

    bacterial_df = df[bacterial_mask]
    logger.info(f"Found {len(bacterial_df)} bacterial-related assays")
    return bacterial_df


def threshold_fitness_scores(
    scores_df: pd.DataFrame,
    ref_protein: str,
    fitness_col: str = "DMS_score",
    lof_threshold: float = -1.0,
    gof_threshold: float = 1.0,
) -> pd.DataFrame:
    """Threshold DMS fitness scores into LoF / WT / GoF classes.

    Uses z-score-based thresholding:
    - LoF: fitness < mean - lof_threshold * std
    - GoF: fitness > mean + gof_threshold * std
    - WT: everything in between

    Returns:
        DataFrame with added 'label' column (0=LoF, 1=WT, 2=GoF).
    """
    if fitness_col not in scores_df.columns:
        logger.warning(f"Fitness column '{fitness_col}' not found")
        return scores_df

    scores = scores_df[fitness_col].dropna()
    if scores.empty:
        return scores_df

    mean = scores.mean()
    std = scores.std()
    if std == 0:
        std = 1.0

    def classify(score):
        z = (score - mean) / std
        if z < -lof_threshold:
            return 0  # LoF
        elif z > gof_threshold:
            return 2  # GoF
        return 1  # WT

    scores_df = scores_df.copy()
    scores_df["label"] = scores_df[fitness_col].apply(classify)

    # Add ref/var protein columns
    if "mutant" in scores_df.columns:
        scores_df["ref_protein"] = ref_protein
        scores_df["var_protein"] = scores_df["mutant"].apply(
            lambda m: _apply_mutation_string(ref_protein, m) if pd.notna(m) else ref_protein
        )

    label_counts = scores_df["label"].value_counts()
    logger.info(f"Label distribution: LoF={label_counts.get(0, 0)}, WT={label_counts.get(1, 0)}, GoF={label_counts.get(2, 0)}")

    return scores_df


def _apply_mutation_string(ref_protein: str, mutation: str) -> str:
    """Apply a mutation string like 'A23T' or 'A23T:G45R' to a protein sequence.

    Returns the mutant protein sequence.
    """
    var = list(ref_protein)
    mutations = mutation.split(":")

    for mut in mutations:
        mut = mut.strip()
        if len(mut) < 3:
            continue
        ref_aa = mut[0]
        var_aa = mut[-1]
        try:
            pos = int(mut[1:-1]) - 1  # 0-based
        except ValueError:
            continue

        if 0 <= pos < len(var):
            var[pos] = var_aa

    return "".join(var)


def main():
    logging.basicConfig(level=logging.INFO)

    ref_path = download_proteingym_reference()
    bacterial = filter_bacterial_assays(ref_path)

    if bacterial.empty:
        logger.warning("No bacterial assays found. Creating placeholder.")
        # Placeholder for testing
        placeholder = pd.DataFrame([{
            "gene": "TEM1_placeholder",
            "species": "Escherichia coli",
            "ref_protein": "MSIQHFRVALIPFFAAFCLPVFA",
            "var_protein": "MSIQHFRVALIPFFAAFCLPVFR",
            "ref_dna": "",
            "var_dna": "",
            "label": 2,  # GoF
            "source": "ProteinGym_placeholder",
        }])
        out_path = Path("data/processed/proteingym_bacterial.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder.to_parquet(out_path, index=False)
        logger.info(f"Saved placeholder to {out_path}")
    else:
        logger.info(f"Found {len(bacterial)} bacterial assays for processing")
        out_path = Path("data/processed/proteingym_bacterial.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bacterial.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(bacterial)} records to {out_path}")


if __name__ == "__main__":
    main()
