"""Download and process CARD (Comprehensive Antibiotic Resistance Database) data.

CARD provides curated AMR mutations — these are GoF (gain-of-function) variants
that confer antibiotic resistance in bacteria.

Source: https://card.mcmaster.ca/download
"""

from __future__ import annotations

import json
import logging
import tarfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

CARD_DATA_URL = "https://card.mcmaster.ca/latest/data"
OUTPUT_DIR = Path("data/raw/card/")


def download_card(output_dir: Path = OUTPUT_DIR) -> Path:
    """Download CARD data archive.

    Returns:
        Path to the extracted data directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "card-data.tar.bz2"

    if archive_path.exists():
        logger.info(f"CARD archive already downloaded: {archive_path}")
    else:
        logger.info(f"Downloading CARD data from {CARD_DATA_URL}...")
        response = urlopen(CARD_DATA_URL)  # noqa: S310 — trusted URL
        data = response.read()
        archive_path.write_bytes(data)
        logger.info(f"Downloaded {len(data) / 1e6:.1f} MB")

    # Extract
    extract_dir = output_dir / "extracted"
    if not extract_dir.exists():
        logger.info("Extracting CARD archive...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(extract_dir, filter="data")

    return extract_dir


def parse_card_variants(card_dir: Path) -> pd.DataFrame:
    """Parse CARD JSON data to extract SNP-mediated resistance mutations (GoF).

    Returns:
        DataFrame with columns: gene, species, ref_protein, var_protein,
        mutation, mechanism, drug_class, source.
    """
    json_path = card_dir / "card.json"
    if not json_path.exists():
        # Try to find it in subdirectories
        candidates = list(card_dir.rglob("card.json"))
        if not candidates:
            raise FileNotFoundError(f"card.json not found in {card_dir}")
        json_path = candidates[0]

    logger.info(f"Parsing CARD JSON: {json_path}")
    with open(json_path, "r") as f:
        card_data = json.load(f)

    records = []
    for key, entry in card_data.items():
        # Skip metadata entries
        if not isinstance(entry, dict) or "ARO_accession" not in entry:
            continue

        gene_name = entry.get("ARO_name", "")
        aro_id = entry.get("ARO_accession", "")

        # Look for SNP models
        snp_models = entry.get("model_sequences", {})
        for seq_key, seq_data in snp_models.items():
            if not isinstance(seq_data, dict):
                continue

            for variant_key, variant_data in seq_data.items():
                if not isinstance(variant_data, dict):
                    continue

                # Extract protein sequence
                protein_seq = variant_data.get("protein_sequence", {})
                if isinstance(protein_seq, dict):
                    ref_protein = protein_seq.get("sequence", "")
                else:
                    ref_protein = ""

                dna_seq = variant_data.get("dna_sequence", {})
                if isinstance(dna_seq, dict):
                    ref_dna = dna_seq.get("sequence", "")
                else:
                    ref_dna = ""

                # Extract SNPs
                snps = variant_data.get("snps", {})
                if not snps:
                    continue

                for snp_key, snp_data in snps.items():
                    if not isinstance(snp_data, dict):
                        continue

                    mutation = snp_data.get("original", "") + str(snp_data.get("position", "")) + snp_data.get("change", "")

                    # Construct variant protein
                    pos = snp_data.get("position")
                    original = snp_data.get("original", "")
                    change = snp_data.get("change", "")

                    if pos is not None and ref_protein and change:
                        pos = int(pos) - 1  # 0-based
                        if 0 <= pos < len(ref_protein):
                            var_protein = ref_protein[:pos] + change + ref_protein[pos + 1:]
                        else:
                            var_protein = ref_protein
                    else:
                        continue

                    # Species info
                    species = ""
                    ncbi_taxonomy = variant_data.get("NCBI_taxonomy", {})
                    if isinstance(ncbi_taxonomy, dict):
                        species = ncbi_taxonomy.get("NCBI_taxonomy_name", "")

                    records.append({
                        "gene": gene_name,
                        "aro_id": aro_id,
                        "species": species,
                        "ref_protein": ref_protein,
                        "var_protein": var_protein,
                        "ref_dna": ref_dna,
                        "mutation": mutation,
                        "label": 2,  # GoF
                        "source": "CARD",
                    })

    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} GoF variants from CARD")
    return df


def main():
    logging.basicConfig(level=logging.INFO)
    card_dir = download_card()
    df = parse_card_variants(card_dir)

    out_path = Path("data/processed/card_gof.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} records to {out_path}")


if __name__ == "__main__":
    main()
