"""Download and process CARD (Comprehensive Antibiotic Resistance Database) data.

CARD provides curated AMR mutations — these are GoF (gain-of-function) variants
that confer antibiotic resistance in bacteria.

Source: https://card.mcmaster.ca/download
"""

from __future__ import annotations

import json
import logging
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen, Request

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
        req = Request(CARD_DATA_URL, headers={"User-Agent": "PLMLoF/1.0"})  # noqa: S310
        response = urlopen(req)  # noqa: S310 — trusted URL
        data = response.read()
        archive_path.write_bytes(data)
        logger.info(f"Downloaded {len(data) / 1e6:.1f} MB")

    # Extract — re-extract if card.json is missing
    extract_dir = output_dir / "extracted"
    card_json_candidates = list(extract_dir.rglob("card.json")) if extract_dir.exists() else []
    if not card_json_candidates:
        logger.info("Extracting CARD archive...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(extract_dir, filter="data")
        except TypeError:
            # Python < 3.12 doesn't have filter param — manually filter unsafe paths
            with tarfile.open(archive_path, "r:bz2") as tar:
                safe_members = [
                    m for m in tar.getmembers()
                    if not m.name.startswith("/") and ".." not in m.name
                ]
                tar.extractall(extract_dir, members=safe_members)

    return extract_dir


def _get_reference_sequence(model_sequences: dict) -> tuple[str, str, str]:
    """Extract reference protein, DNA, and species from model_sequences block.

    CARD structure: model_sequences -> sequence -> {id} -> protein_sequence/dna_sequence/NCBI_taxonomy

    Returns (ref_protein, ref_dna, species).
    """
    ref_protein = ""
    ref_dna = ""
    species = ""

    if not isinstance(model_sequences, dict):
        return ref_protein, ref_dna, species

    seq_block = model_sequences.get("sequence", {})
    if not isinstance(seq_block, dict):
        return ref_protein, ref_dna, species

    for _seq_id, seq_data in seq_block.items():
        if not isinstance(seq_data, dict):
            continue

        # protein_sequence -> sequence
        ps = seq_data.get("protein_sequence", {})
        if isinstance(ps, dict) and ps.get("sequence"):
            ref_protein = ref_protein or ps["sequence"]

        # dna_sequence -> sequence
        ds = seq_data.get("dna_sequence", {})
        if isinstance(ds, dict) and ds.get("sequence"):
            ref_dna = ref_dna or ds["sequence"]

        # NCBI_taxonomy -> NCBI_taxonomy_name
        tax = seq_data.get("NCBI_taxonomy", {})
        if isinstance(tax, dict) and tax.get("NCBI_taxonomy_name"):
            species = species or tax["NCBI_taxonomy_name"]

    return ref_protein, ref_dna, species


# Regex for mutation strings like "L157Q", "H481N", "A42G", etc.
_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _parse_mutation_string(mut_str: str, ref_protein: str) -> tuple[str, str] | None:
    """Parse a CARD mutation string like 'L157Q' and apply it to ref_protein.

    Returns (var_protein, mutation_label) or None if invalid.
    """
    mut_str = mut_str.strip()
    m = _MUT_RE.match(mut_str)
    if not m:
        return None

    ref_aa, pos_str, var_aa = m.group(1), m.group(2), m.group(3)
    pos = int(pos_str) - 1  # 0-based

    if pos < 0 or pos >= len(ref_protein):
        return None

    # Build variant
    var_protein = ref_protein[:pos] + var_aa + ref_protein[pos + 1:]
    mutation = f"{ref_aa}{pos + 1}{var_aa}"
    return var_protein, mutation


def parse_card_variants(card_dir: Path) -> pd.DataFrame:
    """Parse CARD JSON data to extract SNP-mediated resistance mutations (GoF).

    CARD model_param structure:
        model_param -> snp -> param_value -> {id: "L157Q", ...}
        model_param -> snp -> Curated-R  -> {id: "L157Q", ...}
        model_param -> snp -> clinical   -> {id: "L157Q", ...}

    Returns:
        DataFrame with columns: gene, species, ref_protein, var_protein,
        mutation, label, source.
    """
    json_path = card_dir / "card.json"
    if not json_path.exists():
        candidates = list(card_dir.rglob("card.json"))
        if not candidates:
            raise FileNotFoundError(f"card.json not found in {card_dir}")
        json_path = candidates[0]

    logger.info(f"Parsing CARD JSON: {json_path}")
    with open(json_path, "r") as f:
        card_data = json.load(f)

    records = []
    n_entries = 0
    n_variant_models = 0
    n_with_ref = 0
    n_with_snp = 0
    n_mutations_parsed = 0

    for key, entry in card_data.items():
        if not isinstance(entry, dict) or "ARO_accession" not in entry:
            continue
        n_entries += 1

        gene_name = entry.get("ARO_name", "")
        aro_id = entry.get("ARO_accession", "")
        model_type = entry.get("model_type", "")

        # Only process variant models
        if "variant" not in model_type.lower():
            continue
        n_variant_models += 1

        # Get reference sequence
        model_sequences = entry.get("model_sequences", {})
        ref_protein, ref_dna, species = _get_reference_sequence(model_sequences)
        if not ref_protein:
            continue
        n_with_ref += 1

        # Get SNP mutation strings from model_param -> snp -> param_value
        model_param = entry.get("model_param", {})
        if not isinstance(model_param, dict):
            continue

        snp_param = model_param.get("snp", {})
        if not isinstance(snp_param, dict):
            continue

        # Collect all mutation strings from param_value, Curated-R, clinical
        mutation_strings = set()
        for source_key in ("param_value", "Curated-R", "clinical"):
            source = snp_param.get(source_key, {})
            if isinstance(source, dict):
                for _mid, mut_entry in source.items():
                    if isinstance(mut_entry, str):
                        mutation_strings.add(mut_entry)
                    elif isinstance(mut_entry, dict):
                        # CARD param_value entries are dicts with
                        # "param_value_name" holding the mutation string
                        for name_key in ("param_value_name", "param_value_id"):
                            val = mut_entry.get(name_key, "")
                            if isinstance(val, str) and _MUT_RE.match(val.strip()):
                                mutation_strings.add(val.strip())
                                break

        if not mutation_strings:
            continue
        n_with_snp += 1

        for mut_str in mutation_strings:
            result = _parse_mutation_string(mut_str, ref_protein)
            if result is None:
                continue

            var_protein, mutation = result
            n_mutations_parsed += 1

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

    logger.info(
        f"CARD: {n_entries} entries, {n_variant_models} variant models, "
        f"{n_with_ref} with ref protein, {n_with_snp} with SNP strings, "
        f"{n_mutations_parsed} mutations parsed"
    )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["ref_protein", "var_protein", "mutation"])
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
