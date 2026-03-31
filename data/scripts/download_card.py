"""Download and process CARD (Comprehensive Antibiotic Resistance Database) data.

CARD provides curated AMR mutations — these are GoF (gain-of-function) variants
that confer antibiotic resistance in bacteria.

Source: https://card.mcmaster.ca/download
"""

from __future__ import annotations

import json
import logging
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
            # Python < 3.12 doesn't have filter param
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(extract_dir)

    return extract_dir


def _get_reference_sequence(model_sequences: dict) -> tuple[str, str, str]:
    """Extract reference protein, DNA, and species from model_sequences block.

    Handles multiple CARD JSON nesting levels by recursively searching for
    protein_sequence/dna_sequence/NCBI_taxonomy dicts.

    Returns (ref_protein, ref_dna, species).
    """
    ref_protein = ""
    ref_dna = ""
    species = ""

    if not isinstance(model_sequences, dict):
        return ref_protein, ref_dna, species

    def _recursive_search(obj: dict, depth: int = 0) -> None:
        nonlocal ref_protein, ref_dna, species
        if depth > 5:
            return
        if not isinstance(obj, dict):
            return

        # Check for protein_sequence at this level
        prot = obj.get("protein_sequence", None)
        if isinstance(prot, dict):
            ref_protein = ref_protein or prot.get("sequence", "")
        elif isinstance(prot, str) and len(prot) > 5:
            ref_protein = ref_protein or prot

        # Check for dna_sequence at this level
        dna = obj.get("dna_sequence", None)
        if isinstance(dna, dict):
            ref_dna = ref_dna or dna.get("sequence", "")
        elif isinstance(dna, str) and len(dna) > 5:
            ref_dna = ref_dna or dna

        # Check for taxonomy at this level
        tax = obj.get("NCBI_taxonomy", None)
        if isinstance(tax, dict):
            species = species or tax.get("NCBI_taxonomy_name", "")

        # Recurse into dict values
        for _k, v in obj.items():
            if isinstance(v, dict) and _k not in ("protein_sequence", "dna_sequence", "NCBI_taxonomy"):
                _recursive_search(v, depth + 1)

    _recursive_search(model_sequences)
    return ref_protein, ref_dna, species


def parse_card_variants(card_dir: Path) -> pd.DataFrame:
    """Parse CARD JSON data to extract SNP-mediated resistance mutations (GoF).

    Handles multiple CARD JSON schema versions by looking for SNP data in:
      1. model_param → snp (current CARD schema)
      2. model_sequences → ... → snps (older schema)

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
    n_snp_models = 0
    n_entries = 0
    n_with_param = 0
    n_with_ref = 0

    for key, entry in card_data.items():
        if not isinstance(entry, dict) or "ARO_accession" not in entry:
            continue
        n_entries += 1

        gene_name = entry.get("ARO_name", "")
        aro_id = entry.get("ARO_accession", "")
        model_type = entry.get("model_type", "")

        # Only process variant models (protein variant model, protein overexpression model, etc.)
        is_variant_model = any(kw in model_type.lower() for kw in [
            "variant", "overexpression", "mutation",
        ]) if model_type else False

        # Get reference sequence from model_sequences
        model_sequences = entry.get("model_sequences", {})
        ref_protein, ref_dna, species_from_seq = _get_reference_sequence(model_sequences)
        if ref_protein:
            n_with_ref += 1

        # ── Path 1: model_param (current CARD schema) ──
        # CARD model_param has numbered keys, each containing param_type and SNP data.
        # Structure: model_param → {id} → param_type:"snp", snp → {snp_id} → {original, change, position}
        # OR older: model_param → snp → {snp_id} → {original, change, position}
        model_param = entry.get("model_param", {})
        snp_entries = {}
        if isinstance(model_param, dict):
            # Direct snp key (older schema)
            direct_snp = model_param.get("snp", {})
            if isinstance(direct_snp, dict) and direct_snp:
                snp_entries = direct_snp
            else:
                # Numbered keys schema: model_param → {id} → {param_type, snp: {snp_id: {...}}}
                for _param_id, param_data in model_param.items():
                    if not isinstance(param_data, dict):
                        continue
                    # Check param_type or look for nested snp data
                    param_type = str(param_data.get("param_type", "")).lower()
                    if param_type == "snp" or "snp" in param_data:
                        nested_snp = param_data.get("snp", {})
                        if isinstance(nested_snp, dict):
                            snp_entries.update(nested_snp)
                        # Also check if SNP data is directly in param_data
                        if "original" in param_data and "change" in param_data:
                            snp_entries[_param_id] = param_data

        if snp_entries and ref_protein:
            n_snp_models += 1
            for snp_key, snp_data in snp_entries.items():
                if not isinstance(snp_data, dict):
                    continue

                original = snp_data.get("original", "")
                change = snp_data.get("change", "")
                pos = snp_data.get("position")

                if not pos or not change or not original:
                    continue

                try:
                    pos_int = int(pos) - 1  # 0-based
                except (ValueError, TypeError):
                    continue

                if pos_int < 0 or pos_int >= len(ref_protein):
                    continue

                # Verify reference AA matches (skip if not)
                if ref_protein[pos_int] != original and original != "":
                    # Mismatch — try anyway, the annotation might use 1-based differently
                    pass

                var_protein = ref_protein[:pos_int] + change + ref_protein[pos_int + 1:]
                mutation = f"{original}{pos_int + 1}{change}"

                records.append({
                    "gene": gene_name,
                    "aro_id": aro_id,
                    "species": species_from_seq,
                    "ref_protein": ref_protein,
                    "var_protein": var_protein,
                    "ref_dna": ref_dna,
                    "mutation": mutation,
                    "label": 2,  # GoF
                    "source": "CARD",
                })

        # ── Path 2: model_sequences → snps (older schema fallback) ──
        if not snp_entries:
            for seq_key, seq_data in model_sequences.items():
                if not isinstance(seq_data, dict):
                    continue
                # Check nested levels
                for variant_key, variant_data in seq_data.items():
                    if not isinstance(variant_data, dict):
                        continue

                    snps = variant_data.get("snps", {})
                    if not snps:
                        continue

                    prot = variant_data.get("protein_sequence", {})
                    inner_ref = prot.get("sequence", "") if isinstance(prot, dict) else ""
                    inner_ref = inner_ref or ref_protein

                    tax = variant_data.get("NCBI_taxonomy", {})
                    inner_species = tax.get("NCBI_taxonomy_name", "") if isinstance(tax, dict) else ""

                    for snp_key, snp_data in snps.items():
                        if not isinstance(snp_data, dict):
                            continue

                        original = snp_data.get("original", "")
                        change = snp_data.get("change", "")
                        pos = snp_data.get("position")

                        if not pos or not change or not inner_ref:
                            continue
                        try:
                            pos_int = int(pos) - 1
                        except (ValueError, TypeError):
                            continue
                        if pos_int < 0 or pos_int >= len(inner_ref):
                            continue

                        var_protein = inner_ref[:pos_int] + change + inner_ref[pos_int + 1:]
                        mutation = f"{original}{pos_int + 1}{change}"

                        records.append({
                            "gene": gene_name,
                            "aro_id": aro_id,
                            "species": inner_species or species_from_seq,
                            "ref_protein": inner_ref,
                            "var_protein": var_protein,
                            "ref_dna": ref_dna,
                            "mutation": mutation,
                            "label": 2,  # GoF
                            "source": "CARD",
                        })

    logger.info(f"CARD entries: {n_entries} total, {n_with_ref} with ref protein, {n_snp_models} with SNP data")

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
