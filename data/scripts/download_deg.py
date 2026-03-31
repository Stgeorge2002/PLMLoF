"""Download and process DEG (Database of Essential Genes) for LoF variant generation.

DEG contains experimentally validated essential genes in bacteria.
Disrupting these genes leads to loss of function (LoF).
We generate synthetic LoF variants by introducing disruptive mutations.

Source: https://tubic.org/deg/
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from Bio import SeqIO
from io import StringIO

from plmlof.utils.sequence_utils import translate_dna
from plmlof.data.preprocessing import introduce_premature_stop, introduce_frameshift

logger = logging.getLogger(__name__)

# DEG provides FASTA downloads for bacterial essential genes
DEG_PROTEIN_URL = "https://tubic.org/deg/public/download/deg-p-e.dat"
DEG_DNA_URL = "https://tubic.org/deg/public/download/deg-n-e.dat"

OUTPUT_DIR = Path("data/raw/deg/")


def download_deg(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Download DEG protein and nucleotide files.

    Returns:
        Tuple of (protein_path, dna_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_path = output_dir / "deg_proteins.dat"
    dna_path = output_dir / "deg_dna.dat"

    for url, path, desc in [
        (DEG_PROTEIN_URL, protein_path, "protein"),
        (DEG_DNA_URL, dna_path, "nucleotide"),
    ]:
        if path.exists():
            logger.info(f"DEG {desc} data already downloaded: {path}")
        else:
            logger.info(f"Downloading DEG {desc} data from {url}...")
            try:
                response = urlopen(url)  # noqa: S310 — trusted URL
                data = response.read()
                path.write_bytes(data)
                logger.info(f"Downloaded {len(data) / 1e6:.1f} MB")
            except Exception as e:
                logger.warning(f"Could not download DEG {desc}: {e}")
                # Create empty file
                path.write_text("")

    return protein_path, dna_path


def parse_deg_sequences(protein_path: Path, dna_path: Path) -> pd.DataFrame:
    """Parse DEG FASTA files and extract bacterial essential gene sequences.

    Returns:
        DataFrame with columns: gene_id, gene, species, ref_protein, ref_dna
    """
    records = []

    # Parse protein sequences
    proteins = {}
    if protein_path.exists() and protein_path.stat().st_size > 0:
        try:
            for record in SeqIO.parse(str(protein_path), "fasta"):
                proteins[record.id] = {
                    "protein": str(record.seq),
                    "description": record.description,
                }
        except Exception as e:
            logger.warning(f"Error parsing DEG proteins: {e}")

    # Parse DNA sequences
    dna_seqs = {}
    if dna_path.exists() and dna_path.stat().st_size > 0:
        try:
            for record in SeqIO.parse(str(dna_path), "fasta"):
                dna_seqs[record.id] = str(record.seq).upper()
        except Exception as e:
            logger.warning(f"Error parsing DEG DNA: {e}")

    # Merge
    all_ids = set(proteins.keys()) | set(dna_seqs.keys())
    for gid in all_ids:
        protein = proteins.get(gid, {}).get("protein", "")
        desc = proteins.get(gid, {}).get("description", "")
        dna = dna_seqs.get(gid, "")

        # If we have DNA but no protein, translate
        if dna and not protein:
            protein = translate_dna(dna, to_stop=True)

        # Extract species from description (DEG format: "DEG_ID gene_name - Species")
        species = ""
        if " - " in desc:
            species = desc.split(" - ")[-1].strip()

        # Filter for bacterial species (heuristic: exclude human, mouse, yeast, etc.)
        skip_species = {"homo sapiens", "mus musculus", "saccharomyces", "drosophila",
                        "caenorhabditis", "arabidopsis", "danio rerio"}
        if any(s in species.lower() for s in skip_species):
            continue

        if protein:
            records.append({
                "gene_id": gid,
                "gene": desc.split()[1] if len(desc.split()) > 1 else gid,
                "species": species,
                "ref_protein": protein,
                "ref_dna": dna,
            })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} bacterial essential genes from DEG")
    return df


def generate_lof_variants(
    essential_genes: pd.DataFrame,
    variants_per_gene: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic LoF variants from essential genes.

    For each gene, generates variants with:
    - Premature stop codons
    - Frameshift mutations
    - Large truncations

    Returns:
        DataFrame with columns: gene, species, ref_protein, var_protein,
        ref_dna, var_dna, mutation_type, label, source
    """
    rng = random.Random(seed)
    records = []

    for _, row in essential_genes.iterrows():
        ref_protein = row["ref_protein"]
        ref_dna = row["ref_dna"]
        gene = row["gene"]
        species = row.get("species", "")

        if len(ref_protein) < 10:
            continue

        for v in range(variants_per_gene):
            if v == 0 and ref_dna and len(ref_dna) >= 30:
                # Premature stop at random position (first 80% of gene)
                max_codon = int(len(ref_protein) * 0.8)
                if max_codon < 2:
                    continue
                codon_pos = rng.randint(1, max_codon)
                try:
                    var_dna = introduce_premature_stop(ref_dna, codon_pos)
                    var_protein = translate_dna(var_dna, to_stop=True)
                    mutation_type = f"premature_stop_at_codon_{codon_pos}"
                except (ValueError, IndexError):
                    continue
            elif v == 1 and ref_dna and len(ref_dna) >= 30:
                # Frameshift in first half
                pos = rng.randint(3, len(ref_dna) // 2)
                try:
                    var_dna = introduce_frameshift(ref_dna, pos, insert=True)
                    var_protein = translate_dna(var_dna, to_stop=True)
                    mutation_type = f"frameshift_insert_at_{pos}"
                except (ValueError, IndexError):
                    continue
            else:
                # Large truncation (keep only 10-50% of protein)
                keep_frac = rng.uniform(0.1, 0.5)
                keep_len = max(int(len(ref_protein) * keep_frac), 1)
                var_protein = ref_protein[:keep_len]
                var_dna = ref_dna[:keep_len * 3] if ref_dna else ""
                mutation_type = f"truncation_{keep_frac:.0%}"

            records.append({
                "gene": gene,
                "species": species,
                "ref_protein": ref_protein,
                "var_protein": var_protein,
                "ref_dna": ref_dna,
                "var_dna": var_dna,
                "mutation_type": mutation_type,
                "label": 0,  # LoF
                "source": "DEG_synthetic",
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic LoF variants from {len(essential_genes)} essential genes")
    return df


def main():
    logging.basicConfig(level=logging.INFO)

    protein_path, dna_path = download_deg()
    essential = parse_deg_sequences(protein_path, dna_path)

    if essential.empty:
        logger.warning("No essential genes parsed. Creating placeholder dataset.")
        # Create a small placeholder for testing
        essential = pd.DataFrame([{
            "gene_id": "DEG0001",
            "gene": "placeholder_gene",
            "species": "Test bacterium",
            "ref_protein": "MKTLLLTLVVVTLAALGSHYDAIQ",
            "ref_dna": "ATGAAAACCCTGCTGCTGACCCTGGTGGTGGTGACCCTGGCGGCGCTGGGCTCGCACTACGATGCGATCCAG",
        }])

    lof_df = generate_lof_variants(essential)

    out_path = Path("data/processed/deg_lof.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lof_df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(lof_df)} records to {out_path}")


if __name__ == "__main__":
    main()
