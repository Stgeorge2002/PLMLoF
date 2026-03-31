"""VCF and FASTA input handling for PLMLoF inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqIO

from plmlof.data.features import extract_nucleotide_features
from plmlof.data.preprocessing import apply_snp_to_dna, apply_insertion, apply_deletion
from plmlof.utils.sequence_utils import translate_dna

logger = logging.getLogger(__name__)


@dataclass
class VariantRecord:
    """A single gene-level variant record for PLMLoF prediction."""
    gene: str
    ref_protein: str
    var_protein: str
    ref_dna: str = ""
    var_dna: str = ""
    variant_details: list[dict] = field(default_factory=list)
    chrom: str = ""
    position: int = 0


def load_reference_proteins(fasta_path: str | Path) -> dict[str, str]:
    """Load reference gene sequences from FASTA and translate to protein.

    Expects a FASTA file with gene DNA sequences (CDS).
    Returns dict of gene_id → protein_sequence.
    """
    proteins = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        dna = str(record.seq).upper()
        protein = translate_dna(dna, to_stop=True)
        proteins[record.id] = protein
    return proteins


def load_reference_dna(fasta_path: str | Path) -> dict[str, str]:
    """Load reference DNA sequences from FASTA.

    Returns dict of gene_id → dna_sequence.
    """
    seqs = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seqs[record.id] = str(record.seq).upper()
    return seqs


def parse_vcf_variants(
    vcf_path: str | Path,
    reference_dna: dict[str, str],
    gene_annotations: dict[str, tuple[str, int, int]] | None = None,
) -> list[VariantRecord]:
    """Parse a VCF file and produce gene-level variant records.

    This is a simplified parser for bacterial VCFs where CHROM typically
    corresponds to a gene/contig and variants are applied to produce variant
    gene sequences.

    Args:
        vcf_path: Path to VCF file.
        reference_dna: Dict of gene_id → reference DNA sequence.
        gene_annotations: Optional dict of gene_id → (chrom, start, end)
            for mapping variants to genes. If None, CHROM is treated as gene_id.

    Returns:
        List of VariantRecord for each gene with variants.
    """
    # Collect variants per gene
    gene_variants: dict[str, list[dict]] = {}

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            chrom = parts[0]
            pos = int(parts[1]) - 1  # Convert to 0-based
            ref_allele = parts[3]
            alt_allele = parts[4]

            # Determine gene
            gene_id = chrom  # Default: CHROM = gene
            local_pos = pos

            if gene_annotations:
                for gid, (gchr, gstart, gend) in gene_annotations.items():
                    if gchr == chrom and gstart <= pos < gend:
                        gene_id = gid
                        local_pos = pos - gstart
                        break

            if gene_id not in gene_variants:
                gene_variants[gene_id] = []
            gene_variants[gene_id].append({
                "pos": local_pos,
                "ref": ref_allele,
                "alt": alt_allele,
                "chrom": chrom,
                "original_pos": pos,
            })

    # Apply variants to each gene and produce records
    records = []
    for gene_id, variants in gene_variants.items():
        if gene_id not in reference_dna:
            logger.warning(f"Gene {gene_id} not found in reference. Skipping.")
            continue

        ref_dna = reference_dna[gene_id]
        var_dna = ref_dna

        # Sort variants by position (reverse order to avoid index shifts)
        sorted_variants = sorted(variants, key=lambda v: v["pos"], reverse=True)

        for var in sorted_variants:
            p = var["pos"]
            ref_a = var["ref"]
            alt_a = var["alt"]

            if len(ref_a) == 1 and len(alt_a) == 1:
                # SNP
                var_dna = apply_snp_to_dna(var_dna, p, alt_a)
            elif len(ref_a) < len(alt_a):
                # Insertion
                var_dna = apply_insertion(var_dna, p + len(ref_a), alt_a[len(ref_a):])
            elif len(ref_a) > len(alt_a):
                # Deletion
                del_len = len(ref_a) - len(alt_a)
                var_dna = apply_deletion(var_dna, p + len(alt_a), del_len)

        ref_protein = translate_dna(ref_dna, to_stop=True)
        var_protein = translate_dna(var_dna, to_stop=True)

        records.append(VariantRecord(
            gene=gene_id,
            ref_protein=ref_protein,
            var_protein=var_protein,
            ref_dna=ref_dna,
            var_dna=var_dna,
            variant_details=variants,
            chrom=variants[0]["chrom"] if variants else "",
        ))

    return records


def parse_fasta_pairs(
    reference_fasta: str | Path,
    variant_fasta: str | Path,
) -> list[VariantRecord]:
    """Parse paired reference and variant FASTA files.

    Each pair of sequences with matching IDs is treated as a ref/var comparison.
    Sequences can be DNA (will be translated) or protein.

    Args:
        reference_fasta: Path to reference FASTA.
        variant_fasta: Path to variant FASTA.

    Returns:
        List of VariantRecord.
    """
    ref_seqs = {}
    for record in SeqIO.parse(str(reference_fasta), "fasta"):
        ref_seqs[record.id] = str(record.seq).upper()

    records = []
    for record in SeqIO.parse(str(variant_fasta), "fasta"):
        gene_id = record.id
        var_seq = str(record.seq).upper()

        if gene_id not in ref_seqs:
            logger.warning(f"Gene {gene_id} not found in reference. Skipping.")
            continue

        ref_seq = ref_seqs[gene_id]

        # Detect if sequences are DNA or protein
        dna_chars = set("ATGCN")
        is_dna = all(c in dna_chars for c in ref_seq[:100])

        if is_dna:
            ref_dna = ref_seq
            var_dna = var_seq
            ref_protein = translate_dna(ref_dna, to_stop=True)
            var_protein = translate_dna(var_dna, to_stop=True)
        else:
            ref_dna = ""
            var_dna = ""
            ref_protein = ref_seq.replace("*", "")
            var_protein = var_seq.replace("*", "")

        records.append(VariantRecord(
            gene=gene_id,
            ref_protein=ref_protein,
            var_protein=var_protein,
            ref_dna=ref_dna,
            var_dna=var_dna,
        ))

    return records
