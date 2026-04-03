"""Preprocessing utilities for sequence data extraction and transformation."""

from __future__ import annotations

from pathlib import Path

from Bio import SeqIO


def load_fasta(fasta_path: str | Path) -> dict[str, str]:
    """Load sequences from a FASTA file.

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        Dict mapping sequence ID to sequence string.
    """
    sequences = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequences[record.id] = str(record.seq).upper()
    return sequences


def apply_snp_to_dna(ref_dna: str, position: int, alt_base: str) -> str:
    """Apply a single nucleotide polymorphism to a DNA sequence.

    Args:
        ref_dna: Reference DNA sequence.
        position: 0-based position of the SNP.
        alt_base: Alternative base (A/T/G/C).

    Returns:
        Variant DNA sequence with the SNP applied.
    """
    if position < 0 or position >= len(ref_dna):
        raise ValueError(f"Position {position} out of range for sequence of length {len(ref_dna)}")
    return ref_dna[:position] + alt_base.upper() + ref_dna[position + 1 :]


def apply_insertion(ref_dna: str, position: int, insert_seq: str) -> str:
    """Apply an insertion to a DNA sequence.

    Args:
        ref_dna: Reference DNA sequence.
        position: 0-based insertion position (inserted after this position).
        insert_seq: Sequence to insert.

    Returns:
        Variant DNA with insertion.
    """
    if position < 0 or position > len(ref_dna):
        raise ValueError(f"Position {position} out of range for sequence of length {len(ref_dna)}")
    return ref_dna[:position] + insert_seq.upper() + ref_dna[position:]


def apply_deletion(ref_dna: str, start: int, length: int) -> str:
    """Apply a deletion to a DNA sequence.

    Args:
        ref_dna: Reference DNA sequence.
        start: 0-based start position of deletion.
        length: Number of bases to delete.

    Returns:
        Variant DNA with deletion.
    """
    if start < 0 or start >= len(ref_dna):
        raise ValueError(f"Start {start} out of range for sequence of length {len(ref_dna)}")
    end = min(start + length, len(ref_dna))
    return ref_dna[:start] + ref_dna[end:]

