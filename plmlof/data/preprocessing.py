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


def introduce_premature_stop(dna_seq: str, codon_position: int) -> str:
    """Replace a codon with a stop codon (TAA) at the given codon position.

    Args:
        dna_seq: DNA sequence.
        codon_position: 0-based codon index (not nucleotide position).

    Returns:
        DNA with premature stop codon.
    """
    nuc_pos = codon_position * 3
    if nuc_pos + 3 > len(dna_seq):
        raise ValueError(f"Codon position {codon_position} out of range")
    return dna_seq[:nuc_pos] + "TAA" + dna_seq[nuc_pos + 3 :]


def introduce_frameshift(dna_seq: str, position: int, insert: bool = True) -> str:
    """Introduce a frameshift mutation by inserting or deleting 1 base.

    Args:
        dna_seq: DNA sequence.
        position: 0-based nucleotide position.
        insert: If True, insert a base; if False, delete a base.

    Returns:
        DNA with frameshift.
    """
    if insert:
        return apply_insertion(dna_seq, position, "A")
    else:
        return apply_deletion(dna_seq, position, 1)
