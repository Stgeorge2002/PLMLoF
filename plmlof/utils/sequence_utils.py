"""DNA/protein sequence utilities for PLMLoF."""

from __future__ import annotations

from Bio.Seq import Seq
from Bio.Data.CodonTable import TranslationError


BACTERIAL_CODON_TABLE = 11  # NCBI bacterial/archaeal/plant plastid table


def translate_dna(dna_seq: str, to_stop: bool = False) -> str:
    """Translate a DNA sequence to protein using the bacterial codon table.

    Args:
        dna_seq: DNA sequence string (A/T/G/C).
        to_stop: If True, translate only up to the first stop codon.

    Returns:
        Amino acid sequence string. Stop codons are represented as '*'.
    """
    dna_seq = dna_seq.upper().replace("U", "T")
    # Trim to multiple of 3
    trimmed = dna_seq[: len(dna_seq) - len(dna_seq) % 3]
    if not trimmed:
        return ""
    try:
        protein = str(Seq(trimmed).translate(table=BACTERIAL_CODON_TABLE))
    except TranslationError:
        protein = ""
    if to_stop and "*" in protein:
        protein = protein[: protein.index("*")]
    return protein


def reverse_complement(dna_seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return str(Seq(dna_seq.upper()).reverse_complement())


def find_mutations(ref_protein: str, var_protein: str) -> list[dict]:
    """Compare two aligned protein sequences and return a list of mutations.

    Both sequences should be the same length (or var may be shorter if truncated).
    Returns a list of dicts: {position, ref_aa, var_aa, type}.
    """
    mutations: list[dict] = []
    min_len = min(len(ref_protein), len(var_protein))

    for i in range(min_len):
        ref_aa = ref_protein[i]
        var_aa = var_protein[i]
        if ref_aa != var_aa:
            if var_aa == "*":
                mut_type = "nonsense"
            elif ref_aa == "*":
                mut_type = "readthrough"
            else:
                mut_type = "missense"
            mutations.append({
                "position": i + 1,  # 1-based
                "ref_aa": ref_aa,
                "var_aa": var_aa,
                "type": mut_type,
            })

    # Truncation or extension
    if len(var_protein) < len(ref_protein):
        mutations.append({
            "position": len(var_protein) + 1,
            "ref_aa": ref_protein[len(var_protein)] if len(var_protein) < len(ref_protein) else "-",
            "var_aa": "-",
            "type": "truncation",
        })
    elif len(var_protein) > len(ref_protein):
        mutations.append({
            "position": len(ref_protein) + 1,
            "ref_aa": "-",
            "var_aa": var_protein[len(ref_protein)],
            "type": "extension",
        })

    return mutations


def has_premature_stop(ref_protein: str, var_protein: str) -> bool:
    """Check if the variant introduces a premature stop codon."""
    # Find first stop in variant (excluding terminal stop of ref)
    ref_stop = ref_protein.find("*")
    var_stop = var_protein.find("*")
    if var_stop == -1:
        return False
    if ref_stop == -1:
        return True
    return var_stop < ref_stop


def is_frameshift(ref_dna: str, var_dna: str) -> bool:
    """Heuristic: detect if variant DNA length differs from reference by a non-multiple-of-3."""
    length_diff = abs(len(var_dna) - len(ref_dna))
    return length_diff % 3 != 0 and length_diff > 0


def start_codon_lost(ref_dna: str, var_dna: str) -> bool:
    """Check if the start codon (ATG) is disrupted in the variant."""
    ref_start = ref_dna[:3].upper()
    var_start = var_dna[:3].upper()
    return ref_start in ("ATG", "GTG", "TTG") and var_start not in ("ATG", "GTG", "TTG")


def compute_truncation_fraction(ref_protein: str, var_protein: str) -> float:
    """Compute the fraction of protein that is truncated in the variant."""
    if not ref_protein:
        return 0.0
    # Find effective length (up to first stop)
    ref_stop = ref_protein.find("*")
    var_stop = var_protein.find("*")
    ref_len = ref_stop if ref_stop != -1 else len(ref_protein)
    var_len = var_stop if var_stop != -1 else len(var_protein)
    if ref_len == 0:
        return 0.0
    if var_len >= ref_len:
        return 0.0
    return (ref_len - var_len) / ref_len
