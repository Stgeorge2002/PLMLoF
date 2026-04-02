"""Engineered nucleotide-level features for variant classification."""

from __future__ import annotations

import math

import torch

from plmlof.utils.sequence_utils import (
    translate_dna,
    find_mutations,
    has_premature_stop,
    is_frameshift,
    start_codon_lost,
    compute_truncation_fraction,
)


NUM_NUCLEOTIDE_FEATURES = 12

# Region encoding: 0=N-terminal (first 20%), 1=middle, 2=C-terminal (last 20%)
_REGION_MAP = {"N-terminal": 0, "middle": 1, "C-terminal": 2, "none": 1}


def _classify_region(position: int, protein_length: int) -> str:
    """Classify mutation position as N-terminal, middle, or C-terminal."""
    if protein_length == 0:
        return "none"
    frac = position / protein_length
    if frac <= 0.2:
        return "N-terminal"
    elif frac >= 0.8:
        return "C-terminal"
    return "middle"


def _sequence_identity(ref: str, var: str) -> float:
    """Fraction of aligned positions where ref and var match."""
    if not ref and not var:
        return 1.0
    min_len = min(len(ref), len(var))
    max_len = max(len(ref), len(var))
    if max_len == 0:
        return 1.0
    matches = sum(1 for i in range(min_len) if ref[i] == var[i])
    return matches / max_len


def extract_nucleotide_features(
    ref_dna: str,
    var_dna: str,
    ref_protein: str | None = None,
    var_protein: str | None = None,
) -> torch.Tensor:
    """Extract engineered features from ref/var sequence pair.

    When DNA sequences are available, DNA-based features are used for
    positions 0, 2, 5, 11. When only protein sequences are available
    (e.g. ProteinGym), protein-only alternatives are used instead:

    Features (12-dim vector):
        0: is_frameshift (DNA) / is_length_change (protein-only)
        1: has_premature_stop (bool → float)
        2: start_codon_lost (DNA) / met_start_lost (protein-only)
        3: num_missense (int → float, normalized)
        4: num_nonsense (int → float)
        5: num_synonymous (DNA) / 1 - sequence_identity (protein-only)
        6: truncation_fraction (float, 0.0-1.0)
        7: mutation_density (float, mutations per 100 residues)
        8: affected_region_n_terminal (bool → float)
        9: affected_region_c_terminal (bool → float)
        10: total_mutations (int → float, log-scaled)
        11: length_ratio (DNA) / protein_length_ratio (protein-only)

    Args:
        ref_dna: Reference DNA sequence.
        var_dna: Variant DNA sequence.
        ref_protein: Pre-computed reference protein (optional, will translate if None).
        var_protein: Pre-computed variant protein (optional, will translate if None).

    Returns:
        Tensor of shape [12] with float features.
    """
    has_dna = bool(ref_dna) and bool(var_dna)

    if ref_protein is None:
        ref_protein = translate_dna(ref_dna) if ref_dna else ""
    if var_protein is None:
        var_protein = translate_dna(var_dna) if var_dna else ""

    mutations = find_mutations(ref_protein, var_protein)

    # Count mutation types
    n_missense = sum(1 for m in mutations if m["type"] == "missense")
    n_nonsense = sum(1 for m in mutations if m["type"] == "nonsense")
    n_total = len(mutations)

    protein_len = max(len(ref_protein), 1)

    # Feature 0: frameshift (DNA) or length change (protein)
    if has_dna:
        feat_0 = float(is_frameshift(ref_dna, var_dna))
    else:
        feat_0 = float(len(var_protein) != len(ref_protein))

    # Feature 2: start codon lost (DNA) or met start lost (protein)
    if has_dna:
        feat_2 = float(start_codon_lost(ref_dna, var_dna))
    else:
        feat_2 = float(
            len(ref_protein) > 0
            and ref_protein[0] == "M"
            and (len(var_protein) == 0 or var_protein[0] != "M")
        )

    # Feature 5: synonymous mutations (DNA) or 1-sequence_identity (protein)
    if has_dna:
        ref_len = min(len(ref_dna), len(var_dna))
        n_synonymous = 0
        for i in range(0, ref_len - 2, 3):
            ref_codon = ref_dna[i : i + 3].upper()
            var_codon = var_dna[i : i + 3].upper()
            if ref_codon != var_codon:
                aa_pos = i // 3
                if aa_pos < len(ref_protein) and aa_pos < len(var_protein):
                    if ref_protein[aa_pos] == var_protein[aa_pos]:
                        n_synonymous += 1
        feat_5 = n_synonymous / max(protein_len, 1)
    else:
        feat_5 = 1.0 - _sequence_identity(ref_protein, var_protein)

    # Feature 11: DNA length ratio or protein length ratio
    if has_dna:
        feat_11 = len(var_dna) / max(len(ref_dna), 1)
    else:
        feat_11 = len(var_protein) / max(len(ref_protein), 1)

    # Mutation density
    density = (n_total / protein_len) * 100

    # Affected region (average if multiple mutations)
    region_n = 0.0
    region_c = 0.0
    if mutations:
        pos_mutations = [m for m in mutations if "position" in m]
        if pos_mutations:
            avg_pos = sum(m["position"] for m in pos_mutations) / len(pos_mutations)
            region = _classify_region(int(avg_pos), protein_len)
            region_n = float(region == "N-terminal")
            region_c = float(region == "C-terminal")

    features = torch.tensor(
        [
            feat_0,                                            # 0
            float(has_premature_stop(ref_protein, var_protein)),  # 1
            feat_2,                                            # 2
            n_missense / max(protein_len, 1),                  # 3  normalized
            float(n_nonsense),                                 # 4
            feat_5,                                            # 5
            compute_truncation_fraction(ref_protein, var_protein),  # 6
            density,                                           # 7
            region_n,                                          # 8
            region_c,                                          # 9
            math.log1p(n_total),                               # 10  log-scaled
            feat_11,                                           # 11
        ],
        dtype=torch.float32,
    )

    return features
