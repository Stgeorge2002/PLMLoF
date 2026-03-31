"""Engineered nucleotide-level features for variant classification."""

from __future__ import annotations

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


def extract_nucleotide_features(
    ref_dna: str,
    var_dna: str,
    ref_protein: str | None = None,
    var_protein: str | None = None,
) -> torch.Tensor:
    """Extract engineered nucleotide-level features from ref/var DNA pair.

    Features (12-dim vector):
        0: is_frameshift (bool → float)
        1: has_premature_stop (bool → float)
        2: start_codon_lost (bool → float)
        3: num_missense (int → float, normalized)
        4: num_nonsense (int → float)
        5: num_synonymous (int → float, normalized)
        6: truncation_fraction (float, 0.0-1.0)
        7: mutation_density (float, mutations per 100 residues)
        8: affected_region_n_terminal (bool → float)
        9: affected_region_c_terminal (bool → float)
        10: total_mutations (int → float, log-scaled)
        11: length_ratio (float, var_len / ref_len)

    Args:
        ref_dna: Reference DNA sequence.
        var_dna: Variant DNA sequence.
        ref_protein: Pre-computed reference protein (optional, will translate if None).
        var_protein: Pre-computed variant protein (optional, will translate if None).

    Returns:
        Tensor of shape [12] with float features.
    """
    if ref_protein is None:
        ref_protein = translate_dna(ref_dna)
    if var_protein is None:
        var_protein = translate_dna(var_dna)

    mutations = find_mutations(ref_protein, var_protein)

    # Count mutation types
    n_missense = sum(1 for m in mutations if m["type"] == "missense")
    n_nonsense = sum(1 for m in mutations if m["type"] == "nonsense")
    n_total = len(mutations)

    # Synonymous: positions where DNA differs but protein doesn't
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

    # Mutation density
    protein_len = max(len(ref_protein), 1)
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

    # Length ratio
    ref_dna_len = max(len(ref_dna), 1)
    length_ratio = len(var_dna) / ref_dna_len

    import math

    features = torch.tensor(
        [
            float(is_frameshift(ref_dna, var_dna)),           # 0
            float(has_premature_stop(ref_protein, var_protein)),  # 1
            float(start_codon_lost(ref_dna, var_dna)),        # 2
            n_missense / max(protein_len, 1),                 # 3  normalized
            float(n_nonsense),                                # 4
            n_synonymous / max(protein_len, 1),               # 5  normalized
            compute_truncation_fraction(ref_protein, var_protein),  # 6
            density,                                           # 7
            region_n,                                          # 8
            region_c,                                          # 9
            math.log1p(n_total),                              # 10  log-scaled
            length_ratio,                                      # 11
        ],
        dtype=torch.float32,
    )

    return features
