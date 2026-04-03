"""Engineered protein-level features for variant classification.

All features are derived from protein sequences only, as the training data
(ProteinGym DMS) provides protein but not DNA sequences.
"""

from __future__ import annotations

import math

import torch

from plmlof.utils.sequence_utils import (
    find_mutations,
    has_premature_stop,
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
    ref_protein: str,
    var_protein: str,
) -> torch.Tensor:
    """Extract engineered features from a ref/var protein sequence pair.

    All features are protein-derived (ProteinGym provides no DNA).

    Features (12-dim vector):
        0: is_length_change (bool → float)
        1: has_premature_stop (bool → float)
        2: met_start_lost (bool → float)
        3: num_missense (normalized by protein length)
        4: num_nonsense (int → float)
        5: 1 - sequence_identity (float)
        6: truncation_fraction (float, 0.0-1.0)
        7: mutation_density (mutations per 100 residues)
        8: fraction_n_terminal (float, 0.0-1.0)
        9: fraction_c_terminal (float, 0.0-1.0)
        10: total_mutations (log-scaled)
        11: protein_length_ratio (var_len / ref_len)

    Args:
        ref_protein: Reference protein sequence.
        var_protein: Variant protein sequence.

    Returns:
        Tensor of shape [12] with float features.
    """
    if ref_protein is None:
        ref_protein = ""
    if var_protein is None:
        var_protein = ""

    mutations = find_mutations(ref_protein, var_protein)

    n_missense = sum(1 for m in mutations if m["type"] == "missense")
    n_nonsense = sum(1 for m in mutations if m["type"] == "nonsense")
    n_total = len(mutations)

    protein_len = max(len(ref_protein), 1)

    # Feature 0: length change
    feat_0 = float(len(var_protein) != len(ref_protein))

    # Feature 2: methionine start lost
    feat_2 = float(
        len(ref_protein) > 0
        and ref_protein[0] == "M"
        and (len(var_protein) == 0 or var_protein[0] != "M")
    )

    # Feature 5: 1 - sequence identity
    feat_5 = 1.0 - _sequence_identity(ref_protein, var_protein)

    # Feature 11: protein length ratio
    feat_11 = len(var_protein) / max(len(ref_protein), 1)

    # Mutation density
    density = (n_total / protein_len) * 100

    # Affected region — fraction of mutations in each region (continuous)
    region_n = 0.0
    region_c = 0.0
    if mutations:
        pos_mutations = [m for m in mutations if "position" in m]
        if pos_mutations:
            n_pos = len(pos_mutations)
            region_n = sum(
                1.0 for m in pos_mutations
                if _classify_region(m["position"], protein_len) == "N-terminal"
            ) / n_pos
            region_c = sum(
                1.0 for m in pos_mutations
                if _classify_region(m["position"], protein_len) == "C-terminal"
            ) / n_pos

    features = torch.tensor(
        [
            feat_0,                                            # 0
            float(has_premature_stop(ref_protein, var_protein)),  # 1
            feat_2,                                            # 2
            n_missense / max(protein_len, 1),                  # 3
            float(n_nonsense),                                 # 4
            feat_5,                                            # 5
            compute_truncation_fraction(ref_protein, var_protein),  # 6
            density,                                           # 7
            region_n,                                          # 8
            region_c,                                          # 9
            math.log1p(n_total),                               # 10
            feat_11,                                           # 11
        ],
        dtype=torch.float32,
    )

    return features
