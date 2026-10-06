"""Substitution chemistry at a missense site. No MSA, no gene identity."""

from __future__ import annotations

import torch

from plmlof.constants import NUM_SITE_CHEM

# BLOSUM62, rows/cols ACDEFGHIKLMNPQRSTVWY
_AA = "ACDEFGHIKLMNPQRSTVWY"
_BLOSUM62 = (
    ( 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2),
    ( 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2),
    (-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3),
    (-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2),
    (-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3),
    ( 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3),
    (-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2),
    (-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1),
    (-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2),
    (-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1),
    (-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1),
    (-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2),
    (-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3),
    (-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1),
    (-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2),
    ( 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2),
    ( 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2),
    ( 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1),
    (-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2),
    (-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7),
)
_INDEX = {aa: i for i, aa in enumerate(_AA)}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
_HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
_VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1,
    "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1,
    "P": 112.7, "Q": 143.8, "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0,
    "W": 227.8, "Y": 193.6,
}


def _blosum(a: str, b: str) -> float:
    i, j = _INDEX.get(a), _INDEX.get(b)
    if i is None or j is None:
        return 0.0
    return float(_BLOSUM62[i][j])


def substitution_features(ref_aa: str, var_aa: str) -> torch.Tensor:
    """Five scalars: BLOSUM62, identity, charge Δ, hydrophobicity Δ, volume Δ."""
    ra = (ref_aa or "X")[:1].upper()
    va = (var_aa or "X")[:1].upper()
    if ra not in _INDEX or va not in _INDEX:
        return torch.zeros(NUM_SITE_CHEM, dtype=torch.float32)
    return torch.tensor(
        [
            _blosum(ra, va) / 11.0,
            float(ra == va),
            (_CHARGE.get(va, 0.0) - _CHARGE.get(ra, 0.0)) / 2.0,
            (_HYDRO.get(va, 0.0) - _HYDRO.get(ra, 0.0)) / 9.0,
            (_VOLUME.get(va, 0.0) - _VOLUME.get(ra, 0.0)) / 200.0,
        ],
        dtype=torch.float32,
    )


def site_chem_at(ref: str, var: str, center: int) -> torch.Tensor:
    if center < 0 or not ref or not var or center >= len(ref) or center >= len(var):
        return torch.zeros(NUM_SITE_CHEM, dtype=torch.float32)
    return substitution_features(ref[center], var[center])
