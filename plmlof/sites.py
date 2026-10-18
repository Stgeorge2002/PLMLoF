"""Same-length missense site helpers used when building LoF tables."""

from __future__ import annotations


def missense_sites(ref: str, var: str) -> list[int]:
    """0-based indices of same-length substitutions. Empty if unaligned or identity."""
    if not ref or not var or len(ref) != len(var):
        return []
    return [i for i, (a, b) in enumerate(zip(ref, var)) if a != b]


def aligned_site_index(ref: str, var: str) -> int:
    """0-based index of the first substitution, or 0 for identity, or -1 if unaligned."""
    sites = missense_sites(ref, var)
    if sites:
        return sites[0]
    if not ref or not var:
        return -1
    if len(ref) != len(var):
        return -1
    return 0


def aa_hamming(ref: str, var: str) -> int:
    return len(missense_sites(ref, var))
