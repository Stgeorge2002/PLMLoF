"""Rule-based clear LoF (wreck) detection.

Stops, frameshifts, large indels, and start-codon loss are decided here,
not by ESM2. Missense is never a wreck.

HMMER does not invent the class. It grades the wreck: before/inside a
domain is sure (~1.0); strictly after the last domain is a tail (~0.4).
Without domain coordinates, the last 10% of the ORF is the tail proxy.
"""

from __future__ import annotations

from collections.abc import Sequence

from plmlof.domains import PRIOR_SURE, PRIOR_TAIL_STOP, SURE_MIN, in_c_terminal_tail
from plmlof.utils.sequence_utils import compute_truncation_fraction

TRUNCATION_FRACTION = 0.40
LENGTH_RATIO_DELETE = 0.70
LENGTH_RATIO_INSERT = 1.30
MIN_INSERT_AA = 20


def first_affected_residue(
    ref_protein: str,
    var_protein: str,
    ref_dna: str = "",
    var_dna: str = "",
) -> int:
    """1-based index of the first residue that is not WT."""
    ref = (ref_protein or "").replace("*", "")
    var = (var_protein or "").replace("*", "")
    if ref_dna and var_dna:
        n = min(len(ref_dna), len(var_dna))
        i = 0
        while i < n and ref_dna[i] == var_dna[i]:
            i += 1
        return i // 3 + 1
    if not ref:
        return 1
    if not var:
        return 1
    n = min(len(ref), len(var))
    for i in range(n):
        if ref[i] != var[i]:
            return i + 1
    if len(var) != len(ref):
        return n + 1
    return 1


def wreck_grade(
    ref_protein: str,
    var_protein: str,
    ref_dna: str = "",
    var_dna: str = "",
    domains: Sequence[tuple[int, int]] = (),
) -> tuple[bool, str, float | None]:
    """Return (is_structural, kind, rule_prior).

    rule_prior is 1.0 for a sure wreck, ~0.4 for a post-domain tail wreck,
    or None when the network should score the allele (missense / WT).
    """
    ref = (ref_protein or "").replace("*", "")
    var = (var_protein or "").replace("*", "")
    if not ref:
        return True, "empty", PRIOR_SURE
    if not var:
        return True, "truncation", PRIOR_SURE

    pos = first_affected_residue(ref, var, ref_dna, var_dna)
    tail = in_c_terminal_tail(pos, len(ref), domains)

    if ref_dna and var_dna and abs(len(var_dna) - len(ref_dna)) % 3 != 0:
        if tail:
            return True, "tail_frameshift", PRIOR_TAIL_STOP
        return True, "frameshift", PRIOR_SURE

    prefix_trunc = bool(var) and ref.startswith(var) and len(var) < len(ref)
    trunc = compute_truncation_fraction(ref, var)
    if prefix_trunc:
        if tail:
            return True, "tail_stop", PRIOR_TAIL_STOP
        if domains and not tail:
            return True, "truncation", PRIOR_SURE
        if trunc >= TRUNCATION_FRACTION:
            return True, "truncation", PRIOR_SURE

    ratio = len(var) / max(len(ref), 1)
    if ratio < LENGTH_RATIO_DELETE:
        if tail:
            return True, "tail_deletion", PRIOR_TAIL_STOP
        return True, "deletion", PRIOR_SURE
    if ratio > LENGTH_RATIO_INSERT and (len(var) - len(ref)) >= MIN_INSERT_AA:
        if tail:
            return True, "tail_insertion", PRIOR_TAIL_STOP
        return True, "insertion", PRIOR_SURE

    if ref[0] == "M" and var[0] != "M" and abs(len(var) - len(ref)) <= 2:
        return True, "start_lost", PRIOR_SURE

    return False, "none", None


def wreck_call(
    ref_protein: str,
    var_protein: str,
    ref_dna: str = "",
    var_dna: str = "",
    domains: Sequence[tuple[int, int]] = (),
) -> tuple[bool, str]:
    """Return (is_sure_wreck, kind). Tail wrecks are not sure."""
    _, kind, prior = wreck_grade(ref_protein, var_protein, ref_dna, var_dna, domains)
    sure = prior is not None and prior >= SURE_MIN
    return sure, kind


def is_wreck(
    ref_protein: str,
    var_protein: str,
    ref_dna: str = "",
    var_dna: str = "",
    domains: Sequence[tuple[int, int]] = (),
) -> bool:
    flagged, _ = wreck_call(ref_protein, var_protein, ref_dna, var_dna, domains)
    return flagged
