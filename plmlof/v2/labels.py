"""Deterministic LoF score assignment from assay z-scores and wreck tags."""

from __future__ import annotations

from collections.abc import Sequence

from plmlof.v2 import LOF_STRONG, LOF_WEAK, LOF_WRECK, LOF_WT
from plmlof.v2.domains import (
    PRIOR_DOMAIN_DROP,
    PRIOR_EXTRA_MISSENSE,
    PRIOR_INDOMAIN_MISSENSE,
    PRIOR_TAIL_STOP,
    PRIOR_TAIL_WEAK,
    SURE_MIN,
    lof_prior,
)
from plmlof.v2.wreck import first_affected_residue, wreck_call, wreck_grade

Z_STRONG = -2.0
Z_WEAK = -1.0
Z_GOF_LOOSE = 1.0
Z_GOF_STRICT = 2.0
Z_WT = 1.0

SURE_WRECK_TYPES = {
    "stop", "frameshift", "deletion", "insertion", "truncation", "empty",
    "start_lost", "stop_pre_domain", "stop_in_domain",
    "frameshift_pre_domain", "frameshift_in_domain", "indel_in_domain",
}
TAIL_WRECK_TYPES = {
    "stop_tail", "frameshift_tail", "indel_tail", "tail_stop",
    "tail_frameshift", "tail_deletion", "tail_insertion",
}
DOMAIN_DROP_TYPES = {"domain_drop"}
INDOMAIN_MISSENSE_TYPES = {"missense_in_domain", "heavy_missense"}
EXTRA_MISSENSE_TYPES = {"missense_extra_domain"}


def lof_score_from_z(z: float | None, *, is_wreck: bool = False) -> float | None:
    """Map a growth-DMS z-score (and optional wreck flag) to lof_score.

    Returns None when the row is an upper-tail GoF (z > +1) and must not
    enter the LoF table.
    """
    if is_wreck:
        return LOF_WRECK
    if z is None:
        return None
    if z <= Z_STRONG:
        return LOF_STRONG
    if z < Z_WEAK:
        return LOF_WEAK
    if abs(z) <= Z_WT:
        return LOF_WT
    if z > Z_GOF_LOOSE:
        return None
    return LOF_WT


def lof_score_for_wreck_type(wreck_type: str | None) -> float | None:
    if not wreck_type or wreck_type in {"", "none", "identity"}:
        return None
    if wreck_type in TAIL_WRECK_TYPES:
        return PRIOR_TAIL_STOP if "stop" in wreck_type or "frameshift" in wreck_type else PRIOR_TAIL_WEAK
    if wreck_type in DOMAIN_DROP_TYPES:
        return PRIOR_DOMAIN_DROP
    if wreck_type in INDOMAIN_MISSENSE_TYPES:
        return PRIOR_INDOMAIN_MISSENSE
    if wreck_type in EXTRA_MISSENSE_TYPES:
        return PRIOR_EXTRA_MISSENSE
    if wreck_type in SURE_WRECK_TYPES:
        return LOF_WRECK
    return None


def lof_score_for_pair(
    ref_protein: str,
    var_protein: str,
    z: float | None = None,
    wreck_type: str | None = None,
    ref_dna: str = "",
    var_dna: str = "",
    domains: Sequence[tuple[int, int]] = (),
    lof_score: float | None = None,
) -> float | None:
    """Score a pair. Generator lof_score wins; else domain geometry; else z."""
    if lof_score is not None and lof_score == lof_score:  # not NaN
        return float(lof_score)
    typed = lof_score_for_wreck_type(wreck_type)
    if typed is not None and wreck_type not in SURE_WRECK_TYPES | TAIL_WRECK_TYPES | DOMAIN_DROP_TYPES:
        return typed
    if wreck_type in (SURE_WRECK_TYPES | TAIL_WRECK_TYPES | DOMAIN_DROP_TYPES) and domains:
        event = "stop"
        if "frameshift" in (wreck_type or ""):
            event = "frameshift"
        elif wreck_type in DOMAIN_DROP_TYPES:
            event = "domain_drop"
        elif wreck_type in {"deletion", "insertion", "indel_in_domain", "indel_tail"}:
            event = "indel"
        pos = first_affected_residue(ref_protein, var_protein, ref_dna, var_dna)
        score, _ = lof_prior(event, pos, len((ref_protein or "").replace("*", "")), domains)
        return score
    if typed is not None:
        return typed
    _, _, prior = wreck_grade(ref_protein, var_protein, ref_dna, var_dna, domains)
    if prior is not None:
        return prior
    flagged, _ = wreck_call(ref_protein, var_protein, ref_dna, var_dna, domains)
    if flagged:
        return LOF_WRECK
    return lof_score_from_z(z, is_wreck=False)


def channel_for_lof(score: float, wreck_type: str | None, z: float | None) -> str:
    if wreck_type in EXTRA_MISSENSE_TYPES:
        return "wt"
    if wreck_type in INDOMAIN_MISSENSE_TYPES:
        return "strong_missense" if score >= 0.55 else "weak_missense"
    if wreck_type in DOMAIN_DROP_TYPES:
        return "domain_drop"
    if wreck_type in TAIL_WRECK_TYPES or (score < SURE_MIN and wreck_type in SURE_WRECK_TYPES | TAIL_WRECK_TYPES):
        return "tail_wreck"
    if score >= 0.99 or wreck_type in SURE_WRECK_TYPES:
        return "clear_wreck"
    if z is not None and z <= Z_STRONG:
        return "strong_missense"
    if z is not None and z < Z_WEAK:
        return "weak_missense"
    return "wt"
