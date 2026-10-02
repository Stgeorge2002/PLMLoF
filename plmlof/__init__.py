"""PLMLoF: alignment-free LoF, missense-damage MLoF, conservative GoF flags."""

from __future__ import annotations

from plmlof.constants import (
    GOF_CALL_THRESHOLD,
    IN_FAMILY_COSINE,
    LOF_STRONG,
    LOF_WEAK,
    LOF_WRECK,
    LOF_WT,
    REGRESSION_TASKS,
    TASKS,
)
from plmlof.encoders import ESM2_LOF, ESM2_PAIR, esm2_for_task

__version__ = "1.0.0"

__all__ = [
    "ESM2_LOF",
    "ESM2_PAIR",
    "GOF_CALL_THRESHOLD",
    "IN_FAMILY_COSINE",
    "LOF_STRONG",
    "LOF_WEAK",
    "LOF_WRECK",
    "LOF_WT",
    "REGRESSION_TASKS",
    "TASKS",
    "esm2_for_task",
]
