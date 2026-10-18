"""PLMLoF: comparative bacterial LoF score (ref↔var ESM2)."""

from __future__ import annotations

from plmlof.constants import (
    LOF_STRONG,
    LOF_WEAK,
    LOF_WRECK,
    LOF_WT,
    REGRESSION_TASKS,
    TASKS,
    TRAIN_TASKS,
)
from plmlof.encoders import ESM2_LOF, esm2_for_task

__version__ = "1.0.0"

__all__ = [
    "ESM2_LOF",
    "LOF_STRONG",
    "LOF_WEAK",
    "LOF_WRECK",
    "LOF_WT",
    "REGRESSION_TASKS",
    "TASKS",
    "TRAIN_TASKS",
    "esm2_for_task",
]
