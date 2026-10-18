"""LoF score constants."""

from __future__ import annotations

TASKS = ("lof",)
TRAIN_TASKS = ("lof",)
REGRESSION_TASKS = frozenset({"lof"})

LOF_WRECK = 1.00
LOF_STRONG = 0.70
LOF_WEAK = 0.40
LOF_WT = 0.00

IN_FAMILY_COSINE = 0.75
