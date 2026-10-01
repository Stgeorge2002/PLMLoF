"""PLMLoF v2: graded LoF score plus two conservative GoF flags."""

from __future__ import annotations

TASKS = ("lof", "growth_gof", "amr_gof")

LOF_WRECK = 1.00
LOF_STRONG = 0.70
LOF_WEAK = 0.40
LOF_WT = 0.00

GOF_CALL_THRESHOLD = 0.90
IN_FAMILY_COSINE = 0.75
