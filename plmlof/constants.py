"""Task names and score constants."""

from __future__ import annotations

TASKS = ("lof", "mlof", "growth_gof", "amr_gof")
# GoF heads stay in the tree but are not trained or evaluated.
# Restore: TRAIN_TASKS = TASKS
TRAIN_TASKS = ("lof", "mlof")
# TRAIN_TASKS = ("lof", "mlof", "growth_gof", "amr_gof")
REGRESSION_TASKS = frozenset({"lof", "mlof"})

LOF_WRECK = 1.00
LOF_STRONG = 0.70
LOF_WEAK = 0.40
LOF_WT = 0.00

GOF_CALL_THRESHOLD = 0.90
IN_FAMILY_COSINE = 0.75

# MDG site window: centre ± 2. Odd. Cached embeddings store [N, SITE_WINDOW, D].
SITE_WINDOW = 5
SITE_RADIUS = SITE_WINDOW // 2
# Train/eval Hamming-1 plus Hamming-2 (predict already maxes every same-length site).
MLOF_MAX_HAMMING = 2
# BLOSUM / same / charge / hydrophobicity / volume.
NUM_SUB_CHEM = 5
# + unmasked ESM2 log-odds + Pfam in-domain + extra-domain.
NUM_SITE_CHEM = 8
CHEM_LLR = 5
CHEM_IN_DOMAIN = 6
CHEM_EXTRA_DOMAIN = 7
LLR_SCALE = 10.0
