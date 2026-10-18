"""Deterministic LoF table splits (by species)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from plmlof.constants import TASKS

logger = logging.getLogger(__name__)


def parse_tasks(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(TASKS)
    out: list[str] = []
    for item in raw:
        for part in str(item).split(","):
            t = part.strip()
            if t and t not in out:
                out.append(t)
    unknown = [t for t in out if t not in TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Choose from {TASKS}.")
    return out


def split_species(species: Sequence[str], seed: int = 42) -> dict[str, str]:
    rng = np.random.RandomState(seed)
    uniq = sorted({s for s in species if s})
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, round(n * 0.15)) if n else 0
    n_val = max(1, round(n * 0.10)) if n else 0
    test = set(uniq[:n_test])
    val = set(uniq[n_test:n_test + n_val])
    mapping = {s: ("test" if s in test else "val" if s in val else "train") for s in uniq}
    logger.info("Species split train=%s val=%s test=%s", n - n_test - n_val, n_val, n_test)
    return mapping
