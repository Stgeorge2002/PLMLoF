"""Deterministic table splits: proteins (LoF), sites (growth GoF), families (AMR)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from plmlof.constants import TASKS
from plmlof.wreck import first_affected_residue

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


def substitution_sites(ref_protein: str, var_protein: str) -> tuple[int, ...]:
    """Sorted 1-based substituted positions.

    Same-length missense alleles that touch the same residues share a key, so
    A117V and A117T cannot sit on opposite sides of a split. Length-changing
    alleles use (first_affected, -delta_len) so each indel is its own site.
    """
    ref = (ref_protein or "").replace("*", "")
    var = (var_protein or "").replace("*", "")
    if not ref and not var:
        return (0,)
    if len(ref) != len(var):
        return (first_affected_residue(ref, var), -abs(len(ref) - len(var)))
    sites = tuple(i + 1 for i, (a, b) in enumerate(zip(ref, var)) if a != b)
    return sites if sites else (0,)


def _site_labels(n_sites: int, train_frac: float, val_frac: float) -> list[str]:
    """Assign shuffled site slots to train/val/test. Caller shuffles first."""
    if n_sites <= 1:
        return ["train"] * n_sites
    if n_sites == 2:
        return ["train", "test"]
    n_test = max(1, int(round(n_sites * (1.0 - train_frac - val_frac))))
    n_val = max(1, int(round(n_sites * val_frac)))
    if n_test + n_val >= n_sites:
        n_test, n_val = 1, 1
    labels = ["test"] * n_test + ["val"] * n_val + ["train"] * (n_sites - n_test - n_val)
    return labels


def split_residues_within_protein(
    protein_ids: Sequence[str],
    ref_proteins: Sequence[str],
    var_proteins: Sequence[str],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> list[str]:
    """Hold out mutation *sites* inside every protein. Every gene stays in train.

    Returns one split label per input row. A site never appears in two splits.
    Proteins with a single unique site are train-only (nothing honest to hold out).
    """
    if not (len(protein_ids) == len(ref_proteins) == len(var_proteins)):
        raise ValueError("protein_ids, ref_proteins, and var_proteins must be aligned")
    rng = np.random.RandomState(seed)
    n = len(protein_ids)
    splits = ["train"] * n
    by_prot: dict[str, list[int]] = {}
    for i, pid in enumerate(protein_ids):
        by_prot.setdefault(str(pid), []).append(i)

    n_train_p = n_val_p = n_test_p = 0
    for prot in sorted(by_prot):
        idxs = by_prot[prot]
        site_of = [substitution_sites(ref_proteins[i], var_proteins[i]) for i in idxs]
        unique_sites = sorted(set(site_of))
        rng.shuffle(unique_sites)
        labels = _site_labels(len(unique_sites), train_frac, val_frac)
        site_split = dict(zip(unique_sites, labels))
        seen: set[str] = set()
        for i, site in zip(idxs, site_of):
            splits[i] = site_split[site]
            seen.add(splits[i])
        n_train_p += int("train" in seen)
        n_val_p += int("val" in seen)
        n_test_p += int("test" in seen)

    logger.info(
        "Residue split rows train=%s val=%s test=%s | proteins with train/val/test=%s/%s/%s",
        splits.count("train"), splits.count("val"), splits.count("test"),
        n_train_p, n_val_p, n_test_p,
    )
    return splits


def split_proteins(protein_ids: Sequence[str], seed: int = 42) -> dict[str, str]:
    """Protein hold-out for LoF missense (new genes at test time)."""
    rng = np.random.RandomState(seed)
    uniq = sorted(set(protein_ids))
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, round(n * 0.23))
    n_val = max(1, round(n * 0.15))
    test = set(uniq[:n_test])
    val = set(uniq[n_test:n_test + n_val])
    train = set(uniq[n_test + n_val:])
    if not train:
        train, val, test = set(uniq[:-2]), set(uniq[-2:-1]), set(uniq[-1:])
    mapping = {p: ("test" if p in test else "val" if p in val else "train") for p in uniq}
    logger.info("Protein split train=%s val=%s test=%s", len(train), len(val), len(test))
    return mapping


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


def split_families(families: Sequence[str], seed: int = 42) -> dict[str, str]:
    rng = np.random.RandomState(seed)
    uniq = sorted(set(families))
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, round(n * 0.20)) if n else 0
    n_val = max(1, round(n * 0.15)) if n else 0
    test = set(uniq[:n_test])
    val = set(uniq[n_test:n_test + n_val])
    mapping = {f: ("test" if f in test else "val" if f in val else "train") for f in uniq}
    logger.info("Family split train=%s val=%s test=%s", n - n_test - n_val, n_val, n_test)
    return mapping
