"""Assemble LoF train/val/test/null tables on a laptop (no GPU).

Reads ``data/processed/synthetic_lof.parquet`` and writes ``data/processed/lof/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from plmlof.constants import LOF_WT
from plmlof.domains import SURE_MIN, protein_id as pid
from plmlof.labels import (
    EXTRA_MISSENSE_TYPES,
    INDOMAIN_MISSENSE_TYPES,
    channel_for_lof,
    lof_score_for_pair,
)
from plmlof.sites import aligned_site_index, missense_sites
from plmlof.splits import parse_tasks, split_species

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
AA = "ACDEFGHIKLMNPQRSTVWY"

COLS = [
    "ref_protein", "var_protein", "ref_dna", "var_dna", "gene", "species", "source",
    "protein_id", "task", "split", "wreck_type", "channel", "target", "sample_weight",
    "lof_score", "dms_score", "dms_zscore", "is_wreck", "is_missense", "label",
    "site_index", "site_index_2", "n_sites", "taxon", "assay",
]


def _std_row(**kwargs) -> dict:
    numeric = {
        "target", "sample_weight", "lof_score", "dms_score", "dms_zscore",
        "is_wreck", "is_missense", "label", "site_index", "site_index_2", "n_sites",
    }
    row = {c: (0 if c in numeric else "") for c in COLS}
    row.update({
        "dms_score": float("nan"),
        "dms_zscore": float("nan"),
        "lof_score": float("nan"),
        "target": 0.0,
        "sample_weight": 1.0,
        "is_wreck": False,
        "is_missense": False,
        "label": 0,
        "site_index": -1,
        "site_index_2": -1,
        "n_sites": 0,
        "taxon": "",
        "assay": "",
    })
    row.update(kwargs)
    ref = str(row.get("ref_protein") or "")
    var = str(row.get("var_protein") or "")
    sites = missense_sites(ref, var)
    if "n_sites" not in kwargs:
        row["n_sites"] = len(sites)
    if "site_index" not in kwargs:
        row["site_index"] = sites[0] if sites else aligned_site_index(ref, var)
    if "site_index_2" not in kwargs:
        row["site_index_2"] = sites[1] if len(sites) > 1 else -1
    return row


def random_missense(ref: str, rng: np.random.RandomState, n: int = 3) -> list[str]:
    ref = (ref or "").replace("*", "")
    if len(ref) < 2:
        return []
    out = []
    for _ in range(n * 8):
        if len(out) >= n:
            break
        i = int(rng.randint(0, len(ref)))
        choices = [a for a in AA if a != ref[i]]
        if not choices:
            continue
        aa = choices[int(rng.randint(0, len(choices)))]
        var = ref[:i] + aa + ref[i + 1:]
        if var != ref:
            out.append(var)
    return out


def cap_synthetic(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(df) <= cap:
        return df
    types = df["wreck_type"].fillna("unknown")
    per = max(1, cap // max(types.nunique(), 1))
    parts = []
    for _kind, sub in df.groupby(types):
        take = min(len(sub), per)
        parts.append(sub.sample(n=take, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > cap:
        out = out.sample(n=cap, random_state=seed)
    logger.info("Synthetic capped %s → %s (by type)", len(df), len(out))
    return out.reset_index(drop=True)


def write_task(df: pd.DataFrame, task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for col in COLS:
        if col not in df.columns:
            if col == "site_index_2":
                df[col] = -1
            elif col in {
                "target", "sample_weight", "lof_score", "label", "is_wreck",
                "is_missense", "site_index", "n_sites",
            }:
                df[col] = 0
            else:
                df[col] = ""
    df = df[COLS]
    for split in ("train", "val", "test", "protein_test"):
        sub = df[df["split"] == split]
        if split == "protein_test" and sub.empty:
            continue
        path = task_dir / f"{split}.parquet"
        sub.to_parquet(path, index=False)
        logger.info(
            "  %s %s rows genes=%s proteins=%s",
            path.name, len(sub), sub["gene"].nunique(), sub["protein_id"].nunique(),
        )
    null = df[df["split"] == "null"]
    if not null.empty:
        null.to_parquet(task_dir / "null.parquet", index=False)
        logger.info("  null.parquet %s rows", len(null))
    counts = df.groupby(["split", "channel"]).size().unstack(fill_value=0)
    logger.info("Channel counts:\n%s", counts.to_string())


def _score_synthetic(rec) -> tuple[str, float, bool, bool, str]:
    wreck_type = str(getattr(rec, "wreck_type", "stop") or "stop")
    raw = getattr(rec, "lof_score", float("nan"))
    stored = float(raw) if raw is not None and pd.notna(raw) else None
    score = lof_score_for_pair(
        rec.ref_protein,
        rec.var_protein,
        wreck_type=wreck_type,
        ref_dna=getattr(rec, "ref_dna", "") or "",
        var_dna=getattr(rec, "var_dna", "") or "",
        lof_score=stored,
    )
    if score is None:
        score = 1.0
    is_missense = wreck_type in INDOMAIN_MISSENSE_TYPES | EXTRA_MISSENSE_TYPES
    is_wreck = (not is_missense) and score >= SURE_MIN
    return wreck_type, score, is_wreck, is_missense, channel_for_lof(score, wreck_type, None)


def build_lof(syn: pd.DataFrame, species_split: dict[str, str], args) -> pd.DataFrame:
    """Comparative wreck head. Ref vs var pairs."""
    syn = cap_synthetic(syn, args.synthetic_cap, args.seed) if not syn.empty else syn
    syn_rows = []
    for rec in syn.itertuples(index=False):
        wreck_type, score, is_wreck, is_missense, ch = _score_synthetic(rec)
        if is_missense:
            score, is_wreck, ch = LOF_WT, False, "missense_neg"
        syn_rows.append(_std_row(
            ref_protein=rec.ref_protein,
            var_protein=rec.var_protein,
            ref_dna=getattr(rec, "ref_dna", "") or "",
            var_dna=getattr(rec, "var_dna", "") or "",
            gene=rec.gene,
            species=rec.species,
            source="synthetic_lof",
            protein_id=pid(rec.ref_protein),
            task="lof",
            split=species_split.get(rec.species, "train"),
            wreck_type=wreck_type,
            channel=ch,
            target=score,
            lof_score=score,
            is_wreck=is_wreck,
            is_missense=is_missense,
            sample_weight=1.0,
            label=0 if score >= 0.4 else 1,
        ))
    syn_df = pd.DataFrame(syn_rows) if syn_rows else pd.DataFrame(columns=COLS)

    id_rows = []
    seen: set[str] = set()
    rng = np.random.RandomState(args.seed)
    for rec in syn_df.itertuples(index=False):
        key = rec.protein_id
        if key in seen:
            continue
        seen.add(key)
        split = rec.split
        id_rows.append(_std_row(
            ref_protein=rec.ref_protein,
            var_protein=rec.ref_protein,
            gene=rec.gene,
            species=rec.species,
            source="synthetic_identity",
            protein_id=key,
            task="lof",
            split=split,
            wreck_type="identity",
            channel="wt",
            target=LOF_WT,
            lof_score=LOF_WT,
            is_wreck=False,
            is_missense=False,
            sample_weight=1.0,
            label=1,
        ))
        for var in random_missense(str(rec.ref_protein), rng, n=1):
            id_rows.append(_std_row(
                ref_protein=rec.ref_protein,
                var_protein=var,
                gene=rec.gene,
                species=rec.species,
                source="random_missense",
                protein_id=key,
                task="lof",
                split=split,
                wreck_type="none",
                channel="missense_neg",
                target=LOF_WT,
                lof_score=LOF_WT,
                is_wreck=False,
                is_missense=True,
                sample_weight=1.0,
                label=1,
            ))
    id_df = pd.DataFrame(id_rows) if id_rows else pd.DataFrame(columns=COLS)
    lof = (
        pd.concat([syn_df, id_df], ignore_index=True)
        if not syn_df.empty or not id_df.empty
        else pd.DataFrame(columns=COLS)
    )
    n_wreck = int(lof["is_wreck"].sum()) if not lof.empty else 0
    logger.info("LoF (comparative) rows=%s wrecks=%s", len(lof), n_wreck)

    if not lof.empty:
        negs = lof[lof["target"] <= 0.05]
        if not negs.empty:
            lof = pd.concat([lof, negs.assign(split="null")], ignore_index=True)
    return lof


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Build PLMLoF LoF training tables")
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-cap", type=int, default=80_000)
    p.add_argument("--missense-boost", type=float, default=4.0, help="Unused; kept for CLI compat")
    p.add_argument("--tasks", nargs="+", default=None, help="Must be lof (default).")
    args = p.parse_args()
    out = args.out or args.processed
    try:
        want = parse_tasks(args.tasks)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if want != ["lof"]:
        raise SystemExit(f"Only task 'lof' is supported, got {want}")

    syn_path = args.processed / "synthetic_lof.parquet"
    if syn_path.exists():
        syn = pd.read_parquet(syn_path)
        logger.info("Synthetic %s rows", len(syn))
    else:
        logger.warning("No synthetic_lof.parquet — LoF wreck channel will be empty")
        syn = pd.DataFrame()

    species_split = split_species(syn["species"].tolist(), args.seed) if not syn.empty else {}

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["synthetic_rows_used"] = int(len(syn))
    manifest["tasks_written"] = want

    logger.info("── LoF (comparative synthetic wrecks) ──")
    lof = build_lof(syn, species_split, args)
    write_task(lof, out / "lof")
    manifest["lof_rows"] = int(len(lof))

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", manifest_path)
    logger.info("Done. rsync data/processed/lof to Isambard (parquet is gitignored).")


if __name__ == "__main__":
    main()
