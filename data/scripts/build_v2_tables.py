"""Assemble train/val/test/null tables on a laptop (no GPU).

Reads:
  data/processed/proteingym_substitutions.parquet  (MLoF: all taxa / assays)
  data/processed/proteingym_bacterial.parquet      (fallback + GoF OF match)
  data/raw/proteingym/DMS_substitutions.csv
  data/processed/gof_growth_amr.parquet      (optional CARD source)
  data/processed/synthetic_lof.parquet

Writes under data/processed/{lof,mlof,growth_gof,amr_gof}/
  train.parquet val.parquet test.parquet [protein_test.parquet] null.parquet

MLoF is a missense-damage ranker on every ProteinGym substitution gene.
Wreck LoF stays synthetic. Growth/AMR GoF stay prokaryote OrganismalFitness.
Splits: LoF synthetics by species; MLoF nested (residue-within-protein plus
held-out proteins); growth and AMR GoF by residue-within-protein.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from plmlof.constants import LOF_WT
from plmlof.domains import SURE_MIN, protein_id as pid
from plmlof.labels import (
    EXTRA_MISSENSE_TYPES,
    INDOMAIN_MISSENSE_TYPES,
    Z_GOF_LOOSE,
    Z_GOF_STRICT,
    Z_STRONG,
    Z_WEAK,
    Z_WT,
    channel_for_lof,
    damage_from_z,
    lof_score_for_pair,
    lof_score_from_z,
)
from plmlof.sites import aligned_site_index
from plmlof.splits import parse_tasks, split_mlof_nested, split_residues_within_protein, split_species

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
RAW = REPO / "data" / "raw"
AA = "ACDEFGHIKLMNPQRSTVWY"
SNP_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

AMR_TOKENS = (
    "antibiotic", "antibiotics", "amp ", "mic", "resistance", "kanamycin",
    "beta-lactam", "betalactam", "carbapenem", "ceftaz", "cipro",
)
GROWTH_TOKENS = ("growth", "fitness", "toxin", "dilution")

AMR_FAMILIES = (
    "gyra", "gyrb", "parc", "pare", "rpob", "rpoc", "pmra", "pmrb", "phop", "phoq",
    "ompk", "ompc", "ompf", "bla", "tem", "shv", "kpc", "ndm", "vim", "imp", "oxa",
    "ctx", "dhfr", "dyr", "folp", "katg", "inha", "pnca", "rpsl", "erm", "mef",
    "tet", "van", "pbp", "ftsi", "kka2", "aph", "aac", "blat",
)

MIC_GENE_HINTS = ("blat", "tem", "kka2", "vim", "ampc", "dyr", "dhfr")

COLS = [
    "ref_protein", "var_protein", "ref_dna", "var_dna", "gene", "species", "source",
    "protein_id", "task", "split", "wreck_type", "channel", "target", "sample_weight",
    "lof_score", "dms_score", "dms_zscore", "is_wreck", "is_missense", "label",
    "site_index",
]


def _std_row(**kwargs) -> dict:
    numeric = {
        "target", "sample_weight", "lof_score", "dms_score", "dms_zscore",
        "is_wreck", "is_missense", "label", "site_index",
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
    })
    row.update(kwargs)
    if "site_index" not in kwargs:
        ref = str(row.get("ref_protein") or "")
        var = str(row.get("var_protein") or "")
        row["site_index"] = aligned_site_index(ref, var)
    return row


def load_of_meta(ref_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ref_csv)
    mask = df["taxon"].astype(str).str.strip().str.lower().eq("prokaryote")
    mask &= df["coarse_selection_type"].astype(str).str.strip().eq("OrganismalFitness")
    of = df.loc[mask].copy()
    text = (
        of.get("selection_type", pd.Series([""] * len(of))).astype(str) + " " +
        of.get("selection_assay", pd.Series([""] * len(of))).astype(str)
    ).str.lower()
    of["is_amr_assay"] = text.apply(lambda t: any(tok in t for tok in AMR_TOKENS))
    of["is_growth_assay"] = text.apply(
        lambda t: (any(tok in t for tok in GROWTH_TOKENS) or not t.strip()) and not any(tok in t for tok in AMR_TOKENS)
    )
    # Assays with empty text still count as OF; AMR token wins.
    of.loc[of["is_amr_assay"], "is_growth_assay"] = False
    of.loc[~of["is_amr_assay"] & ~of["is_growth_assay"], "is_growth_assay"] = True
    logger.info("OF assays=%s AMR-like=%s growth-like=%s", len(of), int(of["is_amr_assay"].sum()), int(of["is_growth_assay"].sum()))
    return of


def of_keys(of: pd.DataFrame) -> tuple[set[str], dict[str, bool]]:
    keys: set[str] = set()
    amr_by_key: dict[str, bool] = {}
    for _, r in of.iterrows():
        ids = [r.get("DMS_id"), r.get("DMS_filename")]
        if isinstance(r.get("DMS_filename"), str):
            ids.append(Path(str(r["DMS_filename"])).stem)
        is_amr = bool(r.get("is_amr_assay"))
        for k in ids:
            if k is None or (isinstance(k, float) and math.isnan(k)):
                continue
            s = str(k)
            keys.add(s)
            keys.add(Path(s).stem)
            amr_by_key[s] = is_amr
            amr_by_key[Path(s).stem] = is_amr
    return keys, amr_by_key


def load_of_rows(pg: pd.DataFrame, keys: set[str], amr_by_key: dict[str, bool]) -> pd.DataFrame:
    gene = pg["gene"].astype(str)
    stem = gene.map(lambda g: Path(g).stem)
    keep = gene.isin(keys) | stem.isin(keys)
    of = pg.loc[keep].copy()
    of["gene_stem"] = stem
    of["protein_id"] = of["ref_protein"].map(pid)
    of["is_amr_assay"] = of["gene_stem"].map(lambda s: bool(amr_by_key.get(s, False)))
    # Gene-name fallback for TEM/KKA2/VIM if metadata missed them.
    hint = of["gene"].astype(str).str.lower() + of.get("species", "").astype(str).str.lower()
    of.loc[hint.apply(lambda s: any(h in s for h in MIC_GENE_HINTS)), "is_amr_assay"] = True
    logger.info("OF variant rows=%s proteins=%s", len(of), of["protein_id"].nunique())
    return of


MLOF_MAX_LEN = 1024
SUBS_PARQUET = "proteingym_substitutions.parquet"
BACT_PARQUET = "proteingym_bacterial.parquet"


def _single_aa_missense(ref: str, var: str) -> bool:
    """True iff same length and exactly one amino-acid substitution."""
    if not ref or not var or len(ref) != len(var) or ref == var:
        return False
    n = 0
    for a, b in zip(ref, var):
        if a != b:
            n += 1
            if n > 1:
                return False
    return n == 1


def _stratify_mlof_per_protein(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep every gene; cap variants per WT sequence, stratified by z-bin."""
    if cap <= 0 or df.empty:
        return df
    rng = np.random.RandomState(seed)
    parts: list[pd.DataFrame] = []
    n_capped = 0
    for _, sub in df.groupby("protein_id", sort=False):
        sub = sub.reset_index(drop=True)
        if len(sub) <= cap:
            parts.append(sub)
            continue
        n_capped += 1
        z = sub["dms_zscore"].to_numpy(dtype=float)
        masks = (
            z <= Z_STRONG,
            (z > Z_STRONG) & (z < Z_WEAK),
            np.abs(z) <= Z_WT,
            z > Z_GOF_LOOSE,
        )
        quota = max(1, cap // 4)
        picked: list[np.ndarray] = []
        leftover: list[np.ndarray] = []
        for mask in masks:
            idx = np.flatnonzero(mask)
            rng.shuffle(idx)
            n = min(len(idx), quota)
            if n:
                picked.append(idx[:n])
            if len(idx) > n:
                leftover.append(idx[n:])
        have = np.concatenate(picked) if picked else np.array([], dtype=int)
        need = cap - len(have)
        if need > 0 and leftover:
            rest = np.concatenate(leftover)
            rng.shuffle(rest)
            have = np.concatenate([have, rest[:need]])
        parts.append(sub.iloc[have[:cap]])
    out = pd.concat(parts, ignore_index=True)
    logger.info(
        "MLoF per-protein cap=%s: %s → %s rows (%s proteins truncated)",
        cap, len(df), len(out), n_capped,
    )
    return out


def load_mlof_dms(pg: pd.DataFrame) -> pd.DataFrame:
    """All ProteinGym substitution genes, single-site missense, mean z per pair."""
    dms = pg.copy()
    if "protein_id" not in dms.columns:
        dms["protein_id"] = dms["ref_protein"].map(pid)
    refs = dms["ref_protein"].astype(str)
    vars_ = dms["var_protein"].astype(str)
    too_long = (refs.str.len() > MLOF_MAX_LEN) | (vars_.str.len() > MLOF_MAX_LEN)
    if too_long.any():
        logger.info("MLoF drop %s rows with length > %s", int(too_long.sum()), MLOF_MAX_LEN)
        dms = dms.loc[~too_long].copy()
        refs = dms["ref_protein"].astype(str)
        vars_ = dms["var_protein"].astype(str)
    z = pd.to_numeric(dms["dms_zscore"], errors="coerce")
    keep = z.to_numpy()
    miss = [_single_aa_missense(a, b) for a, b in zip(refs.tolist(), vars_.tolist())]
    dms = dms.loc[np.asarray(miss) & np.isfinite(keep)].copy()
    dms["dms_zscore"] = pd.to_numeric(dms["dms_zscore"], errors="coerce")
    agg: dict[str, str] = {
        "dms_zscore": "mean",
        "gene": "first",
        "species": "first",
        "protein_id": "first",
        "source": "first",
        "dms_score": "mean",
    }
    if "taxon" in dms.columns:
        agg["taxon"] = "first"
    if "coarse_selection_type" in dms.columns:
        agg["coarse_selection_type"] = "first"
    grouped = (
        dms.groupby(["ref_protein", "var_protein"], sort=False)
        .agg(agg)
        .reset_index()
    )
    logger.info(
        "MLoF DMS singles=%s proteins=%s assays_as_gene=%s",
        len(grouped), grouped["protein_id"].nunique(), grouped["gene"].nunique(),
    )
    return grouped


def family_of(gene: str) -> str:
    g = gene.lower()
    for fam in AMR_FAMILIES:
        if fam in g:
            return fam
    token = re.split(r"[|;_/\s]+", gene)[0].lower()
    return token[:12] if token else "unknown"


def random_missense(ref: str, rng: np.random.RandomState, n: int = 3) -> list[str]:
    out = []
    if len(ref) < 5:
        return out
    tried = set()
    while len(out) < n and len(tried) < n * 20:
        i = int(rng.randint(1, len(ref) - 1))  # keep start Met
        choices = [a for a in AA if a != ref[i]]
        aa = choices[int(rng.randint(0, len(choices)))]
        var = ref[:i] + aa + ref[i + 1:]
        if var in tried:
            continue
        tried.add(var)
        out.append(var)
    return out


def cap_synthetic(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(df) <= cap:
        return df
    rng = np.random.RandomState(seed)
    parts = []
    types = df["wreck_type"].fillna("unknown")
    per = max(1, cap // max(types.nunique(), 1))
    for kind, sub in df.groupby(types):
        take = min(len(sub), per)
        parts.append(sub.sample(n=take, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > cap:
        out = out.sample(n=cap, random_state=seed)
    logger.info("Synthetic capped %s → %s (by type)", len(df), len(out))
    return out.reset_index(drop=True)


def balance_gof(pos: pd.DataFrame, neg: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    n_pos = len(pos)
    n_neg = min(len(neg), max(1, int(n_pos * ratio)))
    if n_neg < len(neg):
        neg = neg.sample(n=n_neg, random_state=seed)
    logger.info("GoF pos=%s neg=%s (ratio target %.1f)", n_pos, len(neg), ratio)
    return pd.concat([pos, neg], ignore_index=True)


def write_task(df: pd.DataFrame, task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for col in COLS:
        if col not in df.columns:
            df[col] = 0 if col in {"target", "sample_weight", "lof_score", "label", "is_wreck", "is_missense", "site_index"} else ""
    df = df[COLS]
    for split in ("train", "val", "test", "protein_test"):
        sub = df[df["split"] == split]
        if split == "protein_test" and sub.empty:
            continue
        path = task_dir / f"{split}.parquet"
        sub.to_parquet(path, index=False)
        logger.info("  %s %s rows genes=%s proteins=%s", path.name, len(sub), sub["gene"].nunique(), sub["protein_id"].nunique())
    null = df[df["split"] == "null"]
    if not null.empty:
        null.to_parquet(task_dir / "null.parquet", index=False)
        logger.info("  null.parquet %s rows", len(null))
    counts = df.groupby(["split", "channel"]).size().unstack(fill_value=0)
    logger.info("Channel counts:\n%s", counts.to_string())


def missense_weight_scale(n_missense: int, n_wreck: int, boost: float) -> tuple[float, float]:
    """Return (wreck_w, missense_w) so total missense weight ≈ total wreck weight × boost/1."""
    if n_wreck == 0:
        return 1.0, boost
    if n_missense == 0:
        return 1.0, boost
    # n_m * m_w  ≈  n_w * w_w ; set w_w = 1, m_w = (n_w / n_m) * boost
    return 1.0, (n_wreck / max(n_missense, 1)) * boost


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
    """Alignment-free wreck head. One protein sequence; no OF missense."""
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
    lof = pd.concat([syn_df, id_df], ignore_index=True) if not syn_df.empty or not id_df.empty else pd.DataFrame(columns=COLS)
    n_wreck = int(lof["is_wreck"].sum()) if not lof.empty else 0
    logger.info("LoF (alignment-free) rows=%s wrecks=%s", len(lof), n_wreck)

    null_bits = []
    if not lof.empty:
        negs = lof[lof["target"] <= 0.05]
        if not negs.empty:
            null_bits.append(negs.assign(split="null"))
    if null_bits:
        lof = pd.concat([lof, *null_bits], ignore_index=True)
    return lof


def build_mlof(dms: pd.DataFrame, args) -> pd.DataFrame:
    """Pairwise missense-damage ranker. Every ProteinGym substitution gene.

    Target is continuous ``damage_from_z(z)``, not 0/0.40/0.70 bins.
    Nested split: residue hold-out on most proteins plus a protein_test set.
    Beneficial tail stays in the table (damage near 0), not dropped.
    """
    dms = _stratify_mlof_per_protein(dms, args.mlof_per_protein, args.seed)
    nested = split_mlof_nested(
        dms["protein_id"].tolist(),
        dms["ref_protein"].tolist(),
        dms["var_protein"].tolist(),
        seed=args.seed,
    ) if not dms.empty else []
    rows = []
    for rec, split in zip(dms.itertuples(index=False), nested):
        z = rec.dms_zscore
        if not np.isfinite(z):
            continue
        score = damage_from_z(float(z))
        ch = channel_for_lof(lof_score_from_z(float(z), is_wreck=False, drop_gain=False) or LOF_WT, None, float(z))
        src = str(getattr(rec, "source", "") or "ProteinGym")
        rows.append(_std_row(
            ref_protein=rec.ref_protein,
            var_protein=rec.var_protein,
            gene=rec.gene,
            species=getattr(rec, "species", "") or "",
            source=src,
            protein_id=rec.protein_id,
            task="mlof",
            split=split,
            wreck_type="none",
            channel=ch,
            target=score,
            lof_score=score,
            dms_score=float(getattr(rec, "dms_score", float("nan"))),
            dms_zscore=float(z),
            is_wreck=False,
            is_missense=True,
            sample_weight=1.0,
            label=0 if score >= 0.4 else 1,
        ))
    of_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=COLS)

    id_rows = []
    seen: set[str] = set()
    held = set(of_df.loc[of_df["split"] == "protein_test", "protein_id"]) if not of_df.empty else set()
    for rec in of_df.itertuples(index=False):
        if rec.protein_id in seen:
            continue
        seen.add(rec.protein_id)
        id_rows.append(_std_row(
            ref_protein=rec.ref_protein,
            var_protein=rec.ref_protein,
            gene=rec.gene,
            species=rec.species,
            source="mlof_identity",
            protein_id=rec.protein_id,
            task="mlof",
            split="protein_test" if rec.protein_id in held else "train",
            wreck_type="identity",
            channel="wt",
            target=LOF_WT,
            lof_score=LOF_WT,
            is_wreck=False,
            is_missense=False,
            sample_weight=1.0,
            label=1,
            site_index=0,
        ))
    id_df = pd.DataFrame(id_rows) if id_rows else pd.DataFrame(columns=COLS)
    mlof = pd.concat([of_df, id_df], ignore_index=True) if not of_df.empty or not id_df.empty else pd.DataFrame(columns=COLS)
    logger.info(
        "MLoF rows=%s missense=%s identities=%s proteins=%s",
        len(mlof), len(of_df), len(id_df), of_df["protein_id"].nunique() if not of_df.empty else 0,
    )

    rng = np.random.RandomState(args.seed)
    null_bits = []
    wt = mlof[mlof["channel"] == "wt"]
    if not wt.empty:
        null_bits.append(wt.assign(split="null"))
    wt_refs = dms[np.abs(dms["dms_zscore"]) <= Z_WT] if not dms.empty else dms
    if not wt_refs.empty:
        for ref in wt_refs["ref_protein"].drop_duplicates().head(200):
            for var in random_missense(str(ref), rng, n=2):
                null_bits.append(pd.DataFrame([_std_row(
                    ref_protein=ref, var_protein=var, gene="random_missense",
                    species="", source="random_missense", protein_id=pid(ref),
                    task="mlof", split="null", wreck_type="none", channel="wt",
                    target=0.0, lof_score=0.0, is_wreck=False, is_missense=True, label=1,
                )]))
    if null_bits:
        mlof = pd.concat([mlof, *null_bits], ignore_index=True)
    return mlof


def build_growth_gof(of: pd.DataFrame, syn: pd.DataFrame, args) -> pd.DataFrame:
    growth = of.loc[~of["is_amr_assay"]].copy()
    of_rows: list[dict] = []
    for rec in growth.itertuples(index=False):
        z = rec.dms_zscore
        if not np.isfinite(z):
            continue
        if z >= Z_GOF_STRICT:
            channel, target, label = "gof_pos", 1.0, 1
        elif abs(z) <= Z_WT:
            channel, target, label = "gof_neg", 0.0, 0
        else:
            continue  # +1 < z < +2 is the no-call band
        of_rows.append(dict(
            ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
            species=getattr(rec, "species", "") or "", source="ProteinGym_OrganismalFitness",
            protein_id=rec.protein_id, task="growth_gof",
            dms_zscore=float(z), dms_score=float(getattr(rec, "dms_score", float("nan"))),
            is_missense=True, is_wreck=False, wreck_type="none",
            channel=channel, target=target, label=label,
        ))

    site_splits = split_residues_within_protein(
        [r["protein_id"] for r in of_rows],
        [r["ref_protein"] for r in of_rows],
        [r["var_protein"] for r in of_rows],
        seed=args.seed,
    ) if of_rows else []

    pos_rows, neg_rows = [], []
    for rec, split in zip(of_rows, site_splits):
        row = _std_row(**rec, split=split, sample_weight=1.0)
        if rec["channel"] == "gof_pos":
            pos_rows.append(row)
        else:
            neg_rows.append(row)

    # Wreck negatives so truncation cannot be GoF
    syn_cap = cap_synthetic(syn, min(args.synthetic_cap, max(len(pos_rows) * 2, 500)), args.seed)
    for rec in syn_cap.itertuples(index=False):
        neg_rows.append(_std_row(
            ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
            species=rec.species, source="synthetic_lof", protein_id=pid(rec.ref_protein),
            task="growth_gof", split="train", wreck_type=getattr(rec, "wreck_type", "stop"),
            channel="gof_neg", target=0.0, is_wreck=True, is_missense=False, label=0,
        ))

    pos = pd.DataFrame(pos_rows) if pos_rows else pd.DataFrame(columns=COLS)
    of_neg_rows = [r for r in neg_rows if r.get("source") == "ProteinGym_OrganismalFitness"]
    wreck_rows = [r for r in neg_rows if r.get("source") != "ProteinGym_OrganismalFitness"]
    of_neg = pd.DataFrame(of_neg_rows) if of_neg_rows else pd.DataFrame(columns=COLS)
    wreck_neg = pd.DataFrame(wreck_rows) if wreck_rows else pd.DataFrame(columns=COLS)
    if pos.empty:
        logger.warning("No growth GoF positives (z>=+2 on non-AMR OF). Table will be empty.")
        return pd.DataFrame(columns=COLS)
    # Never drop OF rows — that would erase the residue hold-out. Cap wrecks only.
    n_wreck = max(0, int(len(pos) * args.gof_neg_ratio) - len(of_neg))
    if not wreck_neg.empty and len(wreck_neg) > n_wreck:
        wreck_neg = wreck_neg.sample(n=n_wreck, random_state=args.seed)
    logger.info("Growth GoF pos=%s of_neg=%s wreck_neg=%s", len(pos), len(of_neg), len(wreck_neg))
    merged = pd.concat([pos, of_neg, wreck_neg], ignore_index=True)
    null = pd.concat([of_neg, wreck_neg], ignore_index=True)
    null["split"] = "null"
    return pd.concat([merged, null], ignore_index=True)


def build_amr_gof(of: pd.DataFrame, gof_src: pd.DataFrame, syn: pd.DataFrame, args) -> pd.DataFrame:
    """Resistance SNPs on known AMR genes. Residue hold-out, not family hold-out.

    Family transfer is not a ship bar — CARD families are different enzymes.
    """
    missense_rows: list[dict] = []
    id_rows: list[dict] = []
    wreck_rows: list[dict] = []
    rng = np.random.RandomState(args.seed)

    card = gof_src[gof_src["source"].astype(str).str.contains("CARD|AMRFinder", case=False, na=False)].copy() if not gof_src.empty else pd.DataFrame()
    if not card.empty:
        card["protein_id"] = card["ref_protein"].map(pid)
        for rec in card.itertuples(index=False):
            missense_rows.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
                species=getattr(rec, "species", "") or "", source=str(getattr(rec, "source", "CARD_AMR")),
                protein_id=rec.protein_id, task="amr_gof", split="train",
                channel="gof_pos", target=1.0, is_missense=True, wreck_type="none", label=1,
                dms_zscore=float(getattr(rec, "dms_zscore", 3.0) or 3.0),
            ))
            id_rows.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.ref_protein, gene=rec.gene,
                species=getattr(rec, "species", "") or "", source="amr_wt",
                protein_id=rec.protein_id, task="amr_gof", split="train",
                channel="gof_neg", target=0.0, is_missense=False, wreck_type="identity", label=0,
                site_index=0,
            ))
            for var in random_missense(str(rec.ref_protein), rng, n=2):
                if var == rec.var_protein:
                    continue
                missense_rows.append(_std_row(
                    ref_protein=rec.ref_protein, var_protein=var, gene=rec.gene,
                    species=getattr(rec, "species", "") or "", source="amr_random_missense",
                    protein_id=rec.protein_id, task="amr_gof", split="train",
                    channel="gof_neg", target=0.0, is_missense=True, wreck_type="none", label=0,
                ))

    mic = of.loc[of["is_amr_assay"]].copy()
    if not mic.empty:
        for rec in mic.itertuples(index=False):
            z = rec.dms_zscore
            if not np.isfinite(z) or z < Z_GOF_STRICT:
                continue
            missense_rows.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
                species=getattr(rec, "species", "") or "", source="ProteinGym_MIC",
                protein_id=rec.protein_id, task="amr_gof", split="train",
                channel="gof_pos", target=1.0, is_missense=True, label=1, dms_zscore=float(z),
            ))

    if missense_rows:
        site_splits = split_residues_within_protein(
            [r["protein_id"] for r in missense_rows],
            [r["ref_protein"] for r in missense_rows],
            [r["var_protein"] for r in missense_rows],
            seed=args.seed,
        )
        for row, split in zip(missense_rows, site_splits):
            row["split"] = split

    if syn is not None and not syn.empty and missense_rows:
        syn_cap = cap_synthetic(syn, min(2000, max(len(missense_rows), 100)), args.seed)
        for rec in syn_cap.itertuples(index=False):
            wreck_rows.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
                species=rec.species, source="synthetic_lof", protein_id=pid(rec.ref_protein),
                task="amr_gof", split="train", wreck_type=getattr(rec, "wreck_type", "stop"),
                channel="gof_neg", target=0.0, is_wreck=True, label=0,
            ))

    pos = [r for r in missense_rows if r["channel"] == "gof_pos"]
    of_neg = [r for r in missense_rows if r["channel"] == "gof_neg"] + id_rows
    if not pos:
        logger.warning("No AMR GoF positives")
        return pd.DataFrame(columns=COLS)
    n_wreck = max(0, int(len(pos) * args.gof_neg_ratio) - len(of_neg))
    if wreck_rows and len(wreck_rows) > n_wreck:
        wreck_rows = list(pd.DataFrame(wreck_rows).sample(n=n_wreck, random_state=args.seed).to_dict("records"))
    logger.info("AMR GoF pos=%s of_neg=%s wreck_neg=%s", len(pos), len(of_neg), len(wreck_rows))
    merged = pd.DataFrame(pos + of_neg + wreck_rows)
    merged = merged.drop_duplicates(subset=["ref_protein", "var_protein"]).reset_index(drop=True)
    null = pd.DataFrame(of_neg + wreck_rows)
    if not null.empty:
        null = null.drop_duplicates(subset=["ref_protein", "var_protein"]).copy()
        null["split"] = "null"
        return pd.concat([merged, null], ignore_index=True)
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Build PLMLoF training tables")
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-cap", type=int, default=80_000)
    p.add_argument("--missense-boost", type=float, default=4.0, help="Missense vs wreck total-weight ratio")
    p.add_argument("--gof-neg-ratio", type=float, default=10.0)
    p.add_argument(
        "--mlof-per-protein", type=int, default=2000,
        help="Max single-site missense variants per WT (stratified by z). 0 = keep all.",
    )
    p.add_argument(
        "--tasks", nargs="+", default=None,
        help="Subset of lof mlof growth_gof amr_gof (comma-separated ok). Default: all.",
    )
    args = p.parse_args()
    out = args.out or args.processed
    try:
        want = parse_tasks(args.tasks)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    pg_all_path = args.processed / SUBS_PARQUET
    pg_bact_path = args.processed / BACT_PARQUET
    if pg_all_path.exists():
        pg = pd.read_parquet(pg_all_path)
        logger.info("ProteinGym substitutions parquet %s rows", len(pg))
    elif pg_bact_path.exists():
        if "mlof" in want:
            raise SystemExit(
                f"Missing {pg_all_path}. MLoF trains on every ProteinGym substitution gene.\n"
                "On your laptop re-run:\n"
                "  python data/scripts/download_proteingym.py"
            )
        pg = pd.read_parquet(pg_bact_path)
        logger.info("ProteinGym bacterial parquet %s rows (GoF only)", len(pg))
    else:
        raise SystemExit(
            f"Missing {pg_all_path} and {pg_bact_path}. On your laptop run:\n"
            "  python data/scripts/download_proteingym.py"
        )

    ref_csv = RAW / "proteingym" / "DMS_substitutions.csv"
    if not ref_csv.exists():
        alt = list((RAW / "proteingym").glob("*substitutions*.csv"))
        if alt:
            ref_csv = alt[0]
    if not ref_csv.exists():
        raise SystemExit(f"Missing ProteinGym reference CSV at {ref_csv}")
    of_meta = load_of_meta(ref_csv)
    keys, amr_by_key = of_keys(of_meta)
    of = load_of_rows(pg, keys, amr_by_key)
    gof_wanted = "growth_gof" in want or "amr_gof" in want
    if of.empty and gof_wanted:
        raise SystemExit("No OrganismalFitness rows matched in the ProteinGym parquet")
    if of.empty:
        logger.warning("No prokaryote OrganismalFitness rows — GoF tables skipped if requested")

    mlof_src = pd.DataFrame()
    if "mlof" in want:
        mlof_src = load_mlof_dms(pg)
        if mlof_src.empty:
            raise SystemExit("No single-site missense rows in the ProteinGym substitutions parquet")

    syn_path = args.processed / "synthetic_lof.parquet"
    if syn_path.exists():
        syn = pd.read_parquet(syn_path)
        logger.info("Synthetic %s rows", len(syn))
    else:
        logger.warning("No synthetic_lof.parquet — LoF wreck channel will be empty")
        syn = pd.DataFrame()

    species_split = split_species(syn["species"].tolist(), args.seed) if "lof" in want and not syn.empty else {}

    gof_path = args.processed / "gof_growth_amr.parquet"
    gof_src = pd.read_parquet(gof_path) if gof_path.exists() else pd.DataFrame()
    if gof_src.empty:
        logger.warning("No gof_growth_amr.parquet — CARD AMR positives missing. Run curate_gof_table.py")

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["of_proteins"] = int(of["protein_id"].nunique()) if not of.empty else 0
    manifest["synthetic_rows_used"] = int(len(syn))
    manifest["tasks_written"] = want

    if "lof" in want:
        logger.info("── LoF (alignment-free synthetic wrecks) ──")
        lof = build_lof(syn, species_split, args)
        write_task(lof, out / "lof")
        manifest["lof_rows"] = int(len(lof))

    if "mlof" in want:
        logger.info("── MLoF (site ranker; nested residue + protein hold-out) ──")
        mlof = build_mlof(mlof_src, args)
        write_task(mlof, out / "mlof")
        manifest["mlof_rows"] = int(len(mlof))
        manifest["mlof_proteins"] = int(mlof_src["protein_id"].nunique())

    if "growth_gof" in want:
        logger.info("── growth GoF (residue hold-out within every protein) ──")
        growth = build_growth_gof(of, syn, args)
        write_task(growth, out / "growth_gof")
        manifest["growth_gof_rows"] = int(len(growth))

    if "amr_gof" in want:
        logger.info("── AMR GoF (residue hold-out on known families) ──")
        amr = build_amr_gof(of, gof_src, syn, args)
        write_task(amr, out / "amr_gof")
        manifest["amr_gof_rows"] = int(len(amr))

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", manifest_path)
    logger.info("Done. rsync data/processed/{lof,mlof,growth_gof,amr_gof} to Isambard (parquet is gitignored).")


if __name__ == "__main__":
    main()
