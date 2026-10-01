"""Assemble v2 train/val/test/null tables on a laptop (no GPU).

Reads:
  data/processed/proteingym_bacterial.parquet
  data/raw/proteingym/DMS_substitutions.csv  (or downloads reference only)
  data/processed/gof_growth_amr.parquet      (optional CARD source)
  data/processed/synthetic_lof.parquet

Writes under data/processed/v2/{lof,growth_gof,amr_gof}/
  train.parquet val.parquet test.parquet null.parquet

Never mixes GB1 / Tsuboyama into LoF or GoF labels.
Splits by protein (OF), species (synthetic wrecks), gene family (AMR).
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

from plmlof.v2 import LOF_WT
from plmlof.v2.domains import SURE_MIN, protein_id as pid
from plmlof.v2.labels import (
    EXTRA_MISSENSE_TYPES,
    INDOMAIN_MISSENSE_TYPES,
    Z_GOF_STRICT,
    Z_STRONG,
    Z_WEAK,
    Z_WT,
    channel_for_lof,
    lof_score_for_pair,
    lof_score_from_z,
)

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
]


def _std_row(**kwargs) -> dict:
    row = {c: "" if c not in {"target", "sample_weight", "lof_score", "dms_score", "dms_zscore", "is_wreck", "is_missense", "label"} else 0 for c in COLS}
    row.update({
        "dms_score": float("nan"),
        "dms_zscore": float("nan"),
        "lof_score": float("nan"),
        "target": 0.0,
        "sample_weight": 1.0,
        "is_wreck": False,
        "is_missense": False,
        "label": 0,
    })
    row.update(kwargs)
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


def split_proteins(protein_ids: list[str], seed: int = 42) -> dict[str, str]:
    """Deterministic protein hold-out: ~8 train / 2 val / 3 test for 13 proteins."""
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
    mapping = {}
    for p in uniq:
        mapping[p] = "test" if p in test else "val" if p in val else "train"
    logger.info("Protein split train=%s val=%s test=%s", len(train), len(val), len(test))
    return mapping


def split_species(species: list[str], seed: int = 42) -> dict[str, str]:
    rng = np.random.RandomState(seed)
    uniq = sorted({s for s in species if s})
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, round(n * 0.15)) if n else 0
    n_val = max(1, round(n * 0.10)) if n else 0
    test = set(uniq[:n_test])
    val = set(uniq[n_test:n_test + n_val])
    mapping = {}
    for s in uniq:
        mapping[s] = "test" if s in test else "val" if s in val else "train"
    logger.info("Species split train=%s val=%s test=%s", n - n_test - n_val, n_val, n_test)
    return mapping


def family_of(gene: str) -> str:
    g = gene.lower()
    for fam in AMR_FAMILIES:
        if fam in g:
            return fam
    token = re.split(r"[|;_/\s]+", gene)[0].lower()
    return token[:12] if token else "unknown"


def split_families(families: list[str], seed: int = 42) -> dict[str, str]:
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
            df[col] = "" if col not in {"target", "sample_weight", "lof_score", "label", "is_wreck", "is_missense"} else 0
    df = df[COLS]
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
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


def build_lof(of: pd.DataFrame, syn: pd.DataFrame, protein_split: dict[str, str], species_split: dict[str, str], args) -> pd.DataFrame:
    rows = []
    for rec in of.itertuples(index=False):
        z = rec.dms_zscore
        if not np.isfinite(z):
            continue
        score = lof_score_from_z(float(z), is_wreck=False)
        if score is None:
            continue  # GoF tail — not this table
        ch = channel_for_lof(score, None, float(z))
        rows.append(_std_row(
            ref_protein=rec.ref_protein,
            var_protein=rec.var_protein,
            gene=rec.gene,
            species=getattr(rec, "species", "") or "",
            source="ProteinGym_OrganismalFitness",
            protein_id=rec.protein_id,
            task="lof",
            split=protein_split.get(rec.protein_id, "train"),
            wreck_type="none",
            channel=ch,
            target=score,
            lof_score=score,
            dms_score=float(getattr(rec, "dms_score", float("nan"))),
            dms_zscore=float(z),
            is_wreck=False,
            is_missense=True,
            sample_weight=0.5 if ch == "weak_missense" else 1.0,
            label=0 if score >= 0.4 else 1,
        ))
    of_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=COLS)
    n_missense = len(of_df)

    syn = cap_synthetic(syn, args.synthetic_cap, args.seed) if not syn.empty else syn
    syn_rows = []
    for rec in syn.itertuples(index=False):
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
        ch = channel_for_lof(score, wreck_type, None)
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

    # One identity WT per synthetic gene used
    id_rows = []
    seen = set()
    for rec in syn_df.itertuples(index=False):
        key = rec.protein_id
        if key in seen:
            continue
        seen.add(key)
        id_rows.append(_std_row(
            ref_protein=rec.ref_protein,
            var_protein=rec.ref_protein,
            gene=rec.gene,
            species=rec.species,
            source="synthetic_identity",
            protein_id=key,
            task="lof",
            split=rec.split,
            wreck_type="identity",
            channel="wt",
            target=LOF_WT,
            lof_score=LOF_WT,
            is_wreck=False,
            is_missense=False,
            sample_weight=1.0,
            label=1,
        ))
    id_df = pd.DataFrame(id_rows) if id_rows else pd.DataFrame(columns=COLS)

    n_wreck = int(syn_df["is_wreck"].sum()) if not syn_df.empty else 0
    wreck_w, miss_w = missense_weight_scale(n_missense, n_wreck, args.missense_boost)
    if not of_df.empty:
        of_df = of_df.copy()
        of_df["sample_weight"] = of_df["sample_weight"] * miss_w
    if not syn_df.empty:
        syn_df = syn_df.copy()
        syn_df.loc[syn_df["is_wreck"], "sample_weight"] = wreck_w
    logger.info("LoF weights wreck=%.4f missense_scale=%.4f (n_wreck=%s n_missense=%s)", wreck_w, miss_w, n_wreck, n_missense)

    lof = pd.concat([of_df, syn_df, id_df], ignore_index=True)

    # Null: WT-like OF + identities + a slice of random missense on OF WT refs
    rng = np.random.RandomState(args.seed)
    null_bits = []
    wt = lof[lof["channel"] == "wt"]
    if not wt.empty:
        null_bits.append(wt.assign(split="null"))
    of_wt_refs = of[np.abs(of["dms_zscore"]) <= Z_WT] if not of.empty else of
    if not of_wt_refs.empty:
        for ref in of_wt_refs["ref_protein"].drop_duplicates().head(200):
            for var in random_missense(str(ref), rng, n=2):
                null_bits.append(pd.DataFrame([_std_row(
                    ref_protein=ref, var_protein=var, gene="random_missense",
                    species="", source="random_missense", protein_id=pid(ref),
                    task="lof", split="null", wreck_type="none", channel="wt",
                    target=0.0, lof_score=0.0, is_wreck=False, is_missense=True, label=1,
                )]))
    if null_bits:
        lof = pd.concat([lof, *null_bits], ignore_index=True)
    return lof


def build_growth_gof(of: pd.DataFrame, syn: pd.DataFrame, protein_split: dict[str, str], args) -> pd.DataFrame:
    growth = of.loc[~of["is_amr_assay"]].copy()
    pos_rows, neg_rows = [], []
    for rec in growth.itertuples(index=False):
        z = rec.dms_zscore
        if not np.isfinite(z):
            continue
        split = protein_split.get(rec.protein_id, "train")
        base = dict(
            ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
            species=getattr(rec, "species", "") or "", source="ProteinGym_OrganismalFitness",
            protein_id=rec.protein_id, task="growth_gof", split=split,
            dms_zscore=float(z), dms_score=float(getattr(rec, "dms_score", float("nan"))),
            is_missense=True, is_wreck=False, wreck_type="none",
        )
        if z >= Z_GOF_STRICT:
            pos_rows.append(_std_row(**base, channel="gof_pos", target=1.0, label=1, sample_weight=1.0))
        elif abs(z) <= Z_WT:
            neg_rows.append(_std_row(**base, channel="gof_neg", target=0.0, label=0, sample_weight=1.0))
        # +1 < z < +2 is the no-call band — omitted from train/val/test.

    # Wreck negatives so truncation cannot be GoF
    syn_cap = cap_synthetic(syn, min(args.synthetic_cap, max(len(pos_rows) * 2, 500)), args.seed)
    for rec in syn_cap.itertuples(index=False):
        neg_rows.append(_std_row(
            ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
            species=rec.species, source="synthetic_lof", protein_id=pid(rec.ref_protein),
            task="growth_gof", split="train", wreck_type=getattr(rec, "wreck_type", "stop"),
            channel="gof_neg", target=0.0, is_wreck=True, is_missense=False, label=0,
        ))

    pos = pd.DataFrame(pos_rows)
    neg = pd.DataFrame(neg_rows)
    if pos.empty:
        logger.warning("No growth GoF positives (z>=+2 on non-AMR OF). Table will be empty.")
        return pd.DataFrame(columns=COLS)
    merged = balance_gof(pos, neg, args.gof_neg_ratio, args.seed)
    # Null = negatives (held-out copies tagged null, excluding train overlap by split)
    null = neg.copy()
    null["split"] = "null"
    return pd.concat([merged, null], ignore_index=True)


def build_amr_gof(of: pd.DataFrame, gof_src: pd.DataFrame, syn: pd.DataFrame, args) -> pd.DataFrame:
    rows_pos, rows_neg = [], []
    rng = np.random.RandomState(args.seed)

    card = gof_src[gof_src["source"].astype(str).str.contains("CARD|AMRFinder", case=False, na=False)].copy() if not gof_src.empty else pd.DataFrame()
    if not card.empty:
        card["protein_id"] = card["ref_protein"].map(pid)
        card["family"] = card["gene"].map(family_of)
        fam_split = split_families(card["family"].tolist(), args.seed)
        for rec in card.itertuples(index=False):
            fam = rec.family
            rows_pos.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
                species=getattr(rec, "species", "") or "", source=str(getattr(rec, "source", "CARD_AMR")),
                protein_id=rec.protein_id, task="amr_gof", split=fam_split.get(fam, "train"),
                channel="gof_pos", target=1.0, is_missense=True, wreck_type="none", label=1,
                dms_zscore=float(getattr(rec, "dms_zscore", 3.0) or 3.0),
            ))
            # WT identity negative
            rows_neg.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.ref_protein, gene=rec.gene,
                species=getattr(rec, "species", "") or "", source="amr_wt",
                protein_id=rec.protein_id, task="amr_gof", split=fam_split.get(fam, "train"),
                channel="gof_neg", target=0.0, is_missense=False, wreck_type="identity", label=0,
            ))
            for var in random_missense(str(rec.ref_protein), rng, n=2):
                if var == rec.var_protein:
                    continue
                rows_neg.append(_std_row(
                    ref_protein=rec.ref_protein, var_protein=var, gene=rec.gene,
                    species=getattr(rec, "species", "") or "", source="amr_random_missense",
                    protein_id=rec.protein_id, task="amr_gof", split=fam_split.get(fam, "train"),
                    channel="gof_neg", target=0.0, is_missense=True, wreck_type="none", label=0,
                ))

    # MIC-up OF (TEM/VIM/KKA2 etc.) — AMR, never growth
    mic = of.loc[of["is_amr_assay"]].copy()
    if not mic.empty:
        mic["family"] = mic["gene"].map(family_of)
        extra_fams = split_families(mic["family"].tolist(), args.seed + 1)
        for rec in mic.itertuples(index=False):
            z = rec.dms_zscore
            if not np.isfinite(z) or z < Z_GOF_STRICT:
                continue
            split = extra_fams.get(rec.family, "train")
            rows_pos.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
                species=getattr(rec, "species", "") or "", source="ProteinGym_MIC",
                protein_id=rec.protein_id, task="amr_gof", split=split,
                channel="gof_pos", target=1.0, is_missense=True, label=1, dms_zscore=float(z),
            ))

    # Wrecks of AMR proteins
    amr_refs = {r["ref_protein"] for r in rows_pos}
    if syn is not None and not syn.empty and amr_refs:
        # Use generic wrecks as "truncation is not resistance"
        syn_cap = cap_synthetic(syn, min(2000, max(len(rows_pos), 100)), args.seed)
        for rec in syn_cap.itertuples(index=False):
            rows_neg.append(_std_row(
                ref_protein=rec.ref_protein, var_protein=rec.var_protein, gene=rec.gene,
                species=rec.species, source="synthetic_lof", protein_id=pid(rec.ref_protein),
                task="amr_gof", split="train", wreck_type=getattr(rec, "wreck_type", "stop"),
                channel="gof_neg", target=0.0, is_wreck=True, label=0,
            ))

    pos = pd.DataFrame(rows_pos).drop_duplicates(subset=["ref_protein", "var_protein"]) if rows_pos else pd.DataFrame()
    neg = pd.DataFrame(rows_neg).drop_duplicates(subset=["ref_protein", "var_protein"]) if rows_neg else pd.DataFrame()
    if pos.empty:
        logger.warning("No AMR GoF positives")
        return pd.DataFrame(columns=COLS)
    merged = balance_gof(pos, neg, args.gof_neg_ratio, args.seed)
    null = neg.copy()
    null["split"] = "null"
    return pd.concat([merged, null], ignore_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Build PLMLoF v2 training tables")
    p.add_argument("--processed", type=Path, default=PROCESSED)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-cap", type=int, default=80_000)
    p.add_argument("--missense-boost", type=float, default=4.0, help="Missense vs wreck total-weight ratio")
    p.add_argument("--gof-neg-ratio", type=float, default=10.0)
    args = p.parse_args()
    out = args.out or (args.processed / "v2")

    pg_path = args.processed / "proteingym_bacterial.parquet"
    if not pg_path.exists():
        raise SystemExit(
            f"Missing {pg_path}. On your laptop run:\n"
            "  python data/scripts/download_proteingym.py"
        )
    pg = pd.read_parquet(pg_path)
    logger.info("ProteinGym parquet %s rows", len(pg))

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
    if of.empty:
        raise SystemExit("No OrganismalFitness rows matched in the ProteinGym parquet")

    protein_split = split_proteins(of["protein_id"].tolist(), args.seed)

    syn_path = args.processed / "synthetic_lof.parquet"
    if syn_path.exists():
        syn = pd.read_parquet(syn_path)
        logger.info("Synthetic %s rows", len(syn))
    else:
        logger.warning("No synthetic_lof.parquet — LoF wreck channel will be empty")
        syn = pd.DataFrame()

    species_split = split_species(syn["species"].tolist(), args.seed) if not syn.empty else {}

    gof_path = args.processed / "gof_growth_amr.parquet"
    gof_src = pd.read_parquet(gof_path) if gof_path.exists() else pd.DataFrame()
    if gof_src.empty:
        logger.warning("No gof_growth_amr.parquet — CARD AMR positives missing. Run curate_gof_table.py")

    logger.info("── LoF ──")
    lof = build_lof(of, syn, protein_split, species_split, args)
    write_task(lof, out / "lof")

    logger.info("── growth GoF ──")
    growth = build_growth_gof(of, syn, protein_split, args)
    write_task(growth, out / "growth_gof")

    logger.info("── AMR GoF ──")
    amr = build_amr_gof(of, gof_src, syn, args)
    write_task(amr, out / "amr_gof")

    # Manifest for the cluster
    manifest = {
        "lof_rows": int(len(lof)),
        "growth_gof_rows": int(len(growth)),
        "amr_gof_rows": int(len(amr)),
        "of_proteins": int(of["protein_id"].nunique()),
        "synthetic_rows_used": int(len(syn)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out / "manifest.json")
    logger.info("Done. rsync data/processed/v2/ to Isambard (parquet is gitignored).")


if __name__ == "__main__":
    main()
