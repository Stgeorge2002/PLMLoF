"""Assemble a bacterial GoF training table from honest 'up' phenotypes.

Sources (all rows are label=2):
  1. ProteinGym Prokaryote OrganismalFitness, z > +1
     (growth / MIC / in-cell fitness — not GB1 binding or Tsuboyama stability)
  2. CARD protein-variant models: WT protein vs documented resistance SNP
  3. AMRFinderPlus point-mutation catalog, applied to CARD/AMRFinder WT proteins
     when a matching sequence is available
  4. MaveDB prokaryote score sets, z > +1 (skipped if the API is down)

Output columns match train.parquet: ref_protein, var_protein, label, gene,
species, source, dms_score, dms_zscore.

Run from the repo root (compute node if ProteinGym scores are not already local):

    python data/scripts/curate_gof_table.py
"""

from __future__ import annotations

import io
import json
import logging
import re
import ssl
import sys
import tarfile
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from download_proteingym import (  # noqa: E402
    _apply_mutation_string,
    download_proteingym_reference,
    download_proteingym_scores,
    filter_bacterial_assays,
    process_dms_scores,
)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
OUT_PATH = REPO / "data" / "processed" / "gof_growth_amr.parquet"

CARD_URL = "https://card.mcmaster.ca/latest/data"
AMRFINDER_MUTATION_URLS = [
    "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/AMRProt-mutation",
]
MAVEDB_SCORESETS = "https://api.mavedb.org/api/v1/score-sets"
SNP_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]$")
GROWTH_TOKENS = (
    "growth",
    "antibiotic",
    "antibiotics",
    "amp ",
    "mic",
    "resistance",
    "kanamycin",
    "fitness",
    "toxin",
    "dilution",
    "efflux",
)


def _ssl() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _get(url: str, dest: Path | None = None, timeout: int = 180) -> bytes | None:
    req = Request(url, headers={"User-Agent": "PLMLoF/1.0"})
    try:
        with urlopen(req, timeout=timeout, context=_ssl()) as resp:  # noqa: S310
            data = resp.read()
    except (URLError, HTTPError, TimeoutError) as exc:
        logger.warning("GET failed %s: %s", url, exc)
        return None
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def _row(
    *,
    gene: str,
    species: str,
    ref: str,
    var: str,
    source: str,
    score: float = float("nan"),
    z: float = 2.5,
) -> dict | None:
    if not ref or not var or ref == var:
        return None
    if not (20 <= len(ref) <= 1024 and 20 <= len(var) <= 1024):
        return None
    return {
        "gene": gene,
        "species": species,
        "ref_protein": ref,
        "var_protein": var,
        "ref_dna": "",
        "var_dna": "",
        "label": 2,
        "dms_score": score,
        "dms_zscore": z,
        "source": source,
    }


def _is_growth_like(selection_type: object, selection_assay: object) -> bool:
    text = f"{selection_type or ''} {selection_assay or ''}".lower()
    if not text.strip():
        return True
    return any(tok in text for tok in GROWTH_TOKENS)


def organismal_fitness_meta(ref_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ref_path)
    mask = df["taxon"].astype(str).str.strip().str.lower().eq("prokaryote")
    mask &= df["coarse_selection_type"].astype(str).str.strip().eq("OrganismalFitness")
    of = df.loc[mask].copy()
    of = of[of.apply(
        lambda r: _is_growth_like(r.get("selection_type"), r.get("selection_assay")),
        axis=1,
    )]
    logger.info("OrganismalFitness growth/MIC assays: %s", len(of))
    return of


def proteingym_gof(of_assays: pd.DataFrame) -> pd.DataFrame:
    """GoF tail of OrganismalFitness assays."""
    parquet = REPO / "data" / "processed" / "proteingym_bacterial.parquet"
    if parquet.exists():
        pg = pd.read_parquet(parquet)
        ids = set(of_assays["DMS_id"].dropna().astype(str))
        stems = set(Path(str(x)).stem for x in of_assays.get("DMS_filename", pd.Series(dtype=str)).dropna())
        keep = ids | stems
        hit = pg[pg["gene"].astype(str).isin(keep) & (pg["label"].astype(int) == 2)].copy()
        if hit.empty and "source" in pg.columns:
            # gene may be stored as full filename
            hit = pg[
                pg["gene"].astype(str).map(lambda g: Path(g).stem in keep)
                & (pg["label"].astype(int) == 2)
            ].copy()
        if not hit.empty:
            hit = hit.copy()
            hit["source"] = "ProteinGym_OrganismalFitness"
            logger.info("ProteinGym OF GoF from parquet: %s", len(hit))
            return hit

    scores_dir = REPO / "data" / "raw" / "proteingym" / "substitutions"
    if not scores_dir.exists() or not any(scores_dir.rglob("*.csv")):
        download_proteingym_scores()
    scored = process_dms_scores(scores_dir, of_assays)
    if scored.empty:
        logger.warning("No ProteinGym OrganismalFitness scores")
        return pd.DataFrame()
    gof = scored[scored["label"].astype(int) == 2].copy()
    gof["source"] = "ProteinGym_OrganismalFitness"
    logger.info("ProteinGym OF GoF from score CSVs: %s", len(gof))
    return gof


def _walk_snps(obj: object) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        val = obj.get("param_value")
        if isinstance(val, str):
            for part in re.split(r"[,;/]\s*", val):
                part = part.strip().replace(" ", "")
                if SNP_RE.match(part):
                    found.append(part)
        for v in obj.values():
            found.extend(_walk_snps(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_walk_snps(v))
    elif isinstance(obj, str) and SNP_RE.match(obj.strip()):
        found.append(obj.strip())
    return found


def _walk_protein_seq(obj: object) -> str:
    if isinstance(obj, dict):
        for key in ("protein_sequence", "aasequence", "aa_sequence", "sequence"):
            inner = obj.get(key)
            if isinstance(inner, str) and len(inner) > 30 and set(inner) <= set("ACDEFGHIKLMNPQRSTVWY"):
                return inner
            if isinstance(inner, dict):
                seq = inner.get("sequence")
                if isinstance(seq, str) and len(seq) > 30:
                    return seq
        for v in obj.values():
            seq = _walk_protein_seq(v)
            if seq:
                return seq
    elif isinstance(obj, list):
        for v in obj:
            seq = _walk_protein_seq(v)
            if seq:
                return seq
    return ""


def download_card(dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dest_dir.rglob("card.json"))
    if existing:
        return existing[0]
    blob = _get(CARD_URL, dest_dir / "card_latest.tar.bz2", timeout=300)
    if not blob:
        return None
    archive = dest_dir / "card_latest.tar.bz2"
    try:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(dest_dir, filter="data")
    except TypeError:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(dest_dir)
    hits = list(dest_dir.rglob("card.json"))
    return hits[0] if hits else None


def card_gof(card_json: Path) -> pd.DataFrame:
    raw = json.loads(card_json.read_text(encoding="utf-8", errors="replace"))
    models = raw.values() if isinstance(raw, dict) else raw
    rows = []
    n_var_models = 0
    for model in models:
        if not isinstance(model, dict):
            continue
        if str(model.get("model_type", "")).lower() != "protein variant model":
            continue
        n_var_models += 1
        ref = _walk_protein_seq(model.get("model_sequences", model))
        snps = list(dict.fromkeys(_walk_snps(model.get("model_param", {}))))
        if not ref or not snps:
            continue
        name = str(model.get("model_name") or model.get("model_id") or "CARD")
        species = "Bacteria"
        for token in ("Escherichia coli", "Klebsiella", "Staphylococcus", "Pseudomonas",
                      "Salmonella", "Mycobacterium", "Streptococcus", "Enterococcus",
                      "Acinetobacter", "Neisseria", "Helicobacter"):
            if token.lower() in name.lower():
                species = token
                break
        for snp in snps:
            var = _apply_mutation_string(ref, snp, strict=True)
            rec = _row(
                gene=f"{name}|{snp}",
                species=species,
                ref=ref,
                var=var,
                source="CARD_AMR",
                z=3.0,
            )
            if rec:
                rows.append(rec)
    logger.info("CARD protein-variant models=%s usable SNP pairs=%s", n_var_models, len(rows))
    return pd.DataFrame(rows)


def amrfinder_gof(card_df: pd.DataFrame) -> pd.DataFrame:
    """Apply AMRFinder point mutations onto CARD WT sequences of the same gene.

    AMRFinder's mutation file has no sequences. We only keep SNPs whose gene
    token matches a CARD WT we already have, and whose WT residue matches.
    """
    dest = RAW / "amrfinder" / "AMRProt-mutation"
    data = None
    if dest.exists() and dest.stat().st_size > 100:
        data = dest.read_bytes()
    else:
        for url in AMRFINDER_MUTATION_URLS:
            data = _get(url, dest, timeout=120)
            if data:
                break
    if not data:
        logger.warning("AMRFinder mutation catalog not downloaded")
        return pd.DataFrame()

    text = data.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        return pd.DataFrame()

    # Typical: taxgroup gene mutation ...  OR headered TSV
    header = None
    first = lines[0].split("\t")
    if any(h.lower() in {"gene", "mutation", "snp", "variant"} for h in first):
        header = [h.strip().lower() for h in first]
        body = lines[1:]
    else:
        body = lines

    gene_to_ref: dict[str, tuple[str, str]] = {}
    if not card_df.empty:
        for _, r in card_df.iterrows():
            key = str(r["gene"]).split("|")[0].lower()
            gene_to_ref.setdefault(key, (r["ref_protein"], r["species"]))

    def lookup_ref(gene: str) -> tuple[str, str] | None:
        g = gene.lower()
        if g in gene_to_ref:
            return gene_to_ref[g]
        for key, val in gene_to_ref.items():
            if g in key or key.endswith(g) or g in key.replace(" ", ""):
                return val
        return None

    rows = []
    for ln in body:
        parts = ln.split("\t")
        if header:
            rec = dict(zip(header, parts))
            gene = rec.get("gene") or rec.get("gene_symbol") or rec.get("element_symbol") or ""
            mut = rec.get("mutation") or rec.get("snp") or rec.get("variant") or rec.get("name") or ""
            tax = rec.get("taxgroup") or rec.get("taxonomy") or rec.get("organism") or "Bacteria"
        else:
            if len(parts) < 3:
                continue
            tax, gene, mut = parts[0], parts[1], parts[2]
        mut = mut.strip().replace(" ", "")
        if not SNP_RE.match(mut):
            continue
        hit = lookup_ref(gene)
        if not hit:
            continue
        ref, species = hit
        var = _apply_mutation_string(ref, mut, strict=True)
        row = _row(
            gene=f"AMRFinder|{gene}|{mut}",
            species=str(tax) if tax else species,
            ref=ref,
            var=var,
            source="AMRFinder_AMR",
            z=3.0,
        )
        if row:
            rows.append(row)
    logger.info("AMRFinder SNPs applied on CARD WTs: %s", len(rows))
    return pd.DataFrame(rows)


def mavedb_gof(limit_sets: int = 40) -> pd.DataFrame:
    blob = _get(f"{MAVEDB_SCORESETS}?size=100", timeout=60)
    if not blob:
        logger.warning("MaveDB API unavailable — skipping")
        return pd.DataFrame()
    try:
        payload = json.loads(blob.decode("utf-8"))
    except json.JSONDecodeError:
        logger.warning("MaveDB JSON parse failed")
        return pd.DataFrame()

    items = payload.get("items") or payload.get("scoreSets") or payload.get("data") or []
    if isinstance(payload, list):
        items = payload

    rows = []
    used = 0
    for item in items:
        if used >= limit_sets:
            break
        if not isinstance(item, dict):
            continue
        org = json.dumps(item.get("targetGenes") or item.get("target") or item).lower()
        if not any(x in org for x in ("bacter", "escherichia", "coli", "klebsiella", "salmonella",
                                      "staphylococcus", "pseudomonas", "mycobacterium")):
            continue
        urn = item.get("urn") or item.get("id")
        if not urn:
            continue
        seq = ""
        for gene in item.get("targetGenes") or []:
            if isinstance(gene, dict):
                seq = gene.get("sequence") or ""
                if isinstance(seq, dict):
                    seq = seq.get("sequence") or ""
                if seq:
                    break
        if not seq or len(seq) < 30:
            continue
        scores_blob = _get(f"{MAVEDB_SCORESETS}/{urn}/scores", timeout=60)
        if not scores_blob:
            continue
        used += 1
        try:
            sdf = pd.read_csv(io.BytesIO(scores_blob))
        except Exception:
            continue
        score_col = next((c for c in sdf.columns if c.lower() in {"score", "dms_score", "fitness"}), None)
        mut_col = next((c for c in sdf.columns if "hgvs" in c.lower() or c.lower() in {"mutant", "substitution"}), None)
        if score_col is None or mut_col is None or sdf[score_col].std() in (0, None):
            continue
        z = (sdf[score_col] - sdf[score_col].mean()) / sdf[score_col].std()
        species = "Bacteria"
        for gene in item.get("targetGenes") or []:
            if isinstance(gene, dict) and gene.get("targetOrganism"):
                species = str(gene["targetOrganism"])
        for (_, rec), zi in zip(sdf.iterrows(), z):
            if zi <= 1.0:
                continue
            mut = str(rec[mut_col])
            m = re.search(r"([A-Z]\d+[A-Z])", mut.replace("p.", ""))
            if not m:
                continue
            var = _apply_mutation_string(seq, m.group(1), strict=True)
            row = _row(
                gene=f"MaveDB|{urn}|{m.group(1)}",
                species=species,
                ref=seq,
                var=var,
                source="MaveDB",
                score=float(rec[score_col]),
                z=float(zi),
            )
            if row:
                rows.append(row)
    logger.info("MaveDB GoF rows: %s (score sets used=%s)", len(rows), used)
    return pd.DataFrame(rows)


def assemble() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "processed").mkdir(parents=True, exist_ok=True)

    ref_path = download_proteingym_reference()
    of_assays = organismal_fitness_meta(ref_path)
    chunks = [proteingym_gof(of_assays)]

    card_json = download_card(RAW / "card")
    card_df = card_gof(card_json) if card_json else pd.DataFrame()
    chunks.append(card_df)
    chunks.append(amrfinder_gof(card_df) if not card_df.empty else pd.DataFrame())
    chunks.append(mavedb_gof())

    parts = [c for c in chunks if c is not None and not c.empty]
    if not parts:
        raise RuntimeError("No GoF rows from any source")
    cols = [
        "ref_protein", "var_protein", "label", "gene", "species",
        "source", "dms_score", "dms_zscore", "ref_dna", "var_dna",
    ]
    for p in parts:
        for c in cols:
            if c not in p.columns:
                p[c] = "" if c in ("ref_dna", "var_dna", "gene", "species", "source") else float("nan")
            if c == "label":
                p[c] = 2
    out = pd.concat([p[cols] for p in parts], ignore_index=True)
    out = out.drop_duplicates(subset=["ref_protein", "var_protein"])
    out["label"] = 2
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = assemble()
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %s rows -> %s", len(df), OUT_PATH)
    logger.info("By source:\n%s", df["source"].value_counts().to_string())
    logger.info("Unique proteins (ref): %s", df["ref_protein"].nunique())
    logger.info("Unique species: %s", df["species"].nunique())


if __name__ == "__main__":
    main()
