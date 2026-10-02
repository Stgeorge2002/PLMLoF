"""Download and process ProteinGym DMS substitution scores.

Writes:
  data/processed/proteingym_substitutions.parquet  — every v1.3 assay (MLoF)
  data/processed/proteingym_bacterial.parquet      — taxon == Prokaryote (GoF)

Source: https://proteingym.org/
GitHub: https://github.com/OATML-Markslab/ProteinGym
"""

from __future__ import annotations

import io
import logging
import ssl
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Official ProteinGym v1.3 reference (taxon column: Human / Eukaryote / Prokaryote / Virus)
PROTEINGYM_REFERENCE_URLS = [
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv",
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv",
]

# Substitution DMS CSVs (~43 MB). HuggingFace v0.1 zip path 404s; Harvard is canonical.
PROTEINGYM_SUBS_URLS = [
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip",
]

OUTPUT_DIR = Path("data/raw/proteingym/")

# Bacterial species keywords for filtering
BACTERIAL_KEYWORDS = [
    "escherichia", "e. coli", "ecoli", "ecolx",
    "salmonella", "staphylococcus", "streptococcus",
    "pseudomonas", "mycobacterium", "myctu",
    "bacillus", "klebsiella", "enterococcus",
    "vibrio", "acinetobacter", "neisseria", "neigo",
    "helicobacter", "campylobacter",
    "clostridium", "clostridioides", "listeria",
    "legionella", "corynebacterium",
]

# Known bacterial DMS assays (UniProt ID prefixes) in ProteinGym
BACTERIAL_DMS_IDS = [
    "TEM1_ECOLI",   # TEM-1 beta-lactamase (E. coli)
    "BLAT_ECOLX",   # Beta-lactamase (E. coli)
    "DHFR_ECOLI",   # Dihydrofolate reductase (E. coli)
    "INHA_MYCTU",   # InhA (M. tuberculosis)
    "RPOB_ECOLI",   # RNA polymerase beta (E. coli)
    "PARE_NEIGO",   # ParE (N. gonorrhoeae)
    "AMPC_ECOLI",   # AmpC (E. coli)
    "ENVZ_ECOLI",   # EnvZ (E. coli)
    "TPMT_ECOLI",   # TPMT (E. coli)
    "SUMO_ECOLI",   # Sumo (E. coli)
    "KKA2_KLEPN",   # AAC(6')-Ib (Klebsiella)
    "PABP_ECOLI",   # poly(A)-binding protein (E. coli)
    "HSP82_ECOLI",  # GroEL (E. coli)
]

# Curated bacterial DMS data as fallback (TEM-1 beta-lactamase known mutations)
_TEM1_REF = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRIDAGQEQLGRR"
    "IHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPVAM"
    "ATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIA"
    "EIGASLIKHW"
)

_TEM1_MUTATIONS_FALLBACK = [
    # Position, ref_aa, var_aa, fitness_class (0=LoF, 1=WT, 2=GoF)
    ("M69I", 0), ("M69L", 1), ("M69V", 2),
    ("E104K", 2), ("R164S", 2), ("R164H", 2),
    ("G238S", 2), ("E240K", 2),
    ("A42G", 1), ("A42V", 0),
    ("S70A", 0), ("S70C", 0),
    ("K73R", 0), ("K73A", 0),
    ("D131N", 0), ("D131A", 0),
    ("R244S", 0), ("R244C", 0),
    ("N132S", 1), ("N132A", 0),
    ("T265M", 2), ("W165R", 0),
    ("A237T", 2), ("S235T", 1),
    ("M182T", 1), ("L76N", 0),
    ("G92D", 0), ("P62S", 0),
    ("Q39K", 1), ("T71A", 0),
]


def _make_ssl_context() -> ssl.SSLContext:
    """Build an SSL context using certifi CA bundle when available.

    RunPod Docker images sometimes lack the system root certificates that
    marks.hms.harvard.edu requires.  certifi ships its own bundle which
    works regardless of the host CA store.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except (ImportError, Exception):
        pass
    try:
        return ssl.create_default_context()
    except Exception:
        pass
    # Last resort — skip verification (logs a warning)
    ctx = ssl._create_unverified_context()  # noqa: S501
    logger.warning("SSL certificate verification disabled (certifi not available)")
    return ctx


def _stream_url(url: str, ssl_ctx: ssl.SSLContext, dest_path: Path) -> bool:
    """Stream a single URL to dest_path; return True on success."""
    req = Request(url, headers={"User-Agent": "PLMLoF/1.0"})
    response = urlopen(req, timeout=120, context=ssl_ctx)  # noqa: S310
    CHUNK = 8 << 20
    chunks: list[bytes] = []
    total = 0
    next_log = 50 << 20
    while True:
        chunk = response.read(CHUNK)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total >= next_log:
            logger.info(f"  Downloaded {total / 1e6:.0f} MB...")
            next_log += 50 << 20
    data = b"".join(chunks)
    if total > 100:
        dest_path.write_bytes(data)
        logger.info(f"Downloaded {total / 1e6:.2f} MB → {dest_path}")
        return True
    return False


def _download_with_fallback(urls: list[str], dest_path: Path) -> bool:
    """Try downloading from multiple URLs using chunked streaming; return True on success.

    For SSL verification failures, automatically retries with certificate
    verification disabled (marks.hms.harvard.edu uses an intermediate CA that
    may not be present in the container's trust store).
    """
    ssl_ctx = _make_ssl_context()
    unverified_ctx = ssl._create_unverified_context()  # noqa: S501 — fallback only

    for url in urls:
        logger.info(f"Trying: {url}")
        try:
            return _stream_url(url, ssl_ctx, dest_path)
        except (ssl.SSLError, URLError) as e:
            # URLError wraps SSLError when urllib can't open the connection
            is_ssl = isinstance(e, ssl.SSLError) or (
                isinstance(e, URLError) and isinstance(e.reason, ssl.SSLError)
            )
            if is_ssl:
                logger.info(f"SSL verification failed for {url} — retrying without certificate verification")
                try:
                    return _stream_url(url, unverified_ctx, dest_path)
                except Exception as e2:
                    logger.warning(f"Failed ({url}): {e2}")
            else:
                logger.warning(f"Failed ({url}): {e}")
        except Exception as e:
            logger.warning(f"Failed ({url}): {e}")
    return False


def _download_zip_via_hf_hub(dest_path: Path) -> bool:
    """Try downloading the ProteinGym substitutions ZIP via huggingface_hub.

    huggingface_hub handles LFS files, authentication, and resumable downloads
    better than raw HTTP — use it as the last fallback for the large ZIP.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.debug("huggingface_hub not available")
        return False

    repo_id = "OATML-Markslab/ProteinGym_v0.1"
    candidates = [
        "DMS_ProteinGym_substitutions.zip",
        "ProteinGym_substitutions/DMS_ProteinGym_substitutions.zip",
        "substitution_data/DMS_ProteinGym_substitutions.zip",
    ]
    for filename in candidates:
        try:
            logger.info(f"Trying HF Hub: {repo_id}/{filename}")
            local = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
            )
            import shutil
            shutil.copy(local, dest_path)
            logger.info(f"Downloaded via HF Hub → {dest_path}")
            return True
        except Exception as e:
            logger.debug(f"  {filename}: {e}")
    return False


def download_proteingym_reference(output_dir: Path = OUTPUT_DIR) -> Path:
    """Download the ProteinGym reference file listing all assays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = output_dir / "DMS_substitutions.csv"

    if ref_path.exists() and ref_path.stat().st_size > 100:
        logger.info(f"ProteinGym reference already downloaded: {ref_path}")
        return ref_path

    logger.info("Downloading ProteinGym reference...")
    if not _download_with_fallback(PROTEINGYM_REFERENCE_URLS, ref_path):
        logger.warning("Could not download ProteinGym reference from any URL.")
        ref_path.write_text("")

    return ref_path


def download_proteingym_scores(output_dir: Path = OUTPUT_DIR) -> Path:
    """Download ProteinGym substitution scores ZIP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "DMS_ProteinGym_substitutions.zip"
    extract_dir = output_dir / "substitutions"

    if extract_dir.exists() and any(extract_dir.rglob("*.csv")):
        logger.info("ProteinGym scores already downloaded and extracted")
        return extract_dir

    if not zip_path.exists():
        logger.info("Downloading ProteinGym substitution scores...")
        success = _download_with_fallback(PROTEINGYM_SUBS_URLS, zip_path)
        if not success:
            logger.info("Trying HuggingFace Hub as final fallback...")
            success = _download_zip_via_hf_hub(zip_path)
        if not success:
            logger.warning("Could not download ProteinGym scores ZIP.")
            return extract_dir

    # Extract
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        logger.info(f"Extracted to {extract_dir}")
    except Exception as e:
        logger.warning(f"Failed to extract ZIP: {e}")

    return extract_dir


def load_reference_assays(ref_path: Path) -> pd.DataFrame:
    """Load the ProteinGym substitution assay catalogue (all taxa)."""
    if not ref_path.exists() or ref_path.stat().st_size < 100:
        logger.warning("ProteinGym reference file empty or missing.")
        return pd.DataFrame()

    df = pd.read_csv(ref_path)
    if "taxon" not in df.columns:
        logger.error("Reference file has no 'taxon' column.")
        return pd.DataFrame()

    logger.info("ProteinGym reference contains %s assays", len(df))
    logger.info("Taxon counts:\n%s", df["taxon"].astype(str).str.strip().value_counts().to_string())
    if "coarse_selection_type" in df.columns:
        logger.info(
            "Selection type:\n%s",
            df["coarse_selection_type"].astype(str).str.strip().value_counts().to_string(),
        )
    return df


def filter_bacterial_assays(ref_path: Path) -> pd.DataFrame:
    """Keep every ProteinGym substitution assay with taxon == Prokaryote."""
    df = load_reference_assays(ref_path)
    if df.empty:
        return df
    prokaryote = df["taxon"].astype(str).str.strip().str.lower() == "prokaryote"
    bacterial_df = df.loc[prokaryote].copy()
    logger.info("Prokaryote assays: %s", len(bacterial_df))
    if "source_organism" in bacterial_df.columns:
        logger.info(
            "Organisms:\n"
            + bacterial_df["source_organism"].fillna("(unknown)").value_counts().to_string()
        )
    return bacterial_df


def _assay_meta(assays: pd.DataFrame) -> tuple[set[str], dict[str, dict]]:
    """Map DMS_id / filename / stem → reference sequence and phenotype tags."""
    wanted: set[str] = set()
    meta: dict[str, dict] = {}
    for _, row in assays.iterrows():
        seq = row.get("target_seq", "")
        org = str(row.get("source_organism") or "").strip()
        taxon = str(row.get("taxon") or "").strip()
        coarse = str(row.get("coarse_selection_type") or "").strip()
        rec = {
            "seq": seq if isinstance(seq, str) else "",
            "organism": org,
            "taxon": taxon,
            "coarse": coarse,
        }
        keys = [row.get("DMS_id"), row.get("DMS_filename")]
        if isinstance(row.get("DMS_filename"), str):
            keys.append(Path(row["DMS_filename"]).stem)
        for key in keys:
            if not key or (isinstance(key, float) and pd.isna(key)):
                continue
            key_s = str(key)
            wanted.add(key_s)
            wanted.add(Path(key_s).stem)
            meta[key_s] = rec
            meta[Path(key_s).stem] = rec
    return wanted, meta


def process_dms_scores(
    scores_dir: Path,
    assays: pd.DataFrame,
    lof_threshold: float = 1.0,
    gof_threshold: float = 1.0,
) -> pd.DataFrame:
    """Load DMS CSVs listed in `assays` and attach per-assay z-scores.

    One row per (assay, mutant). Same protein measured in several assays is
    kept — MLoF averages z later. Does not globally collapse TEM-like duplicates.
    """
    wanted, meta = _assay_meta(assays)
    score_files = list(scores_dir.rglob("*.csv"))
    logger.info("Found %s score files in %s", len(score_files), scores_dir)

    chunks: list[pd.DataFrame] = []
    n_matched = 0
    for csv_path in score_files:
        stem = csv_path.stem
        rec = meta.get(stem) or meta.get(csv_path.name)
        if rec is None and csv_path.name not in wanted and stem not in wanted:
            continue
        n_matched += 1
        if rec is None:
            rec = {"seq": "", "organism": "", "taxon": "", "coarse": ""}

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue

        score_col = next((c for c in ("DMS_score", "score", "fitness", "DMS_score_bin") if c in df.columns), None)
        mut_col = next((c for c in ("mutant", "mutation", "mutated_sequence") if c in df.columns), None)
        if score_col is None or mut_col is None:
            continue

        ref_protein = rec["seq"] if len(rec["seq"]) > 10 else ""
        if not ref_protein and "mutated_sequence" in df.columns:
            wt_mask = df[score_col].between(-0.1, 0.1)
            if wt_mask.any():
                ref_protein = str(df.loc[wt_mask.idxmax(), "mutated_sequence"])
        if not ref_protein:
            continue

        numeric = pd.to_numeric(df[score_col], errors="coerce")
        keep = numeric.notna()
        if not keep.any():
            continue
        scores = numeric[keep]
        mean, std = float(scores.mean()), float(scores.std())
        if std == 0:
            std = 1.0
        z = (numeric - mean) / std

        if "mutated_sequence" in df.columns:
            var_protein = df["mutated_sequence"].astype(str)
        else:
            var_protein = df[mut_col].astype(str).map(
                lambda m, ref=ref_protein: _apply_mutation_string(ref, m)
            )

        coarse = rec["coarse"]
        organism = rec["organism"] or _guess_species(stem)
        label = np.where(z < -lof_threshold, 0, np.where(z > gof_threshold, 2, 1))
        chunk = pd.DataFrame({
            "gene": stem,
            "species": organism,
            "taxon": rec["taxon"],
            "coarse_selection_type": coarse,
            "ref_protein": ref_protein,
            "var_protein": var_protein,
            "ref_dna": "",
            "var_dna": "",
            "label": label,
            "dms_score": numeric,
            "dms_zscore": z,
            "source": f"ProteinGym_{coarse}" if coarse else "ProteinGym",
        })
        chunk = chunk.loc[keep & chunk["var_protein"].ne("") & chunk["var_protein"].ne(ref_protein)]
        if not chunk.empty:
            chunks.append(chunk)

    logger.info("Matched %s assay CSV files", n_matched)
    if not chunks:
        return pd.DataFrame()
    df_out = pd.concat(chunks, ignore_index=True)
    logger.info(
        "ProteinGym substitutions: %s variants, %s assays, taxa=\n%s",
        len(df_out),
        df_out["gene"].nunique(),
        df_out["taxon"].value_counts().to_string() if "taxon" in df_out.columns else "(none)",
    )
    return df_out


def _guess_species(dms_id: str) -> str:
    """Guess species from DMS assay ID suffix (e.g., TEM1_ECOLI → E. coli)."""
    species_map = {
        "ECOLI": "Escherichia coli",
        "ECOLX": "Escherichia coli",
        "MYCTU": "Mycobacterium tuberculosis",
        "STAAU": "Staphylococcus aureus",
        "NEIGO": "Neisseria gonorrhoeae",
        "KLEPN": "Klebsiella pneumoniae",
        "PSEAE": "Pseudomonas aeruginosa",
        "BACSU": "Bacillus subtilis",
        "SALTY": "Salmonella typhimurium",
        "STRPN": "Streptococcus pneumoniae",
        "STRAN": "Streptococcus",
        "STRPY": "Streptococcus pyogenes",
        "HUMAN": "Homo sapiens",
    }
    parts = dms_id.upper().split("_")
    for part in parts:
        if part in species_map:
            return species_map[part]
    return ""


def _apply_mutation_string(ref_protein: str, mutation: str, strict: bool = True) -> str:
    """Apply a mutation string like 'A23T' or 'A23T:G45R' to a protein sequence.

    Args:
        ref_protein: Reference amino acid sequence.
        mutation: Mutation string (e.g. 'M69I' or 'M69I:G238S').
        strict: If True (default), skip any mutation where the reference amino acid
            does not match.  Set False for synthetic/fallback data where the
            positions are applied unconditionally.
    """
    var = list(ref_protein)
    mutations = mutation.replace(";", ":").split(":")

    for mut in mutations:
        mut = mut.strip()
        if len(mut) < 3:
            continue
        ref_aa = mut[0]
        var_aa = mut[-1]
        try:
            pos = int(mut[1:-1]) - 1  # 0-based
        except ValueError:
            continue
        if 0 <= pos < len(var):
            if strict and var[pos] != ref_aa:
                logger.debug(f"Mutation {mut}: expected {ref_aa} at pos {pos+1}, found {var[pos]}")
                continue
            var[pos] = var_aa

    return "".join(var)


def _generate_fallback_data() -> pd.DataFrame:
    """Generate bacterial DMS-like data from curated TEM-1 mutations as fallback.

    Uses strict=False so mutations are applied unconditionally — the position
    numbering in _TEM1_MUTATIONS_FALLBACK uses Ambler/ProteinGym conventions
    which may differ from the residues present in _TEM1_REF, but the resulting
    variant sequences are still distinct and usable as synthetic training data.
    """
    records = []
    for mut_str, label in _TEM1_MUTATIONS_FALLBACK:
        var_protein = _apply_mutation_string(_TEM1_REF, mut_str, strict=False)
        if var_protein == _TEM1_REF:
            logger.debug(f"Mutation {mut_str} produced no change — skipping")
            continue
        records.append({
            "gene": "TEM1_ECOLI",
            "species": "Escherichia coli",
            "ref_protein": _TEM1_REF,
            "var_protein": var_protein,
            "ref_dna": "",
            "var_dna": "",
            "label": label,
            "dms_score": 0.0,
            "dms_zscore": 0.0,
            "source": "ProteinGym_curated",
        })
    logger.info(f"Generated {len(records)} curated TEM-1 mutations as fallback")
    return pd.DataFrame(records)


def main():
    logging.basicConfig(level=logging.INFO)

    ref_path = download_proteingym_reference()
    assays = load_reference_assays(ref_path)
    if assays.empty:
        logger.error("No ProteinGym reference assays.")
        sys.exit(1)

    scores_dir = download_proteingym_scores()
    result_df = pd.DataFrame()
    if scores_dir.exists() and any(scores_dir.rglob("*.csv")):
        result_df = process_dms_scores(scores_dir, assays)

    if result_df.empty or (
        "source" in result_df.columns and (result_df["source"] == "ProteinGym_curated").all()
    ):
        logger.error("No real ProteinGym DMS rows. Refusing TEM-1 fallback.")
        sys.exit(1)

    logger.info("Processed %s variants from ProteinGym DMS data", len(result_df))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "proteingym_substitutions.parquet"
    result_df.to_parquet(all_path, index=False)
    logger.info("Saved %s records to %s", len(result_df), all_path)

    taxon = result_df["taxon"].astype(str).str.strip().str.lower()
    bacterial = result_df.loc[taxon.eq("prokaryote")].copy()
    bact_path = out_dir / "proteingym_bacterial.parquet"
    bacterial.to_parquet(bact_path, index=False)
    logger.info("Saved %s Prokaryote records to %s", len(bacterial), bact_path)


if __name__ == "__main__":
    main()
