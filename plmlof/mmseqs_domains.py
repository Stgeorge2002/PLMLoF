"""Approximate domain spans via MMseqs2 search against Pfam seed sequences.

Good enough for LoF synthetic priors (domain geometry), far faster than HMMER.
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

from plmlof.hmmer import write_fasta

logger = logging.getLogger(__name__)

_PFAM_ACC = re.compile(r"(PF\d{5})")


def _open_text(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".fasta.gz") or path.name.endswith(".faa.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def pfam_accession(target_id: str) -> str:
    """Pull PF##### from a Pfam-A.fasta header / mmseqs target id."""
    m = _PFAM_ACC.search(target_id)
    return m.group(1) if m else target_id.split()[0][:64]


def ensure_plain_fasta(path: Path, dest: Path | None = None) -> Path:
    """MMseqs prefers plain FASTA; gunzip once next to the download."""
    path = Path(path)
    if not str(path).endswith(".gz"):
        return path
    out = dest if dest is not None else path.with_suffix("")  # .fasta.gz → .fasta (only one suffix)
    # Path.with_suffix on .fasta.gz only strips .gz if name is *.fasta.gz → use manual
    if path.name.endswith(".fasta.gz"):
        out = path.with_name(path.name[: -len(".gz")])
    elif path.name.endswith(".faa.gz"):
        out = path.with_name(path.name[: -len(".gz")])
    else:
        out = path.with_suffix("")
    if out.exists() and out.stat().st_size > 1_000_000:
        logger.debug("Using uncompressed FASTA %s", out)
        return out
    logger.info("Decompressing %s → %s", path, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return out


def annotate_proteins_mmseqs(
    pairs: list[tuple[str, str]],
    target_fasta: Path,
    *,
    threads: int = 20,
    max_evalue: float = 1e-5,
    sensitivity: float = 5.0,
    work_dir: Path | None = None,
) -> list[dict]:
    """Search query proteins against Pfam-A (or any) FASTA; return domain-like hits.

    Each hit: protein_id, hmm_name (Pfam acc), evalue, score (bits), start, end
    on the **query** (1-based inclusive).
    """
    if not pairs:
        return []
    if shutil.which("mmseqs") is None:
        raise FileNotFoundError(
            "mmseqs not on PATH. Install with:  conda install -c bioconda mmseqs2"
        )
    target_fasta = ensure_plain_fasta(Path(target_fasta))
    if not target_fasta.exists():
        raise FileNotFoundError(target_fasta)

    own_tmp = work_dir is None
    root = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="plmlof_mmseqs_dom_"))
    root.mkdir(parents=True, exist_ok=True)
    query_fa = root / "query.faa"
    result = root / "hits.tsv"
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    write_fasta(pairs, query_fa)

    cmd = [
        "mmseqs", "easy-search",
        str(query_fa), str(target_fasta), str(result), str(tmp),
        "--threads", str(max(1, threads)),
        "-e", str(max_evalue),
        "-s", str(sensitivity),
        "--max-seqs", "50",
        "--format-output", "query,target,qstart,qend,evalue,bits",
        "-v", "1",
    ]
    logger.info(
        "MMseqs2 easy-search: %s queries × %s (threads=%s, e≤%g)",
        len(pairs), target_fasta.name, threads, max_evalue,
    )
    subprocess.run(cmd, check=True)

    if not result.exists():
        # easy-search sometimes adds .m8
        alt = Path(str(result) + ".m8")
        if alt.exists():
            result = alt
        else:
            raise FileNotFoundError(f"MMseqs result missing: {result}")

    rows: list[dict] = []
    with result.open("rt", encoding="utf-8", errors="replace") as fh:
        for line in tqdm(fh, desc="Parse hits", unit="line", dynamic_ncols=True):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            query, target, qstart, qend, evalue, bits = parts[:6]
            try:
                start = int(float(qstart))
                end = int(float(qend))
                ev = float(evalue)
                score = float(bits)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            if start < 1 or end < 1:
                continue
            rows.append({
                "protein_id": query,
                "hmm_name": pfam_accession(target),
                "hmm_acc": pfam_accession(target),
                "evalue": ev,
                "score": score,
                "start": start,
                "end": end,
            })

    logger.info("MMseqs raw hits: %s", len(rows))
    if own_tmp:
        shutil.rmtree(root, ignore_errors=True)
    return rows
