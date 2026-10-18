"""Pfam annotation backends (pyhmmer, else hmmscan CLI). Laptop-only."""

from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from plmlof.domains import protein_id

logger = logging.getLogger(__name__)

DOM_EVALUE = 1e-5
MIN_HMM_BYTES = 50_000_000
SCAN_CHUNK = 250


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def filter_high_confidence(
    hits: list[dict],
    *,
    max_evalue: float = DOM_EVALUE,
) -> list[dict]:
    """Keep domain hits at or under the e-value threshold."""
    out: list[dict] = []
    for hit in hits:
        try:
            if float(hit["evalue"]) <= max_evalue:
                out.append(hit)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def resolve_nonoverlapping_hits(hits: list[dict]) -> list[dict]:
    """Per protein: greedy keep highest-score domains that do not overlap.

    Sort by score ↓, e-value ↑, start ↑. A candidate is kept only if its
    envelope does not overlap any already accepted hit on that protein.
    """
    by_prot: dict[str, list[dict]] = {}
    for hit in hits:
        pid = str(hit.get("protein_id") or "")
        if not pid:
            continue
        by_prot.setdefault(pid, []).append(hit)

    resolved: list[dict] = []
    for pid, group in by_prot.items():
        group = sorted(
            group,
            key=lambda h: (-float(h.get("score", 0.0)), float(h.get("evalue", 1.0)), int(h["start"])),
        )
        kept: list[dict] = []
        kept_spans: list[tuple[int, int]] = []
        for hit in group:
            try:
                start = int(hit["start"])
                end = int(hit["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end < start:
                continue
            if any(not (end < ks or start > ke) for ks, ke in kept_spans):
                continue
            kept.append(hit)
            kept_spans.append((start, end))
        kept.sort(key=lambda h: int(h["start"]))
        resolved.extend(kept)
    return resolved


def finalize_domain_hits(
    hits: list[dict],
    *,
    max_evalue: float = DOM_EVALUE,
) -> list[dict]:
    """High-confidence, non-overlapping domain hits ready for parquet."""
    return resolve_nonoverlapping_hits(filter_high_confidence(hits, max_evalue=max_evalue))


def _uncompressed_hmm(hmm: Path) -> Path:
    """hmmscan/hmmpress cannot read .gz. Decompress once next to the download."""
    name = hmm.name
    if name.endswith(".hmm.gz"):
        dest = hmm.with_name(name[:-3])
    elif hmm.suffix == ".gz":
        dest = hmm.with_suffix("")
    else:
        return hmm
    if dest.exists() and dest.stat().st_size >= MIN_HMM_BYTES:
        logger.debug("Using uncompressed HMM %s", dest)
        return dest
    logger.info("Decompressing %s → %s (one-time, several GB)", hmm, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(hmm, "rb") as src, dest.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return dest


def _pressed_hmm(hmm: Path) -> Path:
    plain = _uncompressed_hmm(hmm)
    marker = Path(str(plain) + ".h3m")
    if marker.exists():
        return plain
    press = shutil.which("hmmpress")
    if not press:
        raise FileNotFoundError(
            "hmmpress not on PATH. Either install HMMER 3, or:  pip install pyhmmer"
        )
    logger.info("hmmpress %s (one-time)", plain)
    subprocess.run([press, str(plain)], check=True)
    return plain


def write_fasta(pairs: list[tuple[str, str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for name, seq in pairs:
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i:i + 80] + "\n")


def parse_domtblout(path: Path) -> list[dict]:
    """Parse HMMER 3 `--domtblout` (hmmscan: query is the protein)."""
    rows: list[dict] = []
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 22:
                continue
            try:
                rows.append({
                    "protein_id": parts[3],
                    "hmm_name": parts[0],
                    "hmm_acc": parts[1],
                    "evalue": float(parts[12]),
                    "score": float(parts[13]),
                    "start": int(parts[19]),
                    "end": int(parts[20]),
                })
            except (ValueError, IndexError):
                continue
    return rows


def _hmmscan_one(fasta: Path, db: Path, cpus: int) -> list[dict]:
    exe = shutil.which("hmmscan")
    if not exe:
        raise FileNotFoundError("hmmscan not on PATH")
    with tempfile.NamedTemporaryFile(prefix="plmlof_domtbl_", suffix=".txt", delete=False) as tmp:
        domtbl = Path(tmp.name)
    cmd = [
        exe, "--cpu", str(max(1, cpus)), "--cut_ga", "--noali",
        "--domtblout", str(domtbl), str(db), str(fasta),
    ]
    try:
        # Swallow hmmscan's own progress spam; we show tqdm instead.
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return parse_domtblout(domtbl)
    finally:
        if domtbl.exists():
            domtbl.unlink()


def _scan_hmmscan(
    pairs: list[tuple[str, str]],
    hmm: Path,
    cpus: int,
    *,
    show_progress: bool = True,
    desc: str = "HMMER",
) -> list[dict]:
    """Chunked hmmscan CLI with a gene-count progress bar."""
    db = _pressed_hmm(hmm)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="plmlof_hmm_") as tmp:
        tmp_dir = Path(tmp)
        pbar_ctx = (
            tqdm(total=len(pairs), desc=desc, unit="gene", dynamic_ncols=True)
            if show_progress
            else None
        )
        try:
            for i in range(0, len(pairs), SCAN_CHUNK):
                batch = pairs[i:i + SCAN_CHUNK]
                fasta = tmp_dir / f"batch_{i}.faa"
                write_fasta(batch, fasta)
                rows.extend(_hmmscan_one(fasta, db, cpus))
                if pbar_ctx is not None:
                    pbar_ctx.update(len(batch))
        finally:
            if pbar_ctx is not None:
                pbar_ctx.close()
    return rows


def _collect_pyhmmer_hits(hit_block, valid: set[str]) -> list[dict]:
    rows: list[dict] = []
    if hasattr(hit_block, "query") and hasattr(hit_block.query, "name"):
        query = _as_text(hit_block.query.name)
    else:
        query = _as_text(getattr(hit_block, "query_name", ""))
    for hit in hit_block:
        if hasattr(hit, "included") and not hit.included:
            continue
        hmm_name = _as_text(hit.name)
        protein = query
        if protein not in valid and hmm_name in valid:
            protein, hmm_name = hmm_name, protein
        if protein not in valid:
            continue
        for domain in hit.domains:
            if hasattr(domain, "included") and not domain.included:
                continue
            evalue = float(getattr(domain, "i_evalue", getattr(domain, "evalue", 0.0)))
            start = int(getattr(domain, "env_from", domain.alignment.target_from))
            end = int(getattr(domain, "env_to", domain.alignment.target_to))
            if end < start:
                continue
            rows.append({
                "protein_id": protein,
                "hmm_name": hmm_name,
                "hmm_acc": "",
                "evalue": evalue,
                "score": float(getattr(domain, "score", 0.0)),
                "start": start,
                "end": end,
            })
    return rows


def _scan_pyhmmer(
    pairs: list[tuple[str, str]],
    hmm: Path,
    cpus: int,
    *,
    show_progress: bool = True,
    desc: str = "HMMER",
) -> list[dict]:
    import pyhmmer

    alphabet = pyhmmer.easel.Alphabet.amino()
    valid = {name for name, seq in pairs if seq}
    seqs = [
        pyhmmer.easel.TextSequence(name=name.encode("ascii", "replace"), sequence=seq).digitize(alphabet)
        for name, seq in pairs
        if seq
    ]
    if not seqs:
        return []

    logger.info("Loading HMM profiles from %s", hmm)
    with pyhmmer.plan7.HMMFile(str(hmm)) as handle:
        profiles = list(handle)
    logger.info("Scanning %s genes × %s Pfam profiles (cpus=%s)", len(seqs), len(profiles), cpus)

    def _iter(batch):
        kwargs = {"cpus": max(1, cpus)}
        try:
            return pyhmmer.hmmscan(batch, profiles, bit_cutoffs="gathering", **kwargs)
        except TypeError:
            try:
                return pyhmmer.hmmscan(profiles, batch, bit_cutoffs="gathering", **kwargs)
            except TypeError:
                return pyhmmer.hmmscan(batch, profiles, **kwargs)

    rows: list[dict] = []
    pbar_ctx = (
        tqdm(total=len(seqs), desc=desc, unit="gene", dynamic_ncols=True)
        if show_progress
        else None
    )
    try:
        for i in range(0, len(seqs), SCAN_CHUNK):
            batch = seqs[i:i + SCAN_CHUNK]
            for hit_block in _iter(batch):
                rows.extend(_collect_pyhmmer_hits(hit_block, valid))
            if pbar_ctx is not None:
                pbar_ctx.update(len(batch))
    finally:
        if pbar_ctx is not None:
            pbar_ctx.close()
    return rows


def annotate_proteins(
    pairs: list[tuple[str, str]],
    hmm: Path,
    cpus: int = 20,
    *,
    backend: str = "auto",
    show_progress: bool = True,
    desc: str = "HMMER",
) -> list[dict]:
    """pairs: (protein_id, sequence). ``backend``: auto | pyhmmer | hmmscan."""
    if not pairs:
        return []
    backend = (backend or "auto").lower()
    if backend not in {"auto", "pyhmmer", "hmmscan"}:
        raise ValueError(f"unknown HMMER backend: {backend}")

    if backend in {"auto", "pyhmmer"}:
        try:
            import pyhmmer  # noqa: F401
            return _scan_pyhmmer(
                pairs, hmm, cpus, show_progress=show_progress, desc=desc,
            )
        except ImportError:
            if backend == "pyhmmer":
                raise
            logger.info("pyhmmer not installed; trying hmmscan CLI")
        except Exception as exc:
            if backend == "pyhmmer":
                raise
            logger.warning("pyhmmer scan failed (%s); trying hmmscan CLI", exc)

    return _scan_hmmscan(pairs, hmm, cpus, show_progress=show_progress, desc=desc)


def _shard_slices(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) slices covering ``n_items``."""
    n_shards = max(1, int(n_shards))
    if n_items <= 0:
        return []
    n_shards = min(n_shards, n_items)
    base, rem = divmod(n_items, n_shards)
    slices: list[tuple[int, int]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        end = start + size
        slices.append((start, end))
        start = end
    return slices


def _annotate_shard_worker(
    payload: tuple[int, list[tuple[str, str]], str, int],
) -> tuple[int, int, list[dict]]:
    """Process entrypoint: (shard_id, pairs, hmm_path, cpus) → hits."""
    shard_id, pairs, hmm_path, cpus = payload
    hits = annotate_proteins(
        pairs,
        Path(hmm_path),
        cpus=cpus,
        backend="hmmscan",  # shared pressed DB; avoids 20× Pfam RAM with pyhmmer
        show_progress=False,
    )
    return shard_id, len(pairs), hits


def annotate_proteins_sharded(
    pairs: list[tuple[str, str]],
    hmm: Path,
    *,
    shards: int = 20,
    cpus_per_shard: int = 1,
    genes_per_task: int = 500,
) -> list[dict]:
    """Fan out genes across ``shards`` worker processes (default 1 CPU each).

    Work is split into small tasks (``genes_per_task``) so the progress bar
    updates often enough for tqdm to show gene/s and ETA.
    """
    if not pairs:
        return []
    shards = max(1, int(shards))
    cpus_per_shard = max(1, int(cpus_per_shard))
    genes_per_task = max(50, int(genes_per_task))
    if shards == 1:
        return annotate_proteins(
            pairs, hmm, cpus=cpus_per_shard, backend="hmmscan", show_progress=True,
        )

    # Press once in the parent so workers only read the DB.
    db = _pressed_hmm(Path(hmm))
    n_tasks = (len(pairs) + genes_per_task - 1) // genes_per_task
    logger.info(
        "Sharded HMMER: %s genes → %s workers × %s CPU, %s tasks of ≤%s genes (db=%s)",
        len(pairs), shards, cpus_per_shard, n_tasks, genes_per_task, db,
    )

    payloads = [
        (i, pairs[start:end], str(Path(hmm)), cpus_per_shard)
        for i, (start, end) in enumerate(_shard_slices(len(pairs), n_tasks))
    ]

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=shards) as pool:
        futures = [pool.submit(_annotate_shard_worker, p) for p in payloads]
        with tqdm(
            total=len(pairs),
            desc="HMMER",
            unit="gene",
            dynamic_ncols=True,
            miniters=1,
            smoothing=0.05,
        ) as pbar:
            for fut in as_completed(futures):
                _shard_id, n_done, hits = fut.result()
                rows.extend(hits)
                pbar.update(n_done)
    return rows


def unique_protein_pairs(records: list[tuple[str, str, str, str]]) -> list[tuple[str, str]]:
    """Dedup CDS tuples (gene, species, protein, dna) by sequence hash."""
    seen: dict[str, str] = {}
    for _, _, prot, _ in records:
        key = protein_id(prot)
        seen.setdefault(key, prot)
    return list(seen.items())
