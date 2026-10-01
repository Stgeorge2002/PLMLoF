"""Pfam annotation backends (pyhmmer, else hmmscan CLI). Laptop-only."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from plmlof.v2.domains import protein_id

logger = logging.getLogger(__name__)

# Inclusive domain i-Evalue if gathering cutoffs are unavailable.
DOM_EVALUE = 1e-5


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


def _scan_hmmscan(fasta: Path, hmm: Path, cpus: int) -> list[dict]:
    exe = shutil.which("hmmscan")
    if not exe:
        raise FileNotFoundError("hmmscan not on PATH")
    hmm_arg = str(hmm)
    if hmm.suffix == ".gz":
        raise ValueError("hmmscan needs an uncompressed HMM (gunzip Pfam-A.hmm.gz, then hmmpress)")
    with tempfile.NamedTemporaryFile(prefix="plmlof_domtbl_", suffix=".txt", delete=False) as tmp:
        domtbl = Path(tmp.name)
    cmd = [
        exe, "--cpu", str(max(1, cpus)), "--cut_ga", "--noali",
        "--domtblout", str(domtbl), hmm_arg, str(fasta),
    ]
    logger.info("Running %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return parse_domtblout(domtbl)
    finally:
        if domtbl.exists():
            domtbl.unlink()


def _scan_pyhmmer(pairs: list[tuple[str, str]], hmm: Path, cpus: int) -> list[dict]:
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
    logger.info("Scanning %s proteins against %s profiles (cpus=%s)", len(seqs), len(profiles), cpus)

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
    chunk = 250
    for i in range(0, len(seqs), chunk):
        batch = seqs[i:i + chunk]
        for hits in _iter(batch):
            query = hits.query.name.decode() if hasattr(hits, "query") and hasattr(hits.query, "name") else getattr(hits, "query_name", b"").decode()
            for hit in hits:
                if hasattr(hit, "included") and not hit.included:
                    continue
                hmm_name = hit.name.decode() if isinstance(hit.name, bytes) else str(hit.name)
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
        logger.info("  hmmscan chunk %s/%s", min(i + chunk, len(seqs)), len(seqs))
    return rows


def annotate_proteins(pairs: list[tuple[str, str]], hmm: Path, cpus: int = 4) -> list[dict]:
    """pairs: (protein_id, sequence). Prefers pyhmmer; falls back to hmmscan."""
    if not pairs:
        return []
    try:
        import pyhmmer  # noqa: F401
        return _scan_pyhmmer(pairs, hmm, cpus)
    except ImportError:
        logger.info("pyhmmer not installed; trying hmmscan CLI")
    except Exception as exc:
        logger.warning("pyhmmer scan failed (%s); trying hmmscan CLI", exc)

    with tempfile.TemporaryDirectory(prefix="plmlof_hmm_") as tmp:
        fasta = Path(tmp) / "query.faa"
        write_fasta(pairs, fasta)
        return _scan_hmmscan(fasta, hmm, cpus)


def unique_protein_pairs(records: list[tuple[str, str, str, str]]) -> list[tuple[str, str]]:
    """Dedup CDS tuples (gene, species, protein, dna) by sequence hash."""
    seen: dict[str, str] = {}
    for _, _, prot, _ in records:
        key = protein_id(prot)
        seen.setdefault(key, prot)
    return list(seen.items())
