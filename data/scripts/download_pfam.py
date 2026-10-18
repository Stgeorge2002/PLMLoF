"""Download Pfam libraries onto a laptop (not Isambard).

Default is Pfam-A **seed** (~185 MB Stockholm) → plain FASTA for MMseqs:

    python data/scripts/download_pfam.py
    # writes data/raw/pfam/Pfam-A.seed.gz + data/raw/pfam/Pfam-A.seed.faa

Full NR90 FASTA is huge (~5.8 GB) — avoid unless you need it:

    python data/scripts/download_pfam.py --kind fasta

Optional HMM library (HMMER, ~400 MB):

    python data/scripts/download_pfam.py --kind hmm
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PFAM_SEED_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz"
PFAM_FASTA_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.fasta.gz"
PFAM_HMM_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
MIN_SEED_BYTES = 50_000_000
MIN_FASTA_BYTES = 1_000_000_000
MIN_HMM_BYTES = 50_000_000
_PFAM_AC = re.compile(r"^#=GF\s+AC\s+(PF\d+)")


def _ssl() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download(url: str, dest: Path, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "PLMLoF/2.0"})
    logger.info("Downloading %s → %s", url, dest)
    with urlopen(req, timeout=timeout, context=_ssl()) as resp:  # noqa: S310
        with part.open("wb") as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    part.replace(dest)
    logger.info("Wrote %.1f MB", dest.stat().st_size / 1e6)


def stockholm_seed_to_fasta(seed_gz: Path, fasta_out: Path) -> int:
    """Convert Pfam-A.seed.gz (Stockholm) → ungapped FASTA for MMseqs.

    Headers: ``>{seq_id} {PF#####}`` so domain accession survives search.
    """
    n = 0
    acc = "PF00000"
    fasta_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(seed_gz, "rt", encoding="utf-8", errors="replace") as src, fasta_out.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.startswith("#=GF AC"):
                m = _PFAM_AC.match(line.strip())
                if m:
                    acc = m.group(1)
                continue
            if not line or line[0] in {"#", "/"}:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            seq_id, aln = parts[0], parts[1]
            seq = aln.replace(".", "").replace("-", "").upper()
            if not seq:
                continue
            # Accession first so MMseqs target id keeps PF##### (ids are first token).
            dst.write(f">{acc}|{seq_id}\n")
            for i in range(0, len(seq), 80):
                dst.write(seq[i:i + 80] + "\n")
            n += 1
    logger.info("Wrote %s seed sequences → %s (%.1f MB)", n, fasta_out, fasta_out.stat().st_size / 1e6)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Download Pfam-A seed / FASTA / HMM")
    p.add_argument(
        "--kind",
        choices=("seed", "fasta", "hmm"),
        default="seed",
        help="seed ≈185MB→FASTA for MMseqs (default); fasta ≈5.8GB NR90; hmm ≈400MB",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--url", type=str, default=None)
    args = p.parse_args()

    if args.kind == "seed":
        out = args.out or Path("data/raw/pfam/Pfam-A.seed.gz")
        url = args.url or PFAM_SEED_URL
        min_bytes = MIN_SEED_BYTES
    elif args.kind == "fasta":
        out = args.out or Path("data/raw/pfam/Pfam-A.fasta.gz")
        url = args.url or PFAM_FASTA_URL
        min_bytes = MIN_FASTA_BYTES
    else:
        out = args.out or Path("data/raw/pfam/Pfam-A.hmm.gz")
        url = args.url or PFAM_HMM_URL
        min_bytes = MIN_HMM_BYTES

    if not (out.exists() and out.stat().st_size >= min_bytes):
        download(url, out)
        if out.stat().st_size < min_bytes:
            raise SystemExit(f"Download looks too small: {out}")
    else:
        logger.info("Already present (%s bytes): %s", out.stat().st_size, out)

    if args.kind == "seed":
        faa = out.with_name("Pfam-A.seed.faa")
        if faa.exists() and faa.stat().st_size > 10_000_000:
            logger.info("Seed FASTA already present: %s", faa)
        else:
            stockholm_seed_to_fasta(out, faa)


if __name__ == "__main__":
    main()
