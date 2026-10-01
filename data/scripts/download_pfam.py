"""Download Pfam-A HMM library onto a laptop (not Isambard).

    python data/scripts/download_pfam.py --out data/raw/pfam/Pfam-A.hmm.gz

~1 GB compressed. pyhmmer reads the gz directly. hmmscan needs gunzip + hmmpress.
"""

from __future__ import annotations

import argparse
import logging
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PFAM_HMM_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
MIN_BYTES = 50_000_000


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Download Pfam-A.hmm.gz")
    p.add_argument("--out", type=Path, default=Path("data/raw/pfam/Pfam-A.hmm.gz"))
    p.add_argument("--url", type=str, default=PFAM_HMM_URL)
    args = p.parse_args()

    if args.out.exists() and args.out.stat().st_size >= MIN_BYTES:
        logger.info("Already present (%s bytes): %s", args.out.stat().st_size, args.out)
        return
    download(args.url, args.out)
    if args.out.stat().st_size < MIN_BYTES:
        raise SystemExit(f"Download looks too small: {args.out}")


if __name__ == "__main__":
    main()
