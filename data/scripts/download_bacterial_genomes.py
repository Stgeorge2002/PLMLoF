"""Download ~150 representative complete bacterial genomes (GBFF) on a laptop.

Does not require a GPU. Resume-safe: existing files are skipped.

    python data/scripts/download_bacterial_genomes.py --out data/raw/refseq_gbff
"""

from __future__ import annotations

import argparse
import gzip
import logging
import ssl
import sys
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt"

PRIORITY = [
    "klebsiella pneumoniae",
    "streptococcus pneumoniae",
    "escherichia coli",
    "staphylococcus aureus",
    "pseudomonas aeruginosa",
    "mycobacterium tuberculosis",
    "salmonella enterica",
    "acinetobacter baumannii",
    "enterococcus faecium",
    "neisseria gonorrhoeae",
    "haemophilus influenzae",
    "helicobacter pylori",
    "campylobacter jejuni",
    "vibrio cholerae",
    "listeria monocytogenes",
    "clostridioides difficile",
    "bacillus subtilis",
    "corynebacterium diphtheriae",
    "legionella pneumophila",
    "bordetella pertussis",
    "shigella flexneri",
    "yersinia pestis",
    "streptococcus pyogenes",
    "streptococcus agalactiae",
    "enterobacter cloacae",
    "proteus mirabilis",
    "serratia marcescens",
    "stenotrophomonas maltophilia",
    "burkholderia cenocepacia",
    "mycobacterium abscessus",
]


def _ssl() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch(url: str, dest: Path | None = None, timeout: int = 180) -> bytes:
    req = Request(url, headers={"User-Agent": "PLMLoF/2.0"})
    with urlopen(req, timeout=timeout, context=_ssl()) as resp:  # noqa: S310
        body = resp.read()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    return body


def parse_summary(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 21:
            continue
        rows.append({
            "assembly_accession": parts[0],
            "refseq_category": parts[4],
            "taxid": parts[5],
            "species_taxid": parts[6],
            "organism_name": parts[7],
            "assembly_level": parts[11],
            "ftp_path": parts[19],
        })
    return rows


def pick_assemblies(rows: list[dict], n: int) -> list[dict]:
    complete = [
        r for r in rows
        if r["assembly_level"] == "Complete Genome"
        and r["ftp_path"] not in {"", "na"}
    ]
    # Prefer reference / representative genomes.
    def rank(r: dict) -> tuple:
        cat = r["refseq_category"].lower()
        pref = 0 if "reference" in cat else 1 if "representative" in cat else 2
        return (pref, r["organism_name"])

    complete.sort(key=rank)
    by_species: dict[str, dict] = {}
    priority_left = list(PRIORITY)

    def name_key(r: dict) -> str:
        return r["organism_name"].split(",")[0].strip().lower()

    for needle in priority_left:
        for r in complete:
            if needle in name_key(r) and r["species_taxid"] not in by_species:
                by_species[r["species_taxid"]] = r
                break

    for r in complete:
        if len(by_species) >= n:
            break
        by_species.setdefault(r["species_taxid"], r)
    picked = list(by_species.values())[:n]
    logger.info("Selected %s assemblies (%s priority hits)", len(picked), min(len(PRIORITY), len(picked)))
    return picked


def gbff_url(ftp_path: str) -> str:
    ftp = ftp_path.rstrip("/")
    if ftp.startswith("ftp://"):
        ftp = "https://" + ftp[len("ftp://"):]
    name = ftp.rsplit("/", 1)[-1]
    return f"{ftp}/{name}_genomic.gbff.gz"


def download_one(row: dict, out_dir: Path) -> Path | None:
    acc = row["assembly_accession"]
    dest = out_dir / f"{acc}.gbff.gz"
    if dest.exists() and dest.stat().st_size > 1000:
        logger.info("skip existing %s", dest.name)
        return dest
    url = gbff_url(row["ftp_path"])
    try:
        fetch(url, dest, timeout=300)
        logger.info("got %s (%s) %.1f MB", row["organism_name"], acc, dest.stat().st_size / 1e6)
        return dest
    except (URLError, HTTPError, TimeoutError, ssl.SSLError) as exc:
        logger.warning("failed %s %s: %s", acc, url, exc)
        if dest.exists():
            dest.unlink()
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Download representative bacterial GBFFs")
    p.add_argument("--out", type=Path, default=Path("data/raw/refseq_gbff"))
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--summary", type=Path, default=Path("data/raw/assembly_summary_bacteria.txt"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if args.summary.exists() and args.summary.stat().st_size > 10_000:
        text = args.summary.read_text(encoding="utf-8", errors="replace")
        logger.info("Using cached summary %s", args.summary)
    else:
        logger.info("Downloading assembly summary…")
        raw = fetch(SUMMARY_URL, args.summary, timeout=180)
        if raw[:2] == b"\x1f\x8b":
            text = gzip.decompress(raw).decode("utf-8", errors="replace")
            args.summary.write_text(text, encoding="utf-8")
        else:
            text = raw.decode("utf-8", errors="replace")

    rows = parse_summary(text)
    if not rows:
        logger.error("No rows parsed from assembly summary")
        sys.exit(1)
    picked = pick_assemblies(rows, args.n)
    ok = 0
    for i, row in enumerate(picked, 1):
        logger.info("[%s/%s] %s", i, len(picked), row["organism_name"])
        if download_one(row, args.out):
            ok += 1
    logger.info("Downloaded/kept %s / %s GBFFs in %s", ok, len(picked), args.out)
    if ok < 20:
        sys.exit(1)


if __name__ == "__main__":
    main()
