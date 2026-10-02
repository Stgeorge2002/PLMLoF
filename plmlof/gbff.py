"""Parse length-filtered CDS from bacterial GBFFs."""

from __future__ import annotations

import gzip
from pathlib import Path

from Bio import SeqIO

from plmlof.utils.sequence_utils import translate_dna

AA = "ACDEFGHIKLMNPQRSTVWY"
MIN_AA = 80
MAX_AA = 1024


def open_gbff(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".gbff.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_cds(gbff: Path) -> list[tuple[str, str, str, str]]:
    """Return (gene, species, protein, dna) for length-filtered CDS."""
    out: list[tuple[str, str, str, str]] = []
    species = gbff.stem
    try:
        handle = open_gbff(gbff)
    except OSError:
        return out
    with handle:
        for rec in SeqIO.parse(handle, "genbank"):
            org = rec.annotations.get("organism") or species
            species = str(org)
            for feat in rec.features:
                if feat.type != "CDS":
                    continue
                qual = feat.qualifiers
                gene = (qual.get("gene") or qual.get("locus_tag") or ["unknown"])[0]
                dna = str(feat.extract(rec.seq)).upper().replace("U", "T")
                if len(dna) < 3:
                    continue
                prot = str(qual.get("translation", [""])[0] or "")
                if not prot:
                    prot = translate_dna(dna, to_stop=True)
                prot = prot.replace("*", "")
                if not (MIN_AA <= len(prot) <= MAX_AA):
                    continue
                if not set(prot) <= set(AA + "X"):
                    continue
                out.append((str(gene), species, prot, dna))
    return out
