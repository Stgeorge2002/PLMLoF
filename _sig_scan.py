from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from plmlof.lof_import import match_reference
from plmlof.mutcode import parse_codes, read_fasta

root = Path("PRJNA1078342_results")
all_seqs: dict[str, dict[str, str]] = {
    "chromosome": read_fasta(root / "rustaroo/chromosome/rustroo/genes.fa"),
}
for d in sorted((root / "rustaroo/plasmids").glob("mge_*")):
    fa = d / "rustroo/genes.fa"
    if fa.exists():
        all_seqs[d.name] = read_fasta(fa)

sig_pat = re.compile(
    r"bla|oxa|shv|tem|ctx|ndm|kpc|vim|imp|ampC|aac|aad|aph|armA|rmt|qnr|oqx|"
    r"sul|dfr|tet|cat|cml|flo|mph|erm|mcr|van|mec|fos|arr|"
    r"ompK|ompF|acr|tolC|ramA|marA|soxS|"
    r"fimH|mrk|iuc|iro|rmpA|rmpC|clb|magA|wzi|"
    r"beta-lactam|carbapenem|aminoglycoside|fluoroquinolone|colistin",
    re.I,
)
hot_pat = re.compile(
    r"\b(bla|oxa|shv|tem|ctx|ndm|kpc|vim|imp|aac|aad|aph|qnr|mcr|tet|sul|dfr|"
    r"ompK|acr|rmpA|iuc|iro|Beta-lactamase)\b",
    re.I,
)

missing: Counter[str] = Counter()
sig_miss: Counter[tuple[str, str, str]] = Counter()
remap_hot: dict[tuple[str, str, str], tuple[int, int | None]] = {}
paralog_hot: dict[tuple[str, str, str], tuple[int, int, int | None]] = {}

with (root / "rustaroo_lof_matrix.tsv").open(newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        gene = row["Gene"].strip()
        desc = row["Description"].strip()
        replicon = row["Replicon"].strip()
        key = "chromosome" if replicon == "chromosome" else replicon.split("__", 1)[0]
        seqmap = all_seqs.get(key)
        if seqmap is None:
            continue
        matrix_ref_len = None
        if row.get("Reference_Length") not in (None, ""):
            matrix_ref_len = int(float(row["Reference_Length"]))
        edits = parse_codes(row["Mutation_Codes"])
        min_end = max((e.start + e.delete_length for e in edits), default=0)
        hit = match_reference(
            seqmap,
            gene,
            desc,
            matrix_ref_len=matrix_ref_len,
            min_end=min_end,
        )
        text = f"{gene} {desc}"
        is_sig = bool(sig_pat.search(text))
        is_hot = bool(hot_pat.search(text))
        if hit is None:
            missing[f"{key}:{gene}|{desc[:70]}"] += 1
            if is_sig:
                sig_miss[(key, gene, desc)] += 1
            continue
        header, seq = hit
        if header != gene and is_hot:
            remap_hot[(gene, header, desc)] = (len(seq), matrix_ref_len)
        if gene in seqmap and header != gene and is_hot:
            paralog_hot[(gene, header, desc)] = (
                len(seqmap[gene]),
                len(seq),
                matrix_ref_len,
            )

print("skipped significant gene/desc (all)")
print("unique", len(sig_miss), "rows", sum(sig_miss.values()))
for (k, g, d), n in sig_miss.most_common(40):
    print(f"  {n:4d}  {k}:{g}  |  {d}")

print("\nhot remaps (bla/aac/omp/etc) unique gene->FASTA")
for (g, h, d), (L, mrl) in sorted(remap_hot.items(), key=lambda x: x[0][0])[:50]:
    print(f"  {g} -> {h}  len={L} matrix_ref={mrl}  | {d[:75]}")

print("\nhot paralog switches")
for (g, h, d), (l1, l2, mrl) in sorted(paralog_hot.items()):
    print(f"  {g}({l1}) -> {h}({l2}) matrix_ref={mrl}  | {d[:70]}")

print("\nall missing keys")
for item, n in missing.most_common():
    print(f"  {n:4d}  {item}")
