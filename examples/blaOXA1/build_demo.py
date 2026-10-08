"""Build a 50-allele blaOXA-1 missense ladder (stdlib only).

Paired FASTAs share sample IDs so predict.py can join them. Manifest records
n_missense, severity, and substitutions for scoring MDG against the design.

OXA-1 class D catalytic motif (protein coordinates on *this* CDS, 276 aa):
  S71 / K74 (SxxK), Y144 / G145 (YGN), K215 / T216 (KTG).
  Cysteines are C43 and C63; this allele has no C119.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

CDS = (
    "ATGAAAAACACAATACATATCAACTTCGCTATTTTTTTAATAATTGCAAATATTATCTACAGCAGCGCCAGTGCATCAAC"
    "AGATATCTCTACTGTTGCATCTCCATTATTTGAAGGAACTGAAGGTTGTTTTTTACTTTACGATGCATCCACAAACGCTG"
    "AAATTGCTCAATTCAATAAAGCAAAGTGTGCAACGCAAATGGCACCAGATTCAACTTTCAAGATCGCATTATCACTTATG"
    "GCATTTGATGCGGAAATAATAGATCAGAAAACCATATTCAAATGGGATAAAACCCCCAAAGGAATGGAGATCTGGAACAG"
    "CAATCATACACCAAAGACGTGGATGCAATTTTCTGTTGTTTGGGTTTCGCAAGAAATAACCCAAAAAATTGGATTAAATA"
    "AAATCAAGAATTATCTCAAAGATTTTGATTATGGAAATCAAGACTTCTCTGGAGATAAAGAAAGAAACAACGGATTAACA"
    "GAAGCATGGCTCGAAAGTAGCTTAAAAATTTCACCAGAAGAACAAATTCAATTCCTGCGTAAAATTATTAATCACAATCT"
    "CCCAGTTAAAAACTCAGCCATAGAAAACACCATAGAGAACATGTATCTACAAGATCTGGATAATAGTACAAAACTGTATG"
    "GGAAAACTGGTGCAGGATTCACAGCAAATAGAACCTTACAAAACGGATGGTTTGAAGGGTTTATTATAAGCAAATCAGGA"
    "CATAAATATGTTTTTGTGTCCGCACTTACAGGAAACTTGGGGTCGAATTTAACATCAAGCATAAAAGCCAAGAAAAATGC"
    "GATCACCATTCTAAACACACTAAATTTATAA"
)

CODON = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# 1-based (wt, alt). Conservative: BLOSUM-friendly, away from SxxK / YGN / KTG.
CONS = [
    (5, "I", "V"), (7, "I", "V"), (11, "I", "V"), (14, "I", "V"),
    (13, "L", "I"), (36, "L", "I"), (271, "L", "I"), (276, "L", "I"),
    (4, "T", "S"), (31, "T", "S"), (166, "S", "T"), (167, "S", "T"),
    (10, "A", "G"), (23, "A", "G"), (33, "A", "G"),
    (9, "F", "Y"), (12, "F", "Y"),
    (60, "K", "R"), (62, "K", "R"), (94, "K", "R"), (129, "K", "R"), (169, "K", "R"),
    (119, "V", "I"), (122, "V", "I"), (235, "I", "V"), (236, "I", "V"), (270, "I", "V"),
    (67, "M", "L"), (80, "M", "L"), (102, "M", "L"),
]

# Radical packing / charge / proline, not the nucleophile itself.
RAD = [
    (95, "W", "G"), (105, "W", "A"), (114, "W", "P"), (163, "W", "K"),
    (63, "C", "A"), (43, "C", "R"),
    (45, "L", "K"), (164, "L", "D"),
    (35, "P", "E"), (111, "P", "E"), (172, "P", "D"),
    (42, "G", "W"), (233, "G", "P"),
    (73, "F", "P"), (231, "F", "K"), (243, "Y", "G"),
    (143, "D", "K"), (89, "Q", "P"), (183, "I", "P"), (248, "A", "P"),
]


def translate(cds: str) -> str:
    aa = []
    for i in range(0, len(cds) - 2, 3):
        res = CODON[cds[i : i + 3]]
        if res == "*":
            break
        aa.append(res)
    return "".join(aa)


def apply_muts(wt: str, muts: list[tuple[int, str, str]]) -> str:
    seq = list(wt)
    seen: set[int] = set()
    for pos, expect, alt in muts:
        if pos in seen:
            raise ValueError(f"duplicate position {pos}")
        seen.add(pos)
        i = pos - 1
        if seq[i] != expect:
            raise ValueError(f"pos {pos}: expected {expect}, found {seq[i]}")
        if alt == expect:
            raise ValueError(f"pos {pos}: {alt} is WT")
        seq[i] = alt
    return "".join(seq)


def fmt_muts(muts: list[tuple[int, str, str]]) -> str:
    return ",".join(f"{w}{p}{a}" for p, w, a in muts)


def wrap(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{wrap(seq)}\n")


def samples(wt: str) -> list[tuple[str, str, list[tuple[int, str, str]], str, str]]:
    """(sample_id, severity, muts, note)."""
    rows: list[tuple[str, str, list, str]] = []

    def add(sid: str, severity: str, muts: list, note: str) -> None:
        rows.append((sid, severity, muts, note))

    add("s01_n0_wt", "wt", [], "identity; MDG should be WT-like (bin 0)")

    cons1 = CONS[:8]
    for i, mut in enumerate(cons1, start=2):
        add(f"s{i:02d}_n1_cons_{mut[1]}{mut[0]}{mut[2]}", "conservative", [mut],
            "single conservative; expect low mlof_score")

    add("s10_n1_mid_A56E", "intermediate", [(56, "A", "E")], "surface Ala→Glu")
    add("s11_n1_rad_W95G", "radical", [(95, "W", "G")], "Trp loss")
    add("s12_n1_rad_C63A", "radical", [(63, "C", "A")], "Cys in CATQ motif")
    add("s13_n1_rad_C43R", "radical", [(43, "C", "R")], "Cys→Arg")
    add("s14_n1_rad_L45K", "radical", [(45, "L", "K")], "buried Leu→Lys")
    add("s15_n1_rad_P111E", "radical", [(111, "P", "E")], "Pro loss")
    add("s16_n1_rad_G145P", "radical", [(145, "G", "P")], "YGN glycine→Pro")
    add("s17_n1_cat_T216A", "catalytic", [(216, "T", "A")], "KTG threonine")
    add("s18_n1_cat_Y144F", "catalytic", [(144, "Y", "F")], "YGN; OH loss, ring kept")
    add("s19_n1_cat_Y144A", "catalytic", [(144, "Y", "A")], "YGN; ring gone")
    add("s20_n1_cat_K215A", "catalytic", [(215, "K", "A")], "KTG lysine→Ala")
    add("s21_n1_cat_K215E", "catalytic", [(215, "K", "E")], "KTG charge flip")
    add("s22_n1_cat_K74A", "catalytic", [(74, "K", "A")], "SxxK lysine")
    add("s23_n1_cat_K74E", "catalytic", [(74, "K", "E")], "SxxK charge flip")
    add("s24_n1_cat_S71T", "catalytic", [(71, "S", "T")], "nucleophile Ser→Thr")
    add("s25_n1_cat_S71A", "catalytic", [(71, "S", "A")], "nucleophile killed")
    add("s26_n1_cat_S71G", "catalytic", [(71, "S", "G")], "nucleophile killed, no beta carbon")

    add("s27_n2_cons", "conservative", CONS[0:2], "two conservative")
    add("s28_n2_cons", "conservative", CONS[2:4], "two conservative")
    add("s29_n2_mix", "mixed", [CONS[0], RAD[0]], "cons + Trp loss; max should track W95G")
    add("s30_n2_mix", "mixed", [CONS[7], (71, "S", "A")], "mild C-term + S71A; max ≈ s25")
    add("s31_n2_cat", "catalytic", [(71, "S", "A"), (74, "K", "E")], "both SxxK residues")
    add("s32_n2_cat", "catalytic", [(144, "Y", "A"), (215, "K", "E")], "YGN + KTG")

    add("s33_n3_cons", "conservative", CONS[0:3], "three conservative")
    add("s34_n3_mix", "mixed", [CONS[0], RAD[0], (71, "S", "A")], "max ≈ S71A despite n=3")
    add("s35_n3_rad", "radical", RAD[0:3], "three packing wrecks, no nucleophile")

    add("s36_n5_cons", "conservative", CONS[0:5], "five conservative")
    add("s37_n5_mix", "mixed", CONS[0:3] + RAD[0:1] + [(71, "S", "A")], "n=5 but max ≈ S71A")
    add("s38_n5_rad", "radical", RAD[0:5], "five radical, no S71")

    add("s39_n8_cons", "conservative", CONS[0:8], "eight conservative")
    add("s40_n8_mix", "mixed", CONS[0:6] + [(71, "S", "A"), RAD[0]], "n=8; max should still ≈ s25")
    add("s41_n8_rad", "radical", RAD[0:8], "eight radical, no S71")

    add("s42_n12_cons", "conservative", CONS[0:12], "twelve conservative")
    add("s43_n12_mix", "mixed", CONS[0:10] + [(71, "S", "A"), RAD[0]], "n=12; max ≈ S71A")
    add("s44_n12_rad", "radical", RAD[0:12], "twelve radical, no S71")

    add("s45_n20_cons", "conservative", CONS[0:20], "twenty conservative; count ≠ damage")
    add("s46_n20_mix", "mixed", CONS[0:18] + [(71, "S", "A"), RAD[0]], "n=20; max ≈ S71A")
    add("s47_n20_rad", "radical", RAD[0:20], "twenty radical, nucleophile intact")
    add("s48_n20_cat", "catalytic", RAD[0:18] + [(71, "S", "G"), (74, "K", "E")],
        "n=20 including S71G+K74E")
    add("s49_n4_active_site", "catalytic",
        [(71, "S", "A"), (74, "K", "E"), (144, "Y", "A"), (215, "K", "E")],
        "four catalytic positions")
    add("s50_n1_far_I270V", "conservative", [(270, "I", "V")],
        "single conservative far from active site")

    out = []
    for sid, severity, muts, note in rows:
        seq = wt if not muts else apply_muts(wt, muts)
        if len(seq) != len(wt):
            raise ValueError(f"{sid}: length changed")
        out.append((sid, seq, muts, severity, note))
    if len(out) != 50:
        raise ValueError(f"expected 50 samples, got {len(out)}")
    if len({r[0] for r in out}) != 50:
        raise ValueError("sample IDs are not unique")
    return out


def main() -> None:
    wt = translate(CDS)
    checks = {
        71: "S", 74: "K", 144: "Y", 145: "G", 215: "K", 216: "T",
        43: "C", 63: "C", 95: "W",
    }
    for pos, aa in checks.items():
        if wt[pos - 1] != aa:
            raise SystemExit(f"WT pos {pos} is {wt[pos - 1]!r}, expected {aa!r} (len={len(wt)})")
    for pos, expect, _alt in CONS + RAD:
        if wt[pos - 1] != expect:
            raise SystemExit(f"pool pos {pos}: expected {expect}, found {wt[pos - 1]}")

    rows = samples(wt)
    refs = [(sid, wt) for sid, _, _, _, _ in rows]
    vars_ = [(sid, seq) for sid, seq, _, _, _ in rows]
    write_fasta(ROOT / "ref.faa", refs)
    write_fasta(ROOT / "variants.faa", vars_)

    man = ROOT / "manifest.tsv"
    with man.open("w") as fh:
        fh.write("sample_id\tn_missense\tseverity\tsubstitutions\tnote\n")
        for sid, _, muts, severity, note in rows:
            fh.write(f"{sid}\t{len(muts)}\t{severity}\t{fmt_muts(muts) or '.'}\t{note}\n")

    print(f"WT length {len(wt)} aa")
    print(f"wrote {ROOT / 'ref.faa'}")
    print(f"wrote {ROOT / 'variants.faa'}")
    print(f"wrote {man}")


if __name__ == "__main__":
    main()
