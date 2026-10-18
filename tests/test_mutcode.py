"""MutCode reconstruction + LoF-matrix import (stdlib-level checks)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from plmlof.mutcode import (
    MutCodeError,
    apply_edits,
    expected_target_length,
    parse_codes,
    reconstruct,
    resolve_panaroo_dir,
    write_fasta,
)


# MutCode README worked example (35-base reference).
REF_35 = "ACGTACGTACGTACGTACGTACGTACGTACGTACG"
CODES_35 = '[[12,1,"T"],[20,0,"AC"],[30,3,""]]'
TARGET_35 = "ACGTACGTACGTTCGTACGTACACGTACGTACCG"
TARGET_35_SHA = "c90e5919e2e31101c082f8180017c601f6c697aa04813ede1c42ccfeb6898379"
REF_35_SHA = "2783b12cc1282954ea632785271f67dd047a5d3c223413399f3c85850b2c8253"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


class TestApplyEdits:
    def test_readme_worked_example(self):
        assert len(REF_35) == 35
        assert _sha(REF_35) == REF_35_SHA
        out = apply_edits(REF_35, CODES_35)
        assert out == TARGET_35
        assert len(out) == 34
        assert _sha(out) == TARGET_35_SHA

    def test_empty_codes_identity(self):
        assert apply_edits("ACGT", "[]") == "ACGT"
        assert apply_edits("ACGT", None) == "ACGT"

    def test_substitution(self):
        assert apply_edits("ACGTACGT", '[[2,1,"T"]]') == "ACTTACGT"

    def test_insertion_before_index(self):
        # POS trim example: insert AA before index 5 on ACGTACGT → ACGTA AACGT
        assert apply_edits("ACGTACGT", '[[5,0,"AA"]]') == "ACGTAAAACGT"

    def test_deletion(self):
        # Anchored deletion example: [5,1,""] on ACGTACGT → ACGTAGT
        assert apply_edits("ACGTACGT", '[[5,1,""]]') == "ACGTAGT"

    def test_insert_at_start_and_append(self):
        assert apply_edits("ACGT", '[[0,0,"NN"]]') == "NNACGT"
        assert apply_edits("ACGT", '[[4,0,"T"]]') == "ACGTT"

    def test_delete_whole_gene(self):
        out = apply_edits("ACGT", '[[0,4,""]]')
        assert out == ""
        assert _sha(out) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_adjacent_edits_legal(self):
        # next start == previous start + delete_length
        out = apply_edits("ACGTACGT", '[[2,1,"T"],[3,1,"G"]]')
        assert out == "ACTGACGT"

    def test_overlap_rejected(self):
        with pytest.raises(MutCodeError, match="overlapping"):
            apply_edits("ACGTACGT", '[[2,3,"T"],[3,1,"G"]]')

    def test_duplicate_start_rejected(self):
        with pytest.raises(MutCodeError, match="duplicate"):
            apply_edits("ACGTACGT", '[[2,1,"T"],[2,1,"G"]]')

    def test_empty_noop_rejected(self):
        with pytest.raises(MutCodeError, match="empty insertion"):
            apply_edits("ACGT", '[[1,0,""]]')

    def test_non_iupac_rejected(self):
        with pytest.raises(MutCodeError, match="IUPAC"):
            apply_edits("ACGT", '[[1,1,"X"]]')

    def test_delete_past_end_rejected(self):
        with pytest.raises(MutCodeError, match="past reference"):
            apply_edits("ACGT", '[[2,5,""]]')

    def test_case_preserved(self):
        # Copied stretches keep FASTA case; insertions keep ALT case.
        assert apply_edits("acgtACGT", '[[4,1,"t"]]') == "acgttCGT"


class TestReconstruct:
    def test_length_checks(self):
        out = reconstruct(
            "ACGTACGT",
            '[[5,0,"AA"]]',
            target_length=10,
        )
        assert out == "ACGTAAAACGT"

    def test_bad_target_length(self):
        with pytest.raises(MutCodeError, match="target_length"):
            reconstruct("ACGT", "[]", target_length=9)

    def test_sha256_roundtrip(self):
        from plmlof.mutcode import sha256_seq

        ref = "ACGTACGT"
        out = reconstruct(
            ref,
            '[[5,0,"AA"]]',
            reference_sha256=sha256_seq(ref),
            target_sha256=sha256_seq("ACGTAAAACGT"),
            target_length=10,
        )
        assert out == "ACGTAAAACGT"

    def test_wrong_reference_sha_rejected(self):
        with pytest.raises(MutCodeError, match="SHA-256"):
            reconstruct("ACGT", "[]", reference_sha256="0" * 64)

    def test_expected_target_length(self):
        edits = parse_codes('[[12,1,"T"],[20,0,"AC"],[30,3,""]]')
        assert expected_target_length(35, edits) == 34


class TestImport:
    def test_end_to_end_tiny_matrix(self, tmp_path: Path):
        from plmlof.lof_import import build_from_lof_matrix

        # Minimal Panaroo layout matching N44 nested panaroo_output/
        pan = tmp_path / "panaroo" / "chromosome" / "panaroo_output"
        pan.mkdir(parents=True)
        write_fasta(
            pan / "pan_genome_reference.fa",
            [("geneA", "ACGTACGT"), ("geneB", "TT")],
        )

        matrix = tmp_path / "lof_matrix.tsv"
        with open(matrix, "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "Replicon", "Gene", "Genome_Contig", "Genome_Start",
                    "Genome_End", "Genome_Strand", "Sample", "Description",
                    "Reference_Length", "Sample_Length", "Mutation_Codes",
                ],
                delimiter="\t",
            )
            w.writeheader()
            w.writerow({
                "Replicon": "chromosome",
                "Gene": "geneA",
                "Genome_Contig": "c1",
                "Genome_Start": "1",
                "Genome_End": "8",
                "Genome_Strand": "+",
                "Sample": "s1",
                "Description": "demo",
                "Reference_Length": "8",
                "Sample_Length": "10",
                "Mutation_Codes": '[[2,1,"T"],[5,0,"AA"]]',
            })
            w.writerow({
                "Replicon": "chromosome",
                "Gene": "geneA",
                "Genome_Contig": "c1",
                "Genome_Start": "1",
                "Genome_End": "8",
                "Genome_Strand": "+",
                "Sample": "s2",
                "Description": "demo",
                "Reference_Length": "8",
                "Sample_Length": "8",
                "Mutation_Codes": "[]",
            })
            w.writerow({
                "Replicon": "chromosome",
                "Gene": "geneB",
                "Genome_Contig": "",
                "Genome_Start": "",
                "Genome_End": "",
                "Genome_Strand": "",
                "Sample": "s1",
                "Description": "unchanged",
                "Reference_Length": "2",
                "Sample_Length": "2",
                "Mutation_Codes": "[]",
            })

        out = tmp_path / "plmlof_input"
        manifest, stats = build_from_lof_matrix(
            lof_matrix=matrix,
            results_dir=tmp_path,
            out_dir=out,
            strict_sample_length=True,
            skip_missing_genes=False,
            min_samples=1,
        )
        assert stats[0]["n_records"] == 3
        assert stats[0]["n_mutated"] == 1

        gene_a = out / "chromosome" / "geneA"
        assert (gene_a / "ref.fasta").is_file()
        assert (gene_a / "var.fasta").is_file()
        var_text = (gene_a / "var.fasta").read_text()
        assert "ACTTAAACGT" in var_text.replace("\n", "")
        assert "ACGTACGT" in var_text.replace("\n", "")  # s2 identical
        assert (out / "manifest_all.tsv").is_file()
        assert len(manifest) == 3

    def test_empty_codes_ignores_matrix_reference_length(self, tmp_path: Path):
        """thyA-style: matrix Reference_Length 792, Panaroo FASTA 795, codes []."""
        from plmlof.lof_import import build_from_lof_matrix

        pan = tmp_path / "panaroo" / "chromosome" / "panaroo_output"
        pan.mkdir(parents=True)
        fasta_seq = "A" * 795  # includes terminal stop relative to matrix 792
        write_fasta(pan / "pan_genome_reference.fa", [("thyA", fasta_seq)])

        matrix = tmp_path / "lof_matrix.tsv"
        with open(matrix, "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "Replicon", "Gene", "Sample", "Description",
                    "Reference_Length", "Sample_Length", "Mutation_Codes",
                ],
                delimiter="\t",
            )
            w.writeheader()
            w.writerow({
                "Replicon": "chromosome",
                "Gene": "thyA",
                "Sample": "SRR1",
                "Description": "thymidylate synthase",
                "Reference_Length": "792",
                "Sample_Length": "792",
                "Mutation_Codes": "[]",
            })

        out = tmp_path / "plmlof_input"
        manifest, stats = build_from_lof_matrix(
            lof_matrix=matrix,
            results_dir=tmp_path,
            out_dir=out,
            strict_sample_length=False,
            skip_missing_genes=False,
            min_samples=1,
        )
        assert stats[0]["n_identical"] == 1
        assert stats[0]["n_matrix_ref_length_mismatch"] == 1
        var = (out / "chromosome" / "thyA" / "var.fasta").read_text().replace("\n", "")
        assert "A" * 795 in var
        assert manifest[0]["ref_length_nt"] == 795
        assert manifest[0]["var_length_nt"] == 795

    def test_resolve_panaroo_flat_and_nested(self, tmp_path: Path):
        nested = tmp_path / "a" / "panaroo" / "chromosome" / "panaroo_output"
        nested.mkdir(parents=True)
        (nested / "pan_genome_reference.fa").write_text(">g\nACGT\n")
        assert resolve_panaroo_dir(tmp_path / "a", "chromosome") == nested

        flat = tmp_path / "b" / "panaroo" / "chromosome"
        flat.mkdir(parents=True)
        (flat / "pan_genome_reference.fa").write_text(">g\nACGT\n")
        assert resolve_panaroo_dir(tmp_path / "b", "chromosome") == flat
