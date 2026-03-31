"""Tests for nucleotide feature extraction and sequence utilities."""

from __future__ import annotations

import torch
import pytest

from plmlof.data.features import extract_nucleotide_features, NUM_NUCLEOTIDE_FEATURES
from plmlof.utils.sequence_utils import (
    translate_dna,
    reverse_complement,
    find_mutations,
    has_premature_stop,
    is_frameshift,
    start_codon_lost,
    compute_truncation_fraction,
)


# =========================================================================
# sequence_utils
# =========================================================================


class TestTranslateDna:
    def test_basic_translation(self):
        # ATG=M, AAA=K, TTT=F
        assert translate_dna("ATGAAATTT") == "MKF"

    def test_stop_codon(self):
        # TAA is stop in table 11
        result = translate_dna("ATGTAA")
        assert "*" in result

    def test_to_stop(self):
        result = translate_dna("ATGTAAAAA", to_stop=True)
        assert result == "M"

    def test_empty(self):
        assert translate_dna("") == ""

    def test_partial_codon_trimmed(self):
        # Extra bases beyond a codon boundary are ignored
        result = translate_dna("ATGA")  # Only ATG translates
        assert result == "M"


class TestReverseComplement:
    def test_basic(self):
        assert reverse_complement("ATGC") == "GCAT"

    def test_palindrome(self):
        assert reverse_complement("ATAT") == "ATAT"


class TestFindMutations:
    def test_identical(self):
        assert find_mutations("MKTL", "MKTL") == []

    def test_missense(self):
        muts = find_mutations("MKTL", "MRTL")
        assert len(muts) == 1
        assert muts[0]["type"] == "missense"
        assert muts[0]["position"] == 2
        assert muts[0]["ref_aa"] == "K"
        assert muts[0]["var_aa"] == "R"

    def test_nonsense(self):
        muts = find_mutations("MKTL", "MK*L")
        nonsense = [m for m in muts if m["type"] == "nonsense"]
        assert len(nonsense) == 1
        assert nonsense[0]["position"] == 3

    def test_truncation(self):
        muts = find_mutations("MKTL", "MK")
        trunc = [m for m in muts if m["type"] == "truncation"]
        assert len(trunc) == 1

    def test_extension(self):
        muts = find_mutations("MK", "MKTL")
        ext = [m for m in muts if m["type"] == "extension"]
        assert len(ext) == 1


class TestHasPrematureStop:
    def test_yes(self):
        assert has_premature_stop("MKTLV", "MK*LV") is True

    def test_no_stop(self):
        assert has_premature_stop("MKTLV", "MRTLV") is False

    def test_stop_at_same_pos(self):
        assert has_premature_stop("MK*LV", "MK*LV") is False


class TestIsFrameshift:
    def test_frameshift(self):
        ref = "ATGAAATTTGGG"  # 12 nt
        var = "ATGAAATTTGG"  # 11 nt (1-base del)
        assert is_frameshift(ref, var) is True

    def test_no_frameshift(self):
        ref = "ATGAAATTTGGG"  # 12 nt
        var = "ATGAAATTT"  # 9 nt (3-base del)
        assert is_frameshift(ref, var) is False

    def test_identical(self):
        ref = "ATGAAATTT"
        assert is_frameshift(ref, ref) is False


class TestStartCodonLost:
    def test_lost(self):
        assert start_codon_lost("ATGAAATTT", "TTGAAATTT") is True

    def test_intact(self):
        assert start_codon_lost("ATGAAATTT", "ATGAAAGGG") is False


class TestComputeTruncationFraction:
    def test_half_truncated(self):
        frac = compute_truncation_fraction("MKTLVVVV", "MKTL")
        assert abs(frac - 0.5) < 0.01

    def test_no_truncation(self):
        assert compute_truncation_fraction("MKTL", "MKTL") == 0.0

    def test_extension_returns_zero_or_negative(self):
        frac = compute_truncation_fraction("MK", "MKTL")
        assert frac <= 0.0


# =========================================================================
# extract_nucleotide_features
# =========================================================================


class TestExtractNucleotideFeatures:
    def test_shape(self):
        feat = extract_nucleotide_features("ATGAAATTT", "ATGAAATTT")
        assert isinstance(feat, torch.Tensor)
        assert feat.shape == (NUM_NUCLEOTIDE_FEATURES,)

    def test_identical_sequences(self):
        feat = extract_nucleotide_features("ATGAAATTT", "ATGAAATTT")
        # No frameshift, no premature stop, nothing
        assert feat[0].item() == 0.0  # is_frameshift
        assert feat[1].item() == 0.0  # has_premature_stop
        assert feat[2].item() == 0.0  # start_codon_lost

    def test_frameshift_detected(self):
        ref = "ATGAAATTTGGG"
        var = "ATGAAATTTGG"  # 1-base deletion → frameshift
        feat = extract_nucleotide_features(ref, var)
        assert feat[0].item() == 1.0  # is_frameshift

    def test_premature_stop_detected(self):
        ref = "ATGAAATTTGGG"  # M K F G
        var = "ATGTAATTTGGG"  # M * F G
        feat = extract_nucleotide_features(ref, var)
        assert feat[1].item() == 1.0  # has_premature_stop

    def test_protein_only_fallback(self):
        # When DNA is empty, should still work from protein-level features
        feat = extract_nucleotide_features("", "", "MKTL", "MK*L")
        assert feat.shape == (NUM_NUCLEOTIDE_FEATURES,)
        # Should detect premature stop from protein
        assert feat[1].item() == 1.0

    def test_length_ratio(self):
        ref = "ATGAAATTT"  # 9 nt
        var = "ATGAAATTTGGG"  # 12 nt
        feat = extract_nucleotide_features(ref, var)
        # length_ratio = len(var_prot) / len(ref_prot)
        assert feat[11].item() > 0.0  # length_ratio should be positive
