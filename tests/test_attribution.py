"""Tests for attribution module."""

from __future__ import annotations

import torch
import pytest

from plmlof.inference.attribution import (
    AttributionResult,
    compute_rule_based_attribution,
    generate_attribution,
)


class TestRuleBasedAttribution:
    def test_no_mutations(self):
        result = compute_rule_based_attribution("MKTL", "MKTL")
        assert result == []

    def test_nonsense_high_impact(self):
        result = compute_rule_based_attribution("MKTL", "MK*L")
        nonsense = [r for r in result if r["type"] == "nonsense"]
        assert len(nonsense) == 1
        assert nonsense[0]["impact"] == "high"
        assert nonsense[0]["position"] == 3

    def test_missense_moderate(self):
        result = compute_rule_based_attribution("MKTL", "MRTL")
        missense = [r for r in result if r["type"] == "missense"]
        assert len(missense) == 1
        assert missense[0]["impact"] == "moderate"

    def test_frameshift_from_dna(self):
        ref_dna = "ATGAAATTTGGG"  # 12 nt
        var_dna = "ATGAAATTTGG"   # 11 nt
        result = compute_rule_based_attribution("MKTL", "MKTL", ref_dna, var_dna)
        frameshift = [r for r in result if r["type"] == "frameshift"]
        assert len(frameshift) == 1
        assert frameshift[0]["impact"] == "high"

    def test_truncation(self):
        result = compute_rule_based_attribution("MKTLVV", "MKT")
        trunc = [r for r in result if r["type"] == "truncation"]
        assert len(trunc) == 1
        assert trunc[0]["impact"] == "high"


class TestGenerateAttribution:
    def test_returns_attribution_result(self):
        result = generate_attribution(
            gene="geneA",
            prediction="LoF",
            confidence=0.95,
            ref_protein="MKTLVV",
            var_protein="MK*LVV",
        )
        assert isinstance(result, AttributionResult)
        assert result.gene == "geneA"
        assert result.prediction == "LoF"
        assert result.confidence == 0.95

    def test_position_scores_length(self):
        ref = "MKTLVV"
        var = "MK*LVV"
        result = generate_attribution("g", "LoF", 0.9, ref, var)
        assert len(result.position_scores) == max(len(ref), len(var))

    def test_high_impact_positions_scored(self):
        result = generate_attribution("g", "LoF", 0.9, "MKTLVV", "MK*LVV")
        # Position 3 (0-indexed=2) should have a high score
        assert result.position_scores[2] > 0

    def test_summary_not_empty(self):
        result = generate_attribution("g", "LoF", 0.9, "MKTL", "MK*L")
        assert len(result.summary) > 0

    def test_wildtype_summary(self):
        result = generate_attribution("g", "WT", 0.99, "MKTL", "MKTL")
        assert "wildtype" in result.summary.lower() or "No mutations" in result.summary

    def test_gradient_scores_used(self):
        grad = torch.tensor([0.1, 0.2, 0.9, 0.0])
        result = generate_attribution("g", "LoF", 0.9, "MKTL", "MK*L", gradient_scores=grad)
        assert abs(result.position_scores[2] - 0.9) < 1e-5

    def test_top_positions_sorted_by_impact(self):
        # ref has both nonsense and missense
        result = generate_attribution("g", "LoF", 0.9, "MKTLV", "MR*LV")
        if len(result.top_positions) >= 2:
            # First should be nonsense (high), second missense (moderate)
            assert result.top_positions[0]["impact"] == "high"
