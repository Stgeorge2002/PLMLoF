"""Tests for ComparisonModule (pairwise MLoF / GoF features)."""

from __future__ import annotations

import torch


class TestComparisonModule:
    def test_output_shape(self):
        from plmlof.models.comparison import ComparisonModule

        d = 64
        comp = ComparisonModule(hidden_size=d, pool_strategy="mean_max")
        b, l_ref, l_var = 2, 10, 8
        ref_emb = {"per_residue": torch.randn(b, l_ref, d), "pooled": torch.randn(b, d)}
        var_emb = {"per_residue": torch.randn(b, l_var, d), "pooled": torch.randn(b, d)}
        ref_mask = torch.ones(b, l_ref)
        var_mask = torch.ones(b, l_var)
        out = comp(ref_emb, var_emb, ref_mask, var_mask)
        assert out.shape == (b, comp.output_size)

    def test_mean_strategy(self):
        from plmlof.models.comparison import ComparisonModule

        d = 32
        comp = ComparisonModule(hidden_size=d, pool_strategy="mean")
        assert comp.output_size == 4 * d

    def test_compare_pooled(self):
        from plmlof.models.comparison import ComparisonModule

        d = 16
        comp = ComparisonModule(hidden_size=d, pool_strategy="mean_max")
        b = 3
        out = comp.compare_pooled(
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, d), torch.randn(b, d),
        )
        assert out.shape == (b, comp.output_size)
