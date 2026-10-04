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


class TestSiteBank:
    def test_roundtrip_and_missing(self, tmp_path):
        from plmlof.embed import SiteBank

        bank = SiteBank(n_slots=4, dim=3, path=tmp_path / "sites.dat")
        try:
            bank.add("AAA", [0, 2], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            assert bank.get("AAA", 0).tolist() == [1.0, 2.0, 3.0]
            assert bank.get("AAA", 2).tolist() == [4.0, 5.0, 6.0]
            assert bank.get("AAA", 1) is None
            assert bank.get("BBB", 0) is None
        finally:
            bank.close()

    def test_overflow_fails_fast(self):
        from plmlof.embed import SiteBank

        bank = SiteBank(n_slots=1, dim=2)
        bank.add("A", [0], torch.ones(1, 2))
        try:
            bank.add("B", [0], torch.ones(1, 2))
            raise AssertionError("expected overflow")
        except RuntimeError as exc:
            assert "overflow" in str(exc)


class TestSiteCompare:
    def test_delta_changes_output(self):
        from plmlof.model import SiteCompare

        torch.manual_seed(0)
        cmp = SiteCompare(hidden_size=8)
        cmp.eval()
        ref = torch.randn(2, 3, 8)
        var = ref.clone()
        var[:, 1] = var[:, 1] + 1.0
        with torch.no_grad():
            same = cmp(ref, ref)
            diff = cmp(ref, var)
        assert not torch.allclose(same, diff)
