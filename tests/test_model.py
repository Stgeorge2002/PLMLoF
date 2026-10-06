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


class TestForwardHiddenAndLogits:
    def test_mlm_uses_esm_trunk(self):
        from types import SimpleNamespace

        from plmlof.embed import forward_hidden_and_logits

        class Trunk:
            def __call__(self, ids, attention_mask=None):
                return SimpleNamespace(last_hidden_state=torch.ones(ids.size(0), ids.size(1), 4))

        class Head:
            def __call__(self, hidden):
                return hidden[..., :2]

        model = SimpleNamespace(esm=Trunk(), lm_head=Head())
        ids = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.ones(2, 5)
        hidden, logits = forward_hidden_and_logits(model, ids, mask)
        assert hidden.shape == (2, 5, 4)
        assert logits.shape == (2, 5, 2)


class TestSiteCompare:
    def test_delta_changes_output(self):
        from plmlof.constants import SITE_WINDOW
        from plmlof.model import SiteCompare

        torch.manual_seed(0)
        cmp = SiteCompare(hidden_size=8)
        cmp.eval()
        ref = torch.randn(2, SITE_WINDOW, 8)
        var = ref.clone()
        var[:, SITE_WINDOW // 2] = var[:, SITE_WINDOW // 2] + 1.0
        with torch.no_grad():
            same = cmp(ref, ref)
            diff = cmp(ref, var)
        assert not torch.allclose(same, diff)


class TestHammingAndMaxSite:
    def test_hamming_two_is_mlof_missense(self):
        from plmlof.sites import aa_hamming, is_mlof_missense, missense_sites

        ref = "MKTAA"
        var = "MRTWA"
        assert aa_hamming(ref, var) == 2
        assert is_mlof_missense(ref, var)
        assert missense_sites(ref, var) == [1, 3]
        assert not is_mlof_missense(ref, "MRTWW")  # Hamming 3

    def test_max_of_two_sites(self):
        from plmlof.constants import NUM_SITE_CHEM, SITE_WINDOW
        from plmlof.model import TaskNet

        torch.manual_seed(0)
        net = TaskNet(hidden_size=8, task="mlof")
        net.eval()
        b, d = 2, 8
        site = torch.randn(b, SITE_WINDOW, d)
        site2_ref = torch.randn(b, SITE_WINDOW, d)
        site2_var = site2_ref + 3.0
        dummy = torch.zeros(b, d)
        nuc = torch.zeros(b, 12)
        chem = torch.zeros(b, NUM_SITE_CHEM)
        n_sites = torch.tensor([1, 2])
        with torch.no_grad():
            first = net.forward_from_cache(
                dummy, dummy, dummy, dummy, nuc,
                site_ref=site, site_var=site, site_chem=chem,
            )
            second = net.forward_from_cache(
                dummy, dummy, dummy, dummy, nuc,
                site_ref=site2_ref, site_var=site2_var, site_chem=chem,
            )
            both = net.forward_from_cache(
                dummy, dummy, dummy, dummy, nuc,
                site_ref=site, site_var=site, site_chem=chem,
                site_ref2=site2_ref, site_var2=site2_var, site_chem2=chem,
                n_sites=n_sites,
            )
        assert torch.allclose(both[0], first[0])
        assert torch.allclose(both[1], torch.maximum(first[1], second[1]))
        assert not torch.allclose(first[1], second[1])

    def test_ablate_logodds_zeros_channel(self):
        from plmlof.chem import pack_site_chem
        from plmlof.constants import CHEM_LLR, SITE_WINDOW
        from plmlof.model import TaskNet

        torch.manual_seed(1)
        net = TaskNet(hidden_size=8, task="mlof")
        net.eval()
        b, d = 3, 8
        site = torch.randn(b, SITE_WINDOW, d)
        dummy = torch.zeros(b, d)
        nuc = torch.zeros(b, 12)
        chem = torch.stack([pack_site_chem("A", "W", llr=2.0) for _ in range(b)])
        with torch.no_grad():
            full = net.forward_from_cache(dummy, dummy, dummy, dummy, nuc, site_ref=site, site_var=site, site_chem=chem)
            net.ablate_logodds = True
            ablated = net.forward_from_cache(dummy, dummy, dummy, dummy, nuc, site_ref=site, site_var=site, site_chem=chem)
            zeroed = chem.clone()
            zeroed[:, CHEM_LLR] = 0
            net.ablate_logodds = False
            match = net.forward_from_cache(dummy, dummy, dummy, dummy, nuc, site_ref=site, site_var=site, site_chem=zeroed)
        assert not torch.allclose(full, ablated)
        assert torch.allclose(ablated, match)
