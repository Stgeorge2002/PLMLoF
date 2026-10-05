"""Wreck rules, labels, empirical p-values, and task nets."""

from __future__ import annotations

import random

import numpy as np
import torch

from plmlof.data.features import LOF_LEAK_NUC_INDICES
from plmlof.dataset import ProteinGroupBatchSampler, lof_train_keep_indices
from plmlof.domains import lof_prior, sample_events
from plmlof.encoders import ESM2_LOF, ESM2_PAIR, esm2_for_task
from plmlof.hmmer import parse_domtblout
from plmlof.labels import channel_for_lof, damage_from_z, display_bin, lof_score_for_pair, lof_score_from_z
from plmlof.metrics import collapse_is_fail, gof_metrics, lof_metrics, mlof_metrics, within_gene_auroc
from plmlof.model import SiteCompare, TaskNet
from plmlof.rank import ranknet_loss
from plmlof.sites import aligned_site_index, gather_site_windows
from plmlof.splits import parse_tasks, split_mlof_nested, split_residues_within_protein, substitution_sites
from plmlof.stats import benjamini_hochberg, empirical_p
from plmlof.wreck import wreck_call, wreck_grade


class TestWreck:
    def test_truncation(self):
        ref = "M" + "A" * 99
        var = "M" + "A" * 20
        flagged, kind = wreck_call(ref, var)
        assert flagged
        assert kind == "truncation"

    def test_intact_missense_is_not_wreck(self):
        ref = "MKTLLLTLVVVTLAALG"
        var = "MRTLLLTLVVVTLAALG"
        flagged, kind = wreck_call(ref, var)
        assert not flagged
        assert kind == "none"

    def test_frameshift_from_dna(self):
        ref = "MKTLLL"
        var = "MKTLLL"
        flagged, kind = wreck_call(ref, var, ref_dna="ATGAAAACACTGCTG", var_dna="ATGAAAACCTGCTG")
        assert flagged
        assert kind == "frameshift"

    def test_identity_not_wreck(self):
        s = "MKTLLLTLVVVTLAALG"
        flagged, _ = wreck_call(s, s)
        assert not flagged

    def test_c_terminal_stop_is_tail_not_sure(self):
        ref = "M" + "A" * 99
        var = ref[:95]
        flagged, kind = wreck_call(ref, var)
        assert not flagged
        structural, kind, prior = wreck_grade(ref, var)
        assert structural
        assert kind == "tail_stop"
        assert prior == 0.4

    def test_post_domain_stop_not_sure_even_if_half_gone(self):
        ref = "M" + "A" * 99
        var = ref[:50]
        domains = [(10, 40)]
        flagged, _ = wreck_call(ref, var, domains=domains)
        assert not flagged
        _, _, prior = wreck_grade(ref, var, domains=domains)
        assert prior == 0.4
        flagged_len, _ = wreck_call(ref, var)
        assert flagged_len


class TestLabels:
    def test_z_bins(self):
        assert lof_score_from_z(-2.5) == 0.70
        assert lof_score_from_z(-1.5) == 0.40
        assert lof_score_from_z(0.2) == 0.00
        assert lof_score_from_z(1.2) is None
        assert lof_score_from_z(2.5) is None
        assert lof_score_from_z(1.2, drop_gain=False) == 0.00
        assert lof_score_from_z(2.5, drop_gain=False) == 0.00

    def test_wreck_overrides_z(self):
        ref = "M" + "A" * 80
        var = "M" + "A" * 10
        assert lof_score_for_pair(ref, var, z=0.0) == 1.0

    def test_generator_tail_type_is_not_one(self):
        assert lof_score_for_pair("M" + "A" * 99, "M" + "A" * 90, wreck_type="stop_tail") == 0.4
        assert channel_for_lof(0.4, "stop_tail", None) == "tail_wreck"

    def test_stored_score_wins(self):
        ref = "M" + "A" * 80
        var = "M" + "A" * 10
        assert lof_score_for_pair(ref, var, wreck_type="stop", lof_score=0.4) == 0.4


class TestDomainGeometry:
    def test_stop_before_first_domain_is_sure(self):
        score, geom = lof_prior("stop", 5, 100, [(20, 60)])
        assert score == 1.0
        assert geom == "pre_domain"

    def test_frameshift_inside_domain_is_sure(self):
        score, geom = lof_prior("frameshift", 30, 100, [(20, 60)])
        assert score == 1.0
        assert geom == "in_domain"

    def test_linker_frameshift_is_sure(self):
        score, geom = lof_prior("frameshift", 45, 100, [(10, 40), (60, 80)])
        assert score == 1.0
        assert geom == "linker"

    def test_stop_after_last_domain_is_tail(self):
        score, geom = lof_prior("stop", 90, 100, [(10, 40), (50, 70)])
        assert score == 0.4
        assert geom == "tail"

    def test_whole_domain_drop_of_one_of_two(self):
        score, geom = lof_prior(
            "domain_drop", 10, 100, [(10, 40), (50, 80)],
            last_affected=40, n_domains_dropped=1, n_domains_remaining=1,
        )
        assert score == 0.7
        assert geom == "in_domain"

    def test_dropping_the_only_domain_is_sure(self):
        score, _ = lof_prior(
            "domain_drop", 10, 100, [(10, 80)],
            last_affected=80, n_domains_dropped=1, n_domains_remaining=0,
        )
        assert score == 1.0

    def test_in_domain_vs_extra_missense(self):
        in_d, _ = lof_prior("missense", 25, 100, [(10, 40)])
        extra, _ = lof_prior("missense", 90, 100, [(10, 40)])
        assert in_d == 0.7
        assert extra == 0.15

    def test_no_hmmer_last_tenth_is_tail(self):
        early, _ = lof_prior("stop", 40, 100, ())
        tail, _ = lof_prior("stop", 96, 100, ())
        assert early == 1.0
        assert tail == 0.4

    def test_sample_events_mix_sure_and_tail(self):
        rng = random.Random(0)
        prot = "M" + "A" * 119
        dna = "ATG" + "GCA" * 119
        domains = [(10, 40), (50, 80)]
        evs = sample_events(prot, dna, domains, rng, n=3)
        assert evs
        assert any(e.lof_score >= 0.8 for e in evs)
        assert any(e.lof_score <= 0.5 for e in evs)

    def test_parse_domtblout(self, tmp_path):
        line = (
            "PF00001 PF00001.1 200 abc123 - 100 1e-20 80.0 0.0 "
            "1 1 1e-22 1e-20 80.0 0.0 1 100 5 90 3 95 0.97 desc\n"
        )
        path = tmp_path / "domtbl.txt"
        path.write_text("# comment\n" + line, encoding="utf-8")
        rows = parse_domtblout(path)
        assert rows[0]["protein_id"] == "abc123"
        assert rows[0]["start"] == 3
        assert rows[0]["end"] == 95


class TestEmpiricalP:
    def test_extreme_is_small(self):
        null = np.linspace(0.0, 0.4, 200)
        p = empirical_p(0.95, null)
        assert p.shape == (1,)
        assert p[0] < 0.02

    def test_typical_is_large(self):
        null = np.linspace(0.0, 1.0, 200)
        p = empirical_p(0.1, null)
        assert p[0] > 0.8

    def test_bh_monotone(self):
        p = np.array([0.001, 0.01, 0.2, 0.8])
        q = benjamini_hochberg(p)
        assert np.all(q >= p - 1e-12)
        assert q[-1] <= 1.0


class TestMetrics:
    def test_lof_selects_wreck_auroc(self):
        z = np.array([-3.0, 0.0, 0.1, 0.2, np.nan, np.nan])
        pred = np.array([0.8, 0.1, 0.1, 0.05, 0.99, 0.01])
        target = np.array([0.7, 0.0, 0.0, 0.0, 1.0, 0.0])
        wreck = np.array([False, False, False, False, True, False])
        miss = np.array([True, True, True, True, False, False])
        m = lof_metrics(pred, target, z, wreck, miss)
        assert m["wreck_auroc"] > 0.9
        assert m["selection"] == m["wreck_auroc"]

    def test_lof_selection_ignores_missense_rank(self):
        z = np.array([-3.0, 0.0, 0.1, 0.2])
        good_miss = np.array([0.9, 0.1, 0.1, 0.05])
        bad_miss = np.array([0.05, 0.9, 0.8, 0.7])
        target = np.array([0.7, 0.0, 0.0, 0.0])
        wreck = np.zeros(4, dtype=bool)
        miss = np.ones(4, dtype=bool)
        a = lof_metrics(good_miss, target, z, wreck, miss)
        b = lof_metrics(bad_miss, target, z, wreck, miss)
        assert a["selection"] == b["selection"] == 0.0

    def test_mlof_selects_within_gene(self):
        zs, preds, pids = [], [], []
        for name in ("c", "d", "e"):
            z = np.linspace(-3.0, 0.0, 10)
            zs.append(z)
            preds.append(-z / 3.0)
            pids.extend([name] * 10)
        z = np.concatenate(zs)
        pred = np.concatenate(preds)
        miss = np.ones(z.size, dtype=bool)
        m = mlof_metrics(pred, pred, z, miss, protein_id=pids)
        assert m["n_genes_ranked"] == 3
        assert m["within_gene_spearman"] > 0.9
        assert m["selection"] == m["within_gene_spearman"]
        assert m["collapse_fraction"] == 0.0

    def test_mlof_collapse_zeros_selection(self):
        z = np.concatenate([np.linspace(-3, 0, 10)] * 4)
        pred = np.ones_like(z) * 0.4
        miss = np.ones(z.size, dtype=bool)
        pids = ["w"] * 10 + ["x"] * 10 + ["y"] * 10 + ["z"] * 10
        m = mlof_metrics(pred, pred, z, miss, protein_id=pids)
        assert m["collapse_fraction"] == 1.0
        assert m["selection"] == 0.0

    def test_gof_precision(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        p = np.array([0.95, 0.91, 0.2, 0.1, 0.05, 0.01])
        m = gof_metrics(p, y, threshold=0.90)
        assert m["n_calls"] == 2
        assert m["precision_at_thr"] == 1.0
        assert m["caller_ready"] == 0.0

    def test_within_gene_auroc(self):
        pred = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.7, 0.15])
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        genes = ["a"] * 8
        m = within_gene_auroc(pred, y, genes)
        assert m["n_genes_with_auroc"] == 1
        assert m["within_gene_auroc"] == 1.0

    def test_gof_selects_within_gene(self):
        pred = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.7, 0.2])
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        genes = ["a"] * 8
        m = gof_metrics(pred, y, threshold=0.90, gene=genes)
        assert m["selection"] == m["within_gene_auroc"]
        assert m["caller_ready"] == 0.0  # only one gene

    def test_collapse_fail_needs_three_genes(self):
        assert not collapse_is_fail({"n_genes_scored": 1.0, "collapse_fraction": 1.0})
        assert collapse_is_fail({"n_genes_scored": 4.0, "collapse_fraction": 0.75})


class TestTaskNet:
    def test_lof_range(self):
        net = TaskNet(hidden_size=16, task="lof", pool_strategy="mean_max")
        b, d = 4, 16
        raw = net.forward_from_pooled(
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, 12),
        )
        assert raw.shape == (b,)
        assert torch.all(raw >= 0) and torch.all(raw <= 1)

    def test_gof_logit(self):
        net = TaskNet(hidden_size=16, task="growth_gof", pool_strategy="mean_max")
        b, d = 3, 16
        logit = net.forward_from_pooled(
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, 12),
        )
        p = net.probability(logit)
        assert p.shape == (b,)
        assert torch.all(p >= 0) and torch.all(p <= 1)

    def test_lof_is_alignment_free(self):
        net = TaskNet(hidden_size=16, task="lof", pool_strategy="mean_max")
        net.eval()
        torch.manual_seed(0)
        b, d = 2, 16
        var_mean, var_max = torch.randn(b, d), torch.randn(b, d)
        nuc = torch.zeros(b, 12)
        nuc[:, 0] = 1.0
        with torch.no_grad():
            a = net.forward_from_pooled(torch.randn(b, d), torch.randn(b, d), var_mean, var_max, nuc)
            b_out = net.forward_from_pooled(torch.zeros(b, d), torch.zeros(b, d), var_mean, var_max, torch.zeros_like(nuc))
        assert torch.allclose(a, b_out, atol=1e-6)

    def test_mlof_uses_sites_not_pool(self):
        net = TaskNet(hidden_size=16, task="mlof", pool_strategy="mean_max")
        net.eval()
        torch.manual_seed(0)
        b, d = 2, 16
        site_ref = torch.randn(b, 3, d)
        site_var = torch.randn(b, 3, d)
        nuc = torch.zeros(b, 12)
        for i in LOF_LEAK_NUC_INDICES:
            nuc[:, i] = 1.0
        with torch.no_grad():
            a = net.forward_from_cache(
                torch.randn(b, d), torch.randn(b, d), torch.randn(b, d), torch.randn(b, d),
                nuc, site_ref=site_ref, site_var=site_var,
            )
            b_out = net.forward_from_cache(
                torch.zeros(b, d), torch.zeros(b, d), torch.zeros(b, d), torch.zeros(b, d),
                torch.zeros_like(nuc), site_ref=site_ref, site_var=site_var,
            )
        assert torch.allclose(a, b_out, atol=1e-6)

    def test_mlof_requires_sites(self):
        net = TaskNet(hidden_size=16, task="mlof")
        b, d = 2, 16
        try:
            net.forward_from_pooled(
                torch.randn(b, d), torch.randn(b, d),
                torch.randn(b, d), torch.randn(b, d),
                torch.zeros(b, 12),
            )
        except ValueError as exc:
            assert "site_ref" in str(exc)
        else:
            raise AssertionError("MLoF must refuse pooled-only forward")


class TestLofTrainFilter:
    def test_drops_wrecks_and_caps_identity(self):
        wreck = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
        miss = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
        targets = torch.tensor([1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
        channels = [
            "clear_wreck", "strong_missense", "wt",
            "wt", "wt", "wt", "wt", "tail_wreck",
        ]
        idx = set(lof_train_keep_indices(
            wreck, miss, targets, channels=channels, seed=0, identity_per_missense=1.0,
        ))
        assert 0 not in idx
        assert {1, 2, 7} <= idx
        assert sum(i in idx for i in (3, 4, 5, 6)) == 2


class TestResidueSplit:
    def test_same_site_stays_together(self):
        ref = "M" + "A" * 20
        assert substitution_sites(ref, ref[:5] + "V" + ref[6:]) == (6,)
        assert substitution_sites(ref, ref[:5] + "T" + ref[6:]) == (6,)

    def test_no_site_leak_and_every_protein_in_train(self):
        ref = "M" + "A" * 30
        pids, refs, vars_ = [], [], []
        for site in range(2, 12):
            for aa in "VTL":
                pids.append("protA")
                refs.append(ref)
                vars_.append(ref[:site] + aa + ref[site + 1:])
        for site in range(2, 8):
            pids.append("protB")
            refs.append(ref)
            vars_.append(ref[:site] + "V" + ref[site + 1:])
        splits = split_residues_within_protein(pids, refs, vars_, seed=0)
        keyed: dict[tuple, set[str]] = {}
        for pid, r, v, sp in zip(pids, refs, vars_, splits):
            keyed.setdefault((pid, substitution_sites(r, v)), set()).add(sp)
        assert all(len(s) == 1 for s in keyed.values())
        assert any(p == "protA" and s == "train" for p, s in zip(pids, splits))
        assert any(p == "protB" and s == "train" for p, s in zip(pids, splits))
        assert "val" in splits and "test" in splits

    def test_parse_tasks(self):
        assert parse_tasks(None) == ["lof", "mlof", "growth_gof", "amr_gof"]
        assert parse_tasks(["growth_gof"]) == ["growth_gof"]
        assert parse_tasks(["lof,growth_gof"]) == ["lof", "growth_gof"]


class TestMlofNestedSplit:
    def test_held_proteins_never_in_train(self):
        ref = "M" + "A" * 40
        pids, refs, vars_ = [], [], []
        for prot in ("p0", "p1", "p2", "p3", "p4"):
            for site in range(2, 12):
                pids.append(prot)
                refs.append(ref)
                vars_.append(ref[:site] + "V" + ref[site + 1:])
        splits = split_mlof_nested(pids, refs, vars_, seed=0, protein_test_frac=0.2)
        held = {p for p, s in zip(pids, splits) if s == "protein_test"}
        assert held
        for p in held:
            labels = {s for pid, s in zip(pids, splits) if pid == p}
            assert labels == {"protein_test"}
        assert "train" in splits and "test" in splits


class TestSites:
    def test_hamming_one(self):
        ref = "MKTAA"
        var = "MRTAA"
        assert aligned_site_index(ref, var) == 1

    def test_identity_is_zero(self):
        assert aligned_site_index("MKT", "MKT") == 0

    def test_length_change_unaligned(self):
        assert aligned_site_index("MKTAA", "MKT") == -1

    def test_gather_centre_token(self):
        b, t, d = 1, 8, 4
        hidden = torch.arange(b * t * d, dtype=torch.float32).reshape(b, t, d)
        seqs = ["MKTAA"]
        # residue 1 → token 2 (CLS + 0)
        out = gather_site_windows(hidden, seqs, [1])
        assert out.shape == (1, 3, d)
        assert torch.equal(out[0, 1], hidden[0, 2])


class TestDamageMap:
    def test_monotone(self):
        assert damage_from_z(-3.0) > damage_from_z(-1.0) > damage_from_z(0.0) > damage_from_z(2.0)

    def test_display_bins(self):
        assert display_bin(0.8) == 0.70
        assert display_bin(0.45) == 0.40
        assert display_bin(0.1) == 0.00


class TestRankNet:
    def test_orders_same_protein(self):
        scores = torch.tensor([0.9, 0.1, 0.8, 0.2])
        z = torch.tensor([-3.0, 0.0, -2.0, 0.1])
        pids = ["a", "a", "b", "b"]
        miss = torch.tensor([True, True, True, True])
        loss = ranknet_loss(scores, z, pids, miss)
        flipped = ranknet_loss(1.0 - scores, z, pids, miss)
        assert float(loss) < float(flipped)

    def test_empty_when_one_per_gene(self):
        scores = torch.tensor([0.9, 0.1])
        z = torch.tensor([-3.0, 0.0])
        loss = ranknet_loss(scores, z, ["a", "b"], torch.tensor([True, True]))
        assert float(loss) == 0.0


class TestSiteCompare:
    def test_shape(self):
        cmp = SiteCompare(hidden_size=16)
        out = cmp(torch.randn(3, 3, 16), torch.randn(3, 3, 16))
        assert out.shape == (3, 16)


class TestProteinGroupSampler:
    def test_keeps_proteins_together(self):
        pids = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
        sampler = ProteinGroupBatchSampler(pids, batch_size=16, n_proteins_per_batch=2, seed=0)
        batches = list(sampler)
        assert batches
        for batch in batches:
            prot = {pids[i] for i in batch}
            assert len(prot) <= 4


class TestEncoders:
    def test_lof_is_35m(self):
        assert esm2_for_task("lof") == ESM2_LOF
        assert "35M" in esm2_for_task("lof")

    def test_pair_heads_are_650m(self):
        for task in ("mlof", "growth_gof", "amr_gof"):
            assert esm2_for_task(task) == ESM2_PAIR

    def test_yaml_override(self):
        cfg = {"esm2_model_name_lof": "facebook/esm2_t6_8M_UR50D", "esm2_model_name": "facebook/esm2_t30_150M_UR50D"}
        assert esm2_for_task("lof", cfg).endswith("8M_UR50D")
        assert esm2_for_task("mlof", cfg).endswith("150M_UR50D")


class TestTrainTasks:
    def test_gof_is_frozen(self):
        from plmlof.constants import TASKS, TRAIN_TASKS

        assert TRAIN_TASKS == ("lof", "mlof")
        assert "growth_gof" in TASKS and "amr_gof" in TASKS

    def test_sweep_names_unique(self):
        from pathlib import Path

        import yaml

        payload = yaml.safe_load(Path("configs/sweeps.yaml").read_text())
        for task in ("lof", "mlof"):
            names = [row["name"] for row in payload[task]]
            assert names
            assert len(names) == len(set(names))
