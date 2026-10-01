"""v2 wreck rules, labels, empirical p-values, and tiny task nets."""

from __future__ import annotations

import random

import numpy as np
import torch

from plmlof.v2.domains import lof_prior, sample_events
from plmlof.v2.hmmer import parse_domtblout
from plmlof.v2.labels import channel_for_lof, lof_score_for_pair, lof_score_from_z
from plmlof.v2.metrics import gof_metrics, lof_metrics
from plmlof.v2.model import V2TaskNet
from plmlof.v2.stats import benjamini_hochberg, empirical_p
from plmlof.v2.wreck import wreck_call, wreck_grade


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
        # Without Pfam this is a length wreck.
        flagged_len, _ = wreck_call(ref, var)
        assert flagged_len


class TestLabels:
    def test_z_bins(self):
        assert lof_score_from_z(-2.5) == 0.70
        assert lof_score_from_z(-1.5) == 0.40
        assert lof_score_from_z(0.2) == 0.00
        assert lof_score_from_z(1.2) is None
        assert lof_score_from_z(2.5) is None

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
    def test_lof_ranked(self):
        z = np.array([-3.0, -2.5, 0.0, 0.1, 0.2])
        pred = np.array([0.8, 0.7, 0.1, 0.1, 0.05])
        target = np.array([0.7, 0.7, 0.0, 0.0, 0.0])
        wreck = np.zeros(5, dtype=bool)
        miss = np.ones(5, dtype=bool)
        m = lof_metrics(pred, target, z, wreck, miss)
        assert m["missense_spearman"] > 0.5

    def test_gof_precision(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        p = np.array([0.95, 0.91, 0.2, 0.1, 0.05, 0.01])
        m = gof_metrics(p, y, threshold=0.90)
        assert m["n_calls"] == 2
        assert m["precision_at_thr"] == 1.0


class TestTaskNet:
    def test_lof_range(self):
        net = V2TaskNet(hidden_size=16, task="lof", pool_strategy="mean_max")
        b, d = 4, 16
        raw = net.forward_from_pooled(
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, 12),
        )
        assert raw.shape == (b,)
        assert torch.all(raw >= 0) and torch.all(raw <= 1)

    def test_gof_logit(self):
        net = V2TaskNet(hidden_size=16, task="growth_gof", pool_strategy="mean_max")
        b, d = 3, 16
        logit = net.forward_from_pooled(
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, d), torch.randn(b, d),
            torch.randn(b, 12),
        )
        p = net.probability(logit)
        assert p.shape == (b,)
        assert torch.all(p >= 0) and torch.all(p <= 1)
