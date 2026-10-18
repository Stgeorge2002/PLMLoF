"""Wreck rules, labels, empirical p-values, and LoF task nets."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from plmlof.dataset import lof_train_keep_indices
from plmlof.domains import PRIOR_SURE, PRIOR_TAIL_STOP, lof_prior, sample_events
from plmlof.encoders import ESM2_LOF, ESM2_LOF_FAST, esm2_for_task
from plmlof.hmmer import parse_domtblout
from plmlof.labels import channel_for_lof, damage_from_z, display_bin, lof_score_for_pair, lof_score_from_z
from plmlof.metrics import lof_metrics
from plmlof.model import TaskNet
from plmlof.sites import aa_hamming, aligned_site_index, missense_sites
from plmlof.splits import parse_tasks, split_species
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
        assert prior == PRIOR_TAIL_STOP

    def test_post_domain_stop_not_sure_even_if_half_gone(self):
        ref = "M" + "A" * 99
        var = ref[:50]
        domains = [(10, 40)]
        flagged, _ = wreck_call(ref, var, domains=domains)
        assert not flagged
        _, _, prior = wreck_grade(ref, var, domains=domains)
        assert prior == PRIOR_TAIL_STOP
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
        assert lof_score_for_pair("M" + "A" * 99, "M" + "A" * 90, wreck_type="stop_tail") == PRIOR_TAIL_STOP
        assert channel_for_lof(PRIOR_TAIL_STOP, "stop_tail", None) == "tail_wreck"

    def test_stored_score_wins(self):
        ref = "M" + "A" * 80
        var = "M" + "A" * 10
        assert lof_score_for_pair(ref, var, wreck_type="stop", lof_score=0.4) == 0.4


class TestDomainGeometry:
    def test_stop_before_first_domain_is_sure(self):
        score, geom = lof_prior("stop", 5, 100, [(20, 60)])
        assert score == PRIOR_SURE
        assert geom == "pre_domain"

    def test_frameshift_inside_domain_is_sure(self):
        score, geom = lof_prior("frameshift", 30, 100, [(20, 60)])
        assert score == PRIOR_SURE
        assert geom == "in_domain"

    def test_linker_frameshift_destroys_downstream(self):
        score, geom = lof_prior("frameshift", 45, 100, [(10, 40), (60, 80)])
        assert geom == "linker"
        assert score > PRIOR_TAIL_STOP
        assert score < PRIOR_SURE

    def test_stop_after_last_domain_is_tail(self):
        score, geom = lof_prior("stop", 90, 100, [(10, 40), (50, 70)])
        assert geom == "tail"
        assert score <= PRIOR_TAIL_STOP

    def test_whole_domain_drop_of_one_of_two(self):
        score, geom = lof_prior(
            "domain_drop", 10, 100, [(10, 40), (50, 80)],
            last_affected=40, n_domains_dropped=1, n_domains_remaining=1,
        )
        assert geom == "in_domain"
        assert 0.5 <= score < PRIOR_SURE

    def test_dropping_the_only_domain_is_sure(self):
        score, _ = lof_prior(
            "domain_drop", 10, 100, [(10, 80)],
            last_affected=80, n_domains_dropped=1, n_domains_remaining=0,
        )
        assert score == PRIOR_SURE

    def test_in_domain_vs_extra_missense(self):
        in_d, _ = lof_prior("missense", 25, 100, [(10, 40)])
        extra, _ = lof_prior("missense", 90, 100, [(10, 40)])
        assert in_d == 0.7
        assert extra == 0.15

    def test_no_hmmer_last_tenth_is_tail(self):
        early, _ = lof_prior("stop", 40, 100, ())
        tail, _ = lof_prior("stop", 96, 100, ())
        assert early == PRIOR_SURE
        assert tail <= PRIOR_TAIL_STOP

    def test_pure_tail_below_domain_hit(self):
        domains = [(10, 40), (50, 80)]
        early, _ = lof_prior("stop", 20, 100, domains)
        tail, _ = lof_prior("stop", 90, 100, domains)
        assert early == PRIOR_SURE
        assert tail <= PRIOR_TAIL_STOP
        assert early > tail

    def test_more_domains_dropped_hurts_more(self):
        domains = [(10, 30), (40, 60), (70, 90)]
        one, _ = lof_prior(
            "domain_drop", 10, 100, domains,
            last_affected=30, n_domains_dropped=1, n_domains_remaining=2,
        )
        all_d, _ = lof_prior(
            "domain_drop", 10, 100, domains,
            last_affected=90, n_domains_dropped=3, n_domains_remaining=0,
        )
        assert all_d > one

    def test_sample_events_mix_sure_and_tail(self):
        rng = random.Random(0)
        prot = "M" + "A" * 119
        dna = "ATG" + "GCA" * 119
        domains = [(10, 40), (50, 80)]
        evs = sample_events(prot, dna, domains, rng, n=3)
        assert evs
        scores = [e.lof_score for e in evs]
        assert max(scores) > min(scores)
        assert any(e.geometry == "tail" or e.lof_score <= 0.5 for e in evs)
        assert not any("missense" in e.kind for e in evs)

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

    def test_nonoverlapping_keeps_best_score(self):
        from plmlof.hmmer import finalize_domain_hits

        hits = [
            {"protein_id": "p", "hmm_name": "A", "evalue": 1e-10, "score": 50.0, "start": 1, "end": 50},
            {"protein_id": "p", "hmm_name": "B", "evalue": 1e-8, "score": 80.0, "start": 40, "end": 90},
            {"protein_id": "p", "hmm_name": "C", "evalue": 1e-3, "score": 100.0, "start": 100, "end": 120},
            {"protein_id": "p", "hmm_name": "D", "evalue": 1e-12, "score": 40.0, "start": 130, "end": 150},
        ]
        out = finalize_domain_hits(hits, max_evalue=1e-5)
        names = {h["hmm_name"] for h in out}
        assert "C" not in names  # fails e-value
        assert "B" in names  # wins overlap with A
        assert "A" not in names
        assert "D" in names

    def test_pfam_accession_parse(self):
        from plmlof.mmseqs_domains import pfam_accession

        assert pfam_accession("A0A060YUX3_9BACT/26-209 PF13671.1;Methyltransf_25;") == "PF13671"
        assert pfam_accession("PF00001.21") == "PF00001"

    def test_map_span_same_length(self):
        from plmlof.seqcluster import map_span_to_member

        rep = "M" + "A" * 99
        mem = "M" + "A" * 98 + "V"
        assert map_span_to_member(rep, mem, 10, 40) == (10, 40)

    def test_expand_hits_copies_rep(self):
        from plmlof.seqcluster import expand_hits_to_members

        rep = "rep"
        mem = "mem"
        seqs = {rep: "M" + "A" * 99, mem: "M" + "A" * 99}
        hits = [{
            "protein_id": rep, "hmm_name": "PF1", "evalue": 1e-20,
            "score": 100.0, "start": 10, "end": 50,
        }]
        out = expand_hits_to_members(hits, {rep: rep, mem: rep}, seqs)
        by = {h["protein_id"]: h for h in out}
        assert rep in by and mem in by
        assert by[mem]["start"] == 10 and by[mem]["end"] == 50


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

    def test_lof_uses_ref_var_comparison(self):
        net = TaskNet(hidden_size=16, task="lof", pool_strategy="mean_max")
        net.eval()
        torch.manual_seed(0)
        b, d = 2, 16
        ref_mean, ref_max = torch.randn(b, d), torch.randn(b, d)
        var_mean, var_max = torch.randn(b, d), torch.randn(b, d)
        nuc = torch.zeros(b, 12)
        with torch.no_grad():
            delta = net.forward_from_pooled(ref_mean, ref_max, var_mean, var_max, nuc)
            identity = net.forward_from_pooled(ref_mean, ref_max, ref_mean, ref_max, torch.zeros_like(nuc))
            other_ref = net.forward_from_pooled(torch.randn(b, d), torch.randn(b, d), var_mean, var_max, nuc)
        assert delta.shape == (b,)
        assert not torch.allclose(delta, other_ref, atol=1e-5)
        # Identity path is a distinct feature regime (zero diff); not equal to random delta.
        assert not torch.allclose(delta, identity, atol=1e-5)

    def test_load_net_matches_nondefault_head(self, tmp_path):
        from plmlof.predictor import _load_net

        net = TaskNet(hidden_size=16, task="lof", head_hidden=256, dropout=0.1)
        net.eval()
        path = tmp_path / "model_best.pt"
        torch.save(
            {
                "task": "lof",
                "state_dict": net.state_dict(),
                "model_config": {"hidden_size": 16, "pool_strategy": "mean_max"},
            },
            path,
        )
        loaded = _load_net(path, torch.device("cpu"))
        assert loaded.head.mlp[0].out_features == 256
        x = torch.randn(3, 16)
        with torch.no_grad():
            a = net.forward_from_cache(x, x, x, x, torch.zeros(3, 12))
            b = loaded.forward_from_cache(x, x, x, x, torch.zeros(3, 12))
        assert torch.allclose(a, b, atol=1e-6)


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


class TestSplits:
    def test_parse_tasks(self):
        assert parse_tasks(None) == ["lof"]
        assert parse_tasks(["lof"]) == ["lof"]

    def test_parse_tasks_rejects_non_lof(self):
        with pytest.raises(ValueError, match="Unknown tasks"):
            parse_tasks(["growth_gof"])
        with pytest.raises(ValueError, match="Unknown tasks"):
            parse_tasks(["mlof"])

    def test_split_species(self):
        species = [f"sp{i}" for i in range(20)]
        mapping = split_species(species, seed=0)
        assert set(mapping) == set(species)
        assert {"train", "val", "test"} <= set(mapping.values())
        assert mapping == split_species(species, seed=0)


class TestSites:
    def test_hamming_one(self):
        ref = "MKTAA"
        var = "MRTAA"
        assert aligned_site_index(ref, var) == 1
        assert aa_hamming(ref, var) == 1

    def test_identity_is_zero(self):
        assert aligned_site_index("MKT", "MKT") == 0
        assert aa_hamming("MKT", "MKT") == 0

    def test_length_change_unaligned(self):
        assert aligned_site_index("MKTAA", "MKT") == -1
        assert aa_hamming("MKTAA", "MKT") == 0

    def test_all_same_length_missense(self):
        assert missense_sites("MKTAA", "MRTVA") == [1, 3]
        assert missense_sites("MKT", "MKT") == []
        assert missense_sites("MKT", "MK") == []


class TestDamageMap:
    def test_monotone(self):
        assert damage_from_z(-3.0) > damage_from_z(-1.0) > damage_from_z(0.0) > damage_from_z(2.0)

    def test_display_bins(self):
        assert display_bin(0.8) == 0.70
        assert display_bin(0.45) == 0.40
        assert display_bin(0.1) == 0.00


class TestEncoders:
    def test_lof_is_35m(self):
        assert esm2_for_task("lof") == ESM2_LOF
        assert "35M" in esm2_for_task("lof")

    def test_yaml_override(self):
        cfg = {"esm2_model_name_lof": ESM2_LOF_FAST}
        assert esm2_for_task("lof", cfg) == ESM2_LOF_FAST

    def test_rejects_non_lof(self):
        with pytest.raises(ValueError, match="lof"):
            esm2_for_task("mlof")


class TestTrainTasks:
    def test_product_is_lof_only(self):
        from plmlof.constants import TASKS, TRAIN_TASKS

        assert TRAIN_TASKS == ("lof",)
        assert TASKS == ("lof",)


class TestDistillSoftTargets:
    def test_soft_target_dataset_length(self):
        from plmlof.dataset import SoftTargetDataset

        class _Tiny:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {"target": float(idx), "sample_weight": 1.0}

        soft = torch.tensor([0.1, 0.5, 0.9])
        ds = SoftTargetDataset(_Tiny(), soft)  # type: ignore[arg-type]
        assert len(ds) == 3
        assert float(ds[1]["soft_target"]) == pytest.approx(0.5)

    def test_trainer_distill_alpha_mixes_soft(self):
        from torch.utils.data import DataLoader

        from plmlof.model import TaskNet
        from plmlof.trainer import Trainer

        class _Row(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                d = 8
                return {
                    "ref_mean": torch.randn(d),
                    "ref_max": torch.randn(d),
                    "var_mean": torch.randn(d),
                    "var_max": torch.randn(d),
                    "nucleotide_features": torch.zeros(12),
                    "target": torch.tensor(0.0),
                    "soft_target": torch.tensor(1.0),
                    "sample_weight": torch.tensor(1.0),
                    "dms_zscore": torch.tensor(float("nan")),
                    "is_wreck": torch.tensor(False),
                    "is_missense": torch.tensor(False),
                    "gene": "g",
                    "protein_id": "p",
                }

        loader = DataLoader(_Row(), batch_size=2)
        net = TaskNet(hidden_size=8, task="lof")
        trainer = Trainer(
            net, loader, loader, device="cpu", output_dir=".", distill_alpha=1.0,
        )
        batch = next(iter(loader))
        loss_soft, _ = trainer._step(batch)
        trainer.distill_alpha = 0.0
        loss_hard, _ = trainer._step(batch)
        assert torch.isfinite(loss_soft) and torch.isfinite(loss_hard)


class TestDomainIndex:
    def test_parquet_lookup_grades_tail(self, tmp_path):
        import pandas as pd

        from plmlof.domains import DomainIndex, protein_id
        from plmlof.wreck import wreck_grade

        seq = "M" + "A" * 99
        pid = protein_id(seq)
        df = pd.DataFrame({"protein_id": [pid, pid], "start": [10, 40], "end": [30, 70]})
        path = tmp_path / "pfam.parquet"
        df.to_parquet(path, index=False)
        index = DomainIndex(parquet=path)
        index.ensure([seq])
        assert index.spans(seq) == [(10, 30), (40, 70)]
        var = seq[:90]
        _, kind, prior = wreck_grade(seq, var, domains=index.spans(seq))
        assert prior == PRIOR_TAIL_STOP
        assert "tail" in kind
