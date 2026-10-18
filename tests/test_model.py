"""Tests for ComparisonModule and ESM2 pooling helpers."""

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


class TestTaskNetLof:
    def test_lof_only(self):
        from plmlof.model import TaskNet

        net = TaskNet(hidden_size=16, task="lof")
        b = 2
        score = net.forward_from_pooled(
            torch.randn(b, 16), torch.randn(b, 16),
            torch.randn(b, 16), torch.randn(b, 16),
            torch.zeros(b, 12),
        )
        assert score.shape == (b,)
        assert torch.all((score >= 0) & (score <= 1))

    def test_rejects_other_tasks(self):
        from plmlof.model import TaskNet

        try:
            TaskNet(hidden_size=16, task="mlof")
        except ValueError as exc:
            assert "lof" in str(exc)
        else:
            raise AssertionError("expected ValueError for non-lof task")


class TestUniqueSequenceCache:
    def test_unique_in_order(self):
        from plmlof.predictor import unique_in_order

        assert unique_in_order(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]
        assert unique_in_order([]) == []

    def test_gather_uses_cache_without_reembed(self):
        from plmlof.predictor import Predictor

        # Bypass __init__ GPU/checkpoint load; exercise cache helpers only.
        pred = object.__new__(Predictor)
        pred.device = torch.device("cpu")
        pred.batch_size = 8
        pred.max_seq_length = 1024
        pred.esm_name = "dummy"
        pred._seq_pool = {}
        pred._encoders = {}

        calls = {"n": 0}

        def fake_ensure(sequences):
            missing = [s for s in sequences if s not in pred._seq_pool]
            # dedupe like production
            seen = set()
            uniq = []
            for s in missing:
                if s not in seen and s not in pred._seq_pool:
                    seen.add(s)
                    uniq.append(s)
            for s in uniq:
                pred._seq_pool[s] = (torch.ones(4) * (ord(s[0]) % 7), torch.zeros(4))
            calls["n"] += len(uniq)
            return len(uniq)

        pred._ensure_embedded = fake_ensure  # type: ignore[method-assign]
        refs = ["AAA", "BBB", "AAA"]
        vars_ = ["AAA", "CCC", "BBB"]
        # First pass embeds AAA,BBB,CCC
        assert pred._ensure_embedded(refs + vars_) == 3
        assert calls["n"] == 3
        # Second pass is free
        assert pred._ensure_embedded(refs + vars_) == 0
        ref_mean, ref_max = pred._gather_pooled(refs)
        var_mean, _ = pred._gather_pooled(vars_)
        assert ref_mean.shape == (3, 4)
        assert torch.equal(ref_mean[0], ref_mean[2])  # same AAA ref
        assert torch.equal(ref_mean[1], var_mean[2])  # BBB as ref then var
