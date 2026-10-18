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
