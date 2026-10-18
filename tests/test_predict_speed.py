"""Production predict speed paths: length sort, ensemble trim, rule short-circuit."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from plmlof.embed import _LenSortedSeqs
from plmlof.inference.vcf_handler import VariantRecord
from plmlof.predictor import Predictor, discover_ensemble, unique_in_order


class TestUniqueAndLengthSort:
    def test_unique_in_order(self):
        assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_len_sorted_dataset(self):
        ds = _LenSortedSeqs(["AAAA", "A", "AAA", "AA"])
        assert [ds[i] for i in range(len(ds))] == ["A", "AA", "AAA", "AAAA"]


class TestDiscoverEnsemble:
    def test_respects_ensemble_json_order(self, tmp_path: Path):
        members = []
        for i in (2, 0, 1):
            ckpt = tmp_path / f"seed{i}" / "checkpoints" / "model_best.pt"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"x")
            members.append(f"seed{i}/checkpoints/model_best.pt")
        (tmp_path / "ensemble.json").write_text(json.dumps({"task": "lof", "members": members}))
        found = discover_ensemble(tmp_path)
        assert [p.name for p in found] == ["model_best.pt"] * 3
        assert "seed2" in str(found[0])
        assert "seed0" in str(found[1])

    def test_max_members_trims(self, tmp_path: Path):
        for i in range(3):
            ckpt = tmp_path / "lof" / f"seed{i}" / "checkpoints" / "model_best.pt"
            ckpt.parent.mkdir(parents=True)
            torch.save(
                {
                    "task": "lof",
                    "state_dict": _tiny_lof_state(),
                    "model_config": {
                        "hidden_size": 8,
                        "pool_strategy": "mean_max",
                        "use_cross_attention": False,
                        "head_hidden": 16,
                        "dropout": 0.0,
                        "esm2_model_name": "dummy",
                    },
                },
                ckpt,
            )
        (tmp_path / "lof" / "ensemble.json").write_text(json.dumps({
            "task": "lof",
            "members": [
                "seed0/checkpoints/model_best.pt",
                "seed1/checkpoints/model_best.pt",
                "seed2/checkpoints/model_best.pt",
            ],
        }))
        with patch.object(Predictor, "_get_encoder", side_effect=AssertionError("no ESM2")):
            pred = Predictor(
                tmp_path, device="cpu", batch_size=4,
                max_ensemble_members=1, compile_encoder=False, rule_short_circuit=True,
            )
        assert len(pred.ensembles) == 1


def _tiny_lof_state() -> dict:
    from plmlof.model import TaskNet

    net = TaskNet(hidden_size=8, task="lof", head_hidden=16, dropout=0.0)
    return net.state_dict()


class TestRuleShortCircuit:
    def _predictor(self, tmp_path: Path, **kwargs) -> Predictor:
        ckpt = tmp_path / "lof" / "seed0" / "checkpoints" / "model_best.pt"
        ckpt.parent.mkdir(parents=True)
        torch.save(
            {
                "task": "lof",
                "state_dict": _tiny_lof_state(),
                "model_config": {
                    "hidden_size": 8,
                    "pool_strategy": "mean_max",
                    "use_cross_attention": False,
                    "head_hidden": 16,
                    "dropout": 0.0,
                    "esm2_model_name": "dummy",
                },
            },
            ckpt,
        )
        (tmp_path / "lof" / "ensemble.json").write_text(json.dumps({
            "task": "lof",
            "members": ["seed0/checkpoints/model_best.pt"],
        }))
        defaults = dict(
            device="cpu", batch_size=4, max_ensemble_members=1,
            compile_encoder=False, rule_short_circuit=True,
        )
        defaults.update(kwargs)
        with patch.object(Predictor, "_get_encoder", side_effect=AssertionError("no ESM2")):
            return Predictor(tmp_path, **defaults)

    def test_identity_skips_embed(self, tmp_path: Path):
        pred = self._predictor(tmp_path)
        seq = "MKTLLLTLVVVTLAALG"
        recs = [VariantRecord(gene="g1", ref_protein=seq, var_protein=seq)]
        ensure = MagicMock(return_value=0)
        pred._ensure_embedded = ensure  # type: ignore[method-assign]
        out = pred.predict_records(recs, pair=True)
        ensure.assert_not_called()
        assert out[0]["route"] == "identity"
        assert out[0]["lof_score"] == 0.0

    def test_sure_wreck_skips_embed(self, tmp_path: Path):
        pred = self._predictor(tmp_path, rule_short_circuit=True)
        ref = "M" + "A" * 99
        var = "M" + "A" * 20
        recs = [VariantRecord(gene="g1", ref_protein=ref, var_protein=var)]
        ensure = MagicMock(return_value=0)
        pred._ensure_embedded = ensure  # type: ignore[method-assign]
        out = pred.predict_records(recs, pair=True)
        ensure.assert_not_called()
        assert out[0]["route"] == "sure_wreck"
        assert out[0]["wreck"] is True
        assert out[0]["lof_score"] >= 0.8

    def test_sure_wreck_disabled_needs_model(self, tmp_path: Path):
        pred = self._predictor(tmp_path, rule_short_circuit=False)
        ref = "M" + "A" * 99
        var = "M" + "A" * 20
        recs = [VariantRecord(gene="g1", ref_protein=ref, var_protein=var)]

        h = 8
        mean = torch.zeros(h)
        mx = torch.zeros(h)
        pred._seq_pool[ref] = (mean, mx)
        pred._seq_pool[var] = (mean, mx)
        ensure = MagicMock(side_effect=lambda seqs: 0)
        pred._ensure_embedded = ensure  # type: ignore[method-assign]
        out = pred.predict_records(recs, pair=True)
        ensure.assert_called()
        assert out[0]["route"] == "model"

    def test_missense_uses_model(self, tmp_path: Path):
        pred = self._predictor(tmp_path)
        ref = "MKTLLLTLVVVTLAALG"
        var = "MRTLLLTLVVVTLAALG"
        recs = [VariantRecord(gene="g1", ref_protein=ref, var_protein=var)]
        h = 8
        pred._seq_pool[ref] = (torch.zeros(h), torch.zeros(h))
        pred._seq_pool[var] = (torch.ones(h), torch.ones(h))
        pred._ensure_embedded = MagicMock(return_value=0)  # type: ignore[method-assign]
        out = pred.predict_records(recs, pair=True)
        assert out[0]["route"] == "model"
        assert 0.0 <= out[0]["lof_score"] <= 1.0


class TestExportProduction:
    def test_export_one_member(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import importlib.util

        src = tmp_path / "lof_8m"
        for i in range(3):
            ckpt = src / f"seed{i}" / "checkpoints" / "model_best.pt"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"ckpt")
        (src / "ensemble.json").write_text(json.dumps({
            "task": "lof",
            "members": [f"seed{i}/checkpoints/model_best.pt" for i in range(3)],
        }))
        (src / "null_scores.pt").write_bytes(b"null")
        dst = tmp_path / "prod"
        monkeypatch.setattr("sys.argv", [
            "export_production_ensemble.py",
            "--src", str(src), "--dst", str(dst), "--members", "1",
        ])
        spec = importlib.util.spec_from_file_location(
            "export_production_ensemble",
            Path(__file__).resolve().parents[1] / "scripts" / "export_production_ensemble.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.main()
        payload = json.loads((dst / "ensemble.json").read_text())
        assert len(payload["members"]) == 1
        assert (dst / payload["members"][0]).exists()
        assert (dst / "null_scores.pt").exists()
