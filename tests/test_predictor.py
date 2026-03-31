"""Tests for the PLMLoFPredictor (end-to-end inference)."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from plmlof.inference.predictor import PLMLoFPredictor
from plmlof.inference.vcf_handler import VariantRecord
from plmlof.inference.reference_cache import ReferenceCache


class TestReferenceCache:
    def test_put_and_get(self, tmp_output_dir):
        cache = ReferenceCache(cache_dir=str(tmp_output_dir / "cache"))
        emb = {
            "per_residue": torch.randn(1, 5, 32),
            "pooled": torch.randn(1, 32),
        }
        cache.put("MKTL", emb)
        assert cache.has("MKTL")

        retrieved = cache.get("MKTL", device="cpu")
        assert retrieved is not None
        assert torch.allclose(retrieved["pooled"], emb["pooled"])

    def test_missing_key(self, tmp_output_dir):
        cache = ReferenceCache(cache_dir=str(tmp_output_dir / "cache"))
        assert not cache.has("NONEXISTENT")
        assert cache.get("NONEXISTENT", device="cpu") is None

    def test_clear(self, tmp_output_dir):
        cache = ReferenceCache(cache_dir=str(tmp_output_dir / "cache"))
        emb = {"per_residue": torch.randn(1, 5, 32), "pooled": torch.randn(1, 32)}
        cache.put("MKTL", emb)
        cache.clear()
        assert not cache.has("MKTL")


class TestPredictor:
    def test_predict_records(self, plmlof_model):
        predictor = PLMLoFPredictor(model=plmlof_model, device="cpu", batch_size=2)
        records = [
            VariantRecord(
                gene="geneA",
                ref_protein="MKTLLLTLVVVTLAALG",
                var_protein="MKTLL*TLVVVTLAALG",
            ),
            VariantRecord(
                gene="geneB",
                ref_protein="MKTLLLTLVVVTLAALG",
                var_protein="MKTLLLTLVVVTLAALG",
            ),
        ]
        results = predictor.predict_records(records, compute_attribution=True)
        assert len(results) == 2
        for r in results:
            assert "prediction" in r
            assert r["prediction"] in ("LoF", "WT", "GoF")
            assert "confidence" in r
            assert 0.0 <= r["confidence"] <= 1.0
            assert "probabilities" in r

    def test_predict_fasta(self, plmlof_model, tmp_path):
        ref_fasta = tmp_path / "ref.fasta"
        var_fasta = tmp_path / "var.fasta"

        ref_fasta.write_text(">gene1\nMKTLLLTLVVVTLAALG\n>gene2\nMKTLLLTLVVVTLAALG\n")
        var_fasta.write_text(">gene1\nMKTLL*TLVVVTLAALG\n>gene2\nMRTLLLTLVVVTLAALG\n")

        predictor = PLMLoFPredictor(model=plmlof_model, device="cpu")
        results = predictor.predict_fasta(
            str(var_fasta), reference_fasta=str(ref_fasta), compute_attribution=False
        )
        assert len(results) == 2
        for r in results:
            assert r["prediction"] in ("LoF", "WT", "GoF")

    def test_predict_empty_records(self, plmlof_model):
        predictor = PLMLoFPredictor(model=plmlof_model, device="cpu")
        results = predictor.predict_records([], compute_attribution=False)
        assert results == []
