"""Tests for the PLMLoF model components and full model."""

from __future__ import annotations

import torch
import pytest


class TestESM2Encoder:
    """Tests for plmlof.models.esm2_encoder.ESM2Encoder."""

    def test_forward_shape(self, tiny_model_name):
        from plmlof.models.esm2_encoder import ESM2Encoder

        enc = ESM2Encoder(model_name=tiny_model_name, freeze=True, lora_config=None)
        tokens = enc.tokenize(["MKTL", "MKTLVV"], max_length=32)
        out = enc(tokens["input_ids"], tokens["attention_mask"])

        assert "per_residue" in out
        assert "pooled" in out
        B = 2
        D = enc.hidden_size
        assert out["pooled"].shape == (B, D)
        assert out["per_residue"].shape[0] == B
        assert out["per_residue"].shape[2] == D

    def test_frozen_parameters(self, tiny_model_name):
        from plmlof.models.esm2_encoder import ESM2Encoder

        enc = ESM2Encoder(model_name=tiny_model_name, freeze=True, lora_config=None)
        for param in enc.model.parameters():
            assert not param.requires_grad

    def test_lora_switch(self, tiny_model_name, lora_config):
        from plmlof.models.esm2_encoder import ESM2Encoder

        enc = ESM2Encoder(model_name=tiny_model_name, freeze=True, lora_config=lora_config)

        enc.disable_lora()
        for name, p in enc.model.named_parameters():
            if "lora_" in name:
                assert not p.requires_grad

        enc.enable_lora()
        for name, p in enc.model.named_parameters():
            if "lora_" in name:
                assert p.requires_grad


class TestComparisonModule:
    """Tests for plmlof.models.comparison.ComparisonModule."""

    def test_output_shape(self):
        from plmlof.models.comparison import ComparisonModule

        D = 64
        comp = ComparisonModule(hidden_size=D, pool_strategy="mean_max")

        B, L_ref, L_var = 2, 10, 8
        ref_emb = {"per_residue": torch.randn(B, L_ref, D), "pooled": torch.randn(B, D)}
        var_emb = {"per_residue": torch.randn(B, L_var, D), "pooled": torch.randn(B, D)}
        ref_mask = torch.ones(B, L_ref)
        var_mask = torch.ones(B, L_var)

        out = comp(ref_emb, var_emb, ref_mask, var_mask)
        assert out.shape == (B, comp.output_size)

    def test_mean_strategy(self):
        from plmlof.models.comparison import ComparisonModule

        D = 32
        comp = ComparisonModule(hidden_size=D, pool_strategy="mean")
        assert comp.output_size == 4 * D  # after projection


class TestClassifierHead:
    """Tests for plmlof.models.classifier.ClassifierHead."""

    def test_output_shape(self):
        from plmlof.models.classifier import ClassifierHead

        head = ClassifierHead(input_size=128, hidden_dims=[32, 16], num_classes=3)
        out = head(torch.randn(4, 128))
        assert out.shape == (4, 3)

    def test_gradient_flows(self):
        from plmlof.models.classifier import ClassifierHead

        head = ClassifierHead(input_size=64, num_classes=3)
        x = torch.randn(2, 64, requires_grad=True)
        logits = head(x)
        logits.sum().backward()
        assert x.grad is not None


class TestPLMLoFModel:
    """Integration tests for the full PLMLoFModel."""

    def test_forward_output_shape(self, plmlof_model, collator, synthetic_dataset):
        batch = collator([synthetic_dataset[i] for i in range(3)])
        with torch.no_grad():
            logits = plmlof_model(
                ref_input_ids=batch["ref_input_ids"],
                ref_attention_mask=batch["ref_attention_mask"],
                var_input_ids=batch["var_input_ids"],
                var_attention_mask=batch["var_attention_mask"],
                nucleotide_features=batch["nucleotide_features"],
            )
        assert logits.shape == (3, 3)

    def test_predict_keys(self, plmlof_model, collator, synthetic_dataset):
        batch = collator([synthetic_dataset[0]])
        with torch.no_grad():
            result = plmlof_model.predict(
                ref_input_ids=batch["ref_input_ids"],
                ref_attention_mask=batch["ref_attention_mask"],
                var_input_ids=batch["var_input_ids"],
                var_attention_mask=batch["var_attention_mask"],
                nucleotide_features=batch["nucleotide_features"],
            )
        assert "logits" in result
        assert "probabilities" in result
        assert "predictions" in result
        assert result["probabilities"].shape == (1, 3)
        assert result["predictions"].shape == (1,)

    def test_gradient_through_classifier(self, plmlof_model, collator, synthetic_dataset):
        """Verify gradients flow through the classifier head (Stage 1)."""
        model = plmlof_model
        model.train()
        batch = collator([synthetic_dataset[0]])
        logits = model(
            ref_input_ids=batch["ref_input_ids"],
            ref_attention_mask=batch["ref_attention_mask"],
            var_input_ids=batch["var_input_ids"],
            var_attention_mask=batch["var_attention_mask"],
            nucleotide_features=batch["nucleotide_features"],
        )
        loss = logits.sum()
        loss.backward()

        # Classifier should have gradients
        for p in model.classifier.parameters():
            if p.requires_grad:
                assert p.grad is not None

        model.eval()

    def test_lora_toggle(self, plmlof_model):
        plmlof_model.enable_lora_training()
        for name, p in plmlof_model.encoder.model.named_parameters():
            if "lora_" in name:
                assert p.requires_grad
        plmlof_model.disable_lora_training()
        for name, p in plmlof_model.encoder.model.named_parameters():
            if "lora_" in name:
                assert not p.requires_grad
