"""Evaluation script for PLMLoF model.

Usage:
    python scripts/evaluate.py --model outputs/checkpoints/model_best.pt --test-data data/processed/test.parquet
    python scripts/evaluate.py --model outputs/checkpoints/model_best.pt --tiny

Supports both full-model and cached-training checkpoints automatically.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from plmlof.models.plmlof_model import PLMLoFModel
from plmlof.data.dataset import PLMLoFDataset, SyntheticPLMLoFDataset
from plmlof.data.collator import PLMLoFCollator
from plmlof.training.metrics import compute_metrics, compute_confusion_matrix, format_classification_report
from plmlof import LABEL_MAP

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PLMLoF model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tiny", action="store_true", help="Use tiny model + synthetic data")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in data_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        logits = model(
            ref_input_ids=batch["ref_input_ids"],
            ref_attention_mask=batch["ref_attention_mask"],
            var_input_ids=batch["var_input_ids"],
            var_attention_mask=batch["var_attention_mask"],
            nucleotide_features=batch["nucleotide_features"],
        )

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].cpu().numpy())
        all_probs.extend(probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)

    # ── Cached-training checkpoint ────────────────────────────────────────
    if checkpoint.get("cached_training"):
        logger.info("Detected cached-training checkpoint — using embedding-based evaluation")
        model_cfg = checkpoint.get("model_config", {})
        esm2_name = model_cfg.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D")
        pool_strategy = model_cfg.get("pool_strategy", "mean_max")

        from plmlof.models.comparison import ComparisonModule
        from plmlof.models.classifier import ClassifierHead, RegressionHead
        from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES
        from transformers import AutoTokenizer, AutoModel

        # Load comparison + classifier
        # Infer hidden_size from checkpoint state dict
        comparison_state = checkpoint["comparison_state_dict"]
        # _pre_norm.weight shape is [raw_size], raw_size = 8*D for mean_max, 4*D for mean
        pre_norm_shape = comparison_state["_pre_norm.weight"].shape[0]
        if pool_strategy == "mean_max":
            hidden_size = pre_norm_shape // 8
        else:
            hidden_size = pre_norm_shape // 4
        comparison = ComparisonModule(hidden_size=hidden_size, pool_strategy=pool_strategy)
        comparison.load_state_dict(checkpoint["comparison_state_dict"])

        classifier_input = comparison.output_size + NUM_NUCLEOTIDE_FEATURES
        classifier = ClassifierHead(
            input_size=classifier_input,
            hidden_dims=model_cfg.get("classifier_hidden_dims", [256, 64]),
            num_classes=3,
            dropout=model_cfg.get("classifier_dropout", 0.3),
        )
        classifier.load_state_dict(checkpoint["classifier_state_dict"])

        # Load regression head if present
        regressor = None
        if model_cfg.get("has_regressor") and "regressor_state_dict" in checkpoint:
            regressor = RegressionHead(input_size=classifier_input)
            regressor.load_state_dict(checkpoint["regressor_state_dict"])
            logger.info("Loaded regression head for DMS score prediction")

        # Load feature normalization (LayerNorm for engineered features)
        feature_norm = nn.LayerNorm(NUM_NUCLEOTIDE_FEATURES)
        if "feature_norm_state_dict" in checkpoint:
            feature_norm.load_state_dict(checkpoint["feature_norm_state_dict"])
            logger.info("Loaded feature normalization")

        # Load cross-attention if used
        cross_attn = None
        if model_cfg.get("use_cross_attention") and "cross_attn_state_dict" in checkpoint:
            from plmlof.models.comparison import PooledCrossAttention
            ca_heads = model_cfg.get("cross_attn_heads", 4)
            ca_dropout = model_cfg.get("cross_attn_dropout", 0.1)
            cross_attn = PooledCrossAttention(
                hidden_size=hidden_size, num_heads=ca_heads, dropout=ca_dropout,
            )
            cross_attn.load_state_dict(checkpoint["cross_attn_state_dict"])
            logger.info("Loaded cross-attention module")

        device = torch.device(args.device)
        comparison = comparison.to(device).eval()
        classifier = classifier.to(device).eval()
        feature_norm = feature_norm.to(device).eval()
        if cross_attn is not None:
            cross_attn = cross_attn.to(device).eval()
        if regressor is not None:
            regressor = regressor.to(device).eval()

        # Load ESM2 for embedding the test set
        logger.info(f"Loading ESM2: {esm2_name}")
        tokenizer = AutoTokenizer.from_pretrained(esm2_name)
        esm2 = AutoModel.from_pretrained(esm2_name).to(device)
        esm2.eval()
        for p in esm2.parameters():
            p.requires_grad = False

        # Dataset
        if args.tiny or args.test_data is None:
            dataset = SyntheticPLMLoFDataset(num_samples=30)
        else:
            dataset = PLMLoFDataset(args.test_data)

        max_len = 1024

        def _collate_eval(batch):
            ref_seqs = [s["ref_protein"] for s in batch]
            var_seqs = [s["var_protein"] for s in batch]
            all_seqs = ref_seqs + var_seqs
            enc = tokenizer(all_seqs, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            nuc = torch.stack([s["nucleotide_features"] for s in batch])
            labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)
            dms_scores = torch.tensor([s.get("dms_score", 0.0) for s in batch], dtype=torch.float32)
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "n_ref": len(ref_seqs),
                "nucleotide_features": nuc,
                "labels": labels,
                "dms_scores": dms_scores,
            }

        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=_collate_eval,
                            num_workers=4, pin_memory=True)

        all_preds, all_labels, all_probs = [], [], []
        all_reg_preds, all_dms_targets = [], []

        @torch.no_grad()
        def _pool(emb, mask):
            m_f = mask.unsqueeze(-1).float()
            mean_p = (emb * m_f).sum(1) / m_f.sum(1).clamp(min=1)
            emb_masked = emb.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            max_p = emb_masked.max(dim=1).values
            max_p = max_p.masked_fill(max_p == float("-inf"), 0.0)
            return mean_p, max_p

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                n_ref = batch["n_ref"]
                nuc = batch["nucleotide_features"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = esm2(ids, attention_mask=mask).last_hidden_state

                ref_out, var_out = out[:n_ref], out[n_ref:]
                ref_mask, var_mask = mask[:n_ref], mask[n_ref:]

                ref_mean, ref_max = _pool(ref_out, ref_mask)
                var_mean, var_max = _pool(var_out, var_mask)

                # Optional cross-attention between pooled vectors
                if cross_attn is not None:
                    tokens = torch.stack([ref_mean, ref_max, var_mean, var_max], dim=1)
                    tokens = cross_attn(tokens)
                    ref_mean, ref_max, var_mean, var_max = tokens[:, 0], tokens[:, 1], tokens[:, 2], tokens[:, 3]

                ref_pool = torch.cat([ref_mean, ref_max], dim=-1)
                var_pool = torch.cat([var_mean, var_max], dim=-1)
                diff_pool = ref_pool - var_pool
                prod_pool = ref_pool * var_pool
                comp = torch.cat([diff_pool, prod_pool, ref_pool, var_pool], dim=-1)
                comp = comparison.project(comp)
                nuc_normed = feature_norm(nuc)
                features = torch.cat([comp, nuc_normed], dim=-1)
                logits = classifier(features)

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds_batch = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds_batch)
                all_labels.extend(batch["labels"].numpy())
                all_probs.extend(probs)

                if regressor is not None:
                    reg_pred = regressor(features)
                    all_reg_preds.extend(reg_pred.cpu().numpy())
                    all_dms_targets.extend(batch["dms_scores"].numpy())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        probs = np.array(all_probs)
        reg_preds = np.array(all_reg_preds) if all_reg_preds else None
        dms_targets = np.array(all_dms_targets) if all_dms_targets else None

    else:
        # ── Full-model checkpoint ─────────────────────────────────────────
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_cfg = checkpoint.get("model_config", {})
        if args.tiny:
            esm2_name = "facebook/esm2_t6_8M_UR50D"
        else:
            esm2_name = model_cfg.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D")

        model = PLMLoFModel(
            esm2_model_name=esm2_name,
            freeze_esm2=True,
            lora_config=model_cfg.get("lora_config"),
            classifier_hidden_dims=model_cfg.get("classifier_hidden_dims", [256, 64]),
            classifier_dropout=model_cfg.get("classifier_dropout", 0.3),
            pool_strategy=model_cfg.get("pool_strategy", "mean_max"),
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(args.device)

        if args.tiny or args.test_data is None:
            dataset = SyntheticPLMLoFDataset(num_samples=30)
        else:
            dataset = PLMLoFDataset(args.test_data)

        collator = PLMLoFCollator(tokenizer_name=esm2_name)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

        preds, labels, probs = evaluate(model, loader, args.device)
        reg_preds = None
        dms_targets = None

    metrics = compute_metrics(preds, labels, probs)
    cm = compute_confusion_matrix(preds, labels)
    report = format_classification_report(preds, labels)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy:   {metrics['accuracy']:.4f}")
    print(f"Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"AUROC:      {metrics.get('auroc_macro', 'N/A')}")
    print(f"\nConfusion Matrix:")
    print(f"{'':>10} {'Pred LoF':>10} {'Pred WT':>10} {'Pred GoF':>10}")
    for i, row in enumerate(cm):
        print(f"{'True '+LABEL_MAP[i]:>10} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
    print(f"\nClassification Report:\n{report}")

    # Regression metrics
    if reg_preds is not None and len(reg_preds) > 0:
        from scipy.stats import spearmanr, pearsonr
        rho, rho_p, r, r_p = 0.0, 1.0, 0.0, 1.0
        if dms_targets.std() > 0 and reg_preds.std() > 0:
            rho, rho_p = spearmanr(dms_targets, reg_preds)
            r, r_p = pearsonr(dms_targets, reg_preds)
        mae = float(np.mean(np.abs(dms_targets - reg_preds)))
        mse = float(np.mean((dms_targets - reg_preds) ** 2))
        print("=" * 60)
        print("REGRESSION RESULTS (DMS z-score prediction)")
        print("=" * 60)
        print(f"  Spearman ρ:   {rho:.4f} (p={rho_p:.2e})")
        print(f"  Pearson r:    {r:.4f} (p={r_p:.2e})")
        print(f"  MAE:          {mae:.4f}")
        print(f"  MSE:          {mse:.4f}")


if __name__ == "__main__":
    main()
