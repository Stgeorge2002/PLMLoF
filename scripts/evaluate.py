"""Evaluation script for PLMLoF model.

Usage:
    python scripts/evaluate.py --model outputs/checkpoints/model_best.pt --test-data data/processed/test.parquet
    python scripts/evaluate.py --model outputs/checkpoints/model_best.pt --tiny
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Reconstruct model from saved config or CLI flags
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

    # Dataset
    if args.tiny or args.test_data is None:
        dataset = SyntheticPLMLoFDataset(num_samples=30)
    else:
        dataset = PLMLoFDataset(args.test_data)

    collator = PLMLoFCollator(tokenizer_name=esm2_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    # Evaluate
    preds, labels, probs = evaluate(model, loader, args.device)
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


if __name__ == "__main__":
    main()
