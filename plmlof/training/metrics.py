"""Evaluation metrics for PLMLoF."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from plmlof import LABEL_MAP


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Predicted class indices [N].
        labels: True class indices [N].
        probabilities: Class probabilities [N, 3] (optional, for AUROC).

    Returns:
        Dict of metric name → value.
    """
    metrics = {}

    # Per-class and macro metrics
    metrics["macro_f1"] = float(f1_score(labels, predictions, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, predictions, average="weighted", zero_division=0))

    for cls_id, cls_name in LABEL_MAP.items():
        binary_labels = (labels == cls_id).astype(int)
        binary_preds = (predictions == cls_id).astype(int)

        metrics[f"{cls_name}_precision"] = float(
            precision_score(binary_labels, binary_preds, zero_division=0)
        )
        metrics[f"{cls_name}_recall"] = float(
            recall_score(binary_labels, binary_preds, zero_division=0)
        )
        metrics[f"{cls_name}_f1"] = float(
            f1_score(binary_labels, binary_preds, zero_division=0)
        )

    # AUROC (one-vs-rest)
    if probabilities is not None and len(np.unique(labels)) > 1:
        try:
            metrics["auroc_macro"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["auroc_macro"] = 0.0

    # Accuracy
    metrics["accuracy"] = float(np.mean(predictions == labels))

    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix.

    Returns:
        Confusion matrix [num_classes, num_classes].
    """
    return confusion_matrix(labels, predictions, labels=[0, 1, 2])


def format_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> str:
    """Format a human-readable classification report."""
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    return classification_report(
        labels, predictions, target_names=target_names, zero_division=0
    )
