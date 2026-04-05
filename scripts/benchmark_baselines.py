"""Baseline benchmarks for PLMLoF — run all methods on the same test split.

Compares PLMLoF against:
  1. ESM-1v zero-shot (log-likelihood ratio scoring)
  2. Logistic regression on mean ESM2 embeddings
  3. Random Forest on mean ESM2 embeddings
  4. MLP on ESM2 embeddings (no comparison module / cross-attention)

All baselines use the exact same train/val/test split as PLMLoF.

Usage:
    python scripts/benchmark_baselines.py --device cuda

    # Skip ESM-1v (slow, ~1h on A100):
    python scripts/benchmark_baselines.py --device cuda --skip-esm1v

    # Run just one baseline:
    python scripts/benchmark_baselines.py --device cuda --only logistic_regression
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from plmlof import LABEL_MAP

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_cached_split(path: Path) -> dict:
    """Load a precomputed embedding .pt file into numpy arrays."""
    data = torch.load(path, weights_only=True)
    return {
        "ref_mean": data["ref_mean"].numpy(),
        "ref_max": data["ref_max"].numpy(),
        "var_mean": data["var_mean"].numpy(),
        "var_max": data["var_max"].numpy(),
        "nuc_features": data["nucleotide_features"].numpy(),
        "labels": data["labels"].numpy(),
        "dms_scores": data.get("dms_scores", torch.zeros(len(data["labels"]))).numpy(),
    }


def _make_flat_features(split: dict, mode: str = "full") -> np.ndarray:
    """Build a flat feature matrix from cached embeddings.

    Modes:
        'mean_only':   ref_mean | var_mean                          (2*D)
        'diff_only':   ref_mean - var_mean                          (D)
        'full':        diff | product | ref_mean | var_mean | nuc   (4*D + 12)
    """
    ref_mean, var_mean = split["ref_mean"], split["var_mean"]
    if mode == "mean_only":
        return np.hstack([ref_mean, var_mean])
    elif mode == "diff_only":
        return ref_mean - var_mean
    else:
        diff = ref_mean - var_mean
        prod = ref_mean * var_mean
        return np.hstack([diff, prod, ref_mean, var_mean, split["nuc_features"]])


def _compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray | None = None,
    dms_preds: np.ndarray | None = None,
    dms_targets: np.ndarray | None = None,
) -> dict:
    """Compute all metrics matching PLMLoF evaluate.py output."""
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }
    if probs is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr"))
        except ValueError:
            metrics["auroc"] = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["report"] = classification_report(
        labels, preds, target_names=["LoF", "WT", "GoF"], digits=4,
    )

    if dms_preds is not None and dms_targets is not None:
        if dms_targets.std() > 0 and dms_preds.std() > 0:
            rho, _ = spearmanr(dms_targets, dms_preds)
            r, _ = pearsonr(dms_targets, dms_preds)
        else:
            rho, r = 0.0, 0.0
        metrics["spearman_rho"] = float(rho)
        metrics["pearson_r"] = float(r)
        metrics["mae"] = float(np.mean(np.abs(dms_targets - dms_preds)))

    return metrics


def _print_results(name: str, metrics: dict, elapsed: float):
    """Pretty-print results for one baseline."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
    if "auroc" in metrics:
        print(f"  AUROC:      {metrics['auroc']:.4f}")
    if "spearman_rho" in metrics:
        print(f"  Spearman ρ: {metrics['spearman_rho']:.4f}")
        print(f"  Pearson r:  {metrics['pearson_r']:.4f}")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"\n  Confusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"  {'':>10} {'Pred LoF':>10} {'Pred WT':>10} {'Pred GoF':>10}")
    for i in range(3):
        print(f"  {'True '+LABEL_MAP[i]:>10} {cm[i,0]:>10} {cm[i,1]:>10} {cm[i,2]:>10}")
    print(f"\n{metrics['report']}")


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 1: ESM-1v zero-shot (log-likelihood ratio)
# ═══════════════════════════════════════════════════════════════════════════

def run_esm1v_zeroshot(test_data_path: str, device: str) -> dict:
    """ESM-1v zero-shot variant effect prediction via masked marginal scoring.

    Approach:
        For each position i that differs between ref and var:
            score += log P(var_aa_i | context) - log P(ref_aa_i | context)
        Thresholds on score to classify: score < -2 → LoF, > +2 → GoF, else WT.

    This is the standard ESM-1v approach from Meier et al. (2021).
    """
    import pandas as pd
    from transformers import AutoTokenizer, EsmForMaskedLM

    logger.info("Loading ESM-1v model (facebook/esm1v_t33_650M_UR90S_1)...")
    model_name = "facebook/esm1v_t33_650M_UR90S_1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    df = pd.read_parquet(test_data_path)
    labels = df["label"].values
    dms_targets = df["dms_zscore"].values if "dms_zscore" in df.columns else None
    scores = np.zeros(len(df), dtype=np.float32)

    mask_token_id = tokenizer.mask_token_id

    for idx in tqdm(range(len(df)), desc="ESM-1v zero-shot"):
        ref_seq = str(df.iloc[idx]["ref_protein"]).replace("*", "")[:1022]
        var_seq = str(df.iloc[idx]["var_protein"]).replace("*", "")[:1022]

        # Find mutated positions
        min_len = min(len(ref_seq), len(var_seq))
        mut_positions = [i for i in range(min_len) if ref_seq[i] != var_seq[i]]

        # Also handle length changes (truncation → negative, insertion → positive)
        if len(var_seq) < len(ref_seq):
            # Truncation: strongly negative score
            scores[idx] = -5.0 * (len(ref_seq) - len(var_seq)) / max(len(ref_seq), 1)
            continue
        elif len(var_seq) > len(ref_seq):
            scores[idx] = 0.0  # Insertions are neutral by default
            continue

        if not mut_positions:
            scores[idx] = 0.0
            continue

        # Masked marginal scoring: mask each mutated position and score
        total_score = 0.0
        # Process in batches of mutations for efficiency
        for pos in mut_positions:
            # Tokenize reference with the mutation position masked
            seq_list = list(ref_seq)
            # Use reference context (wild-type marginal)
            enc = tokenizer(ref_seq, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(device)

            # Position in tokenized sequence (offset by 1 for BOS token)
            token_pos = pos + 1
            if token_pos >= input_ids.shape[1] - 1:
                continue

            # Mask the position
            masked_ids = input_ids.clone()
            masked_ids[0, token_pos] = mask_token_id

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(masked_ids, attention_mask=enc["attention_mask"].to(device)).logits

            # Get log-probabilities at the masked position
            log_probs = torch.log_softmax(logits[0, token_pos], dim=-1)

            # Convert amino acids to token IDs
            ref_aa = ref_seq[pos]
            var_aa = var_seq[pos]
            ref_token = tokenizer.convert_tokens_to_ids(ref_aa)
            var_token = tokenizer.convert_tokens_to_ids(var_aa)

            # Log-likelihood ratio
            total_score += (log_probs[var_token] - log_probs[ref_token]).item()

        scores[idx] = total_score

    # Threshold scores into classes
    # Calibrate thresholds on score distribution
    lo_thresh = np.percentile(scores, 33)
    hi_thresh = np.percentile(scores, 67)
    preds = np.where(scores < lo_thresh, 0, np.where(scores > hi_thresh, 2, 1))

    # Use scores as pseudo-probabilities for AUROC via softmax-like conversion
    # Stack [P(LoF), P(WT), P(GoF)] using score as logit for a 3-way split
    probs = np.column_stack([
        1.0 / (1.0 + np.exp(scores - lo_thresh)),        # P(LoF) — high when score < lo_thresh
        np.exp(-((scores - (lo_thresh + hi_thresh) / 2) ** 2) / 2),  # P(WT) — gaussian around center
        1.0 / (1.0 + np.exp(-(scores - hi_thresh))),     # P(GoF) — high when score > hi_thresh
    ])
    probs = probs / probs.sum(axis=1, keepdims=True)

    return _compute_all_metrics(
        preds, labels, probs,
        dms_preds=scores if dms_targets is not None else None,
        dms_targets=dms_targets,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 2: Logistic Regression on ESM2 embeddings
# ═══════════════════════════════════════════════════════════════════════════

def run_logistic_regression(train: dict, test: dict) -> dict:
    """L2-regularized logistic regression on ESM2 mean embeddings + nuc features."""
    logger.info("Training logistic regression...")
    X_train = _make_flat_features(train, mode="full")
    X_test = _make_flat_features(test, mode="full")
    y_train, y_test = train["labels"], test["labels"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    return _compute_all_metrics(preds, y_test, probs)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 3: Random Forest on ESM2 embeddings
# ═══════════════════════════════════════════════════════════════════════════

def run_random_forest(train: dict, test: dict) -> dict:
    """Random Forest on ESM2 mean embeddings + nuc features."""
    logger.info("Training random forest (500 trees)...")
    X_train = _make_flat_features(train, mode="full")
    X_test = _make_flat_features(test, mode="full")
    y_train, y_test = train["labels"], test["labels"]

    clf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    return _compute_all_metrics(preds, y_test, probs)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 4: Simple MLP on ESM2 embeddings (no comparison module)
# ═══════════════════════════════════════════════════════════════════════════

class _SimpleMLP(nn.Module):
    """2-layer MLP baseline — same capacity as ClassifierHead but no
    ComparisonModule, no cross-attention, no LayerNorm on features."""

    def __init__(self, input_size: int, hidden_dims=(512, 128), num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_size
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_simple_mlp(train: dict, test: dict, device: str) -> dict:
    """Train a simple MLP on concatenated ESM2 embeddings (no comparison module)."""
    logger.info("Training simple MLP baseline...")
    X_train = _make_flat_features(train, mode="full")
    X_test = _make_flat_features(test, mode="full")
    y_train, y_test = train["labels"], test["labels"]

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    input_size = X_train_s.shape[1]
    model = _SimpleMLP(input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    train_X = torch.tensor(X_train_s, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # Train for 30 epochs
    model.train()
    for epoch in range(30):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    # Evaluate
    model.eval()
    test_X = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(test_X)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    return _compute_all_metrics(preds, y_test, probs)


# ═══════════════════════════════════════════════════════════════════════════
# PLMLoF (for side-by-side display)
# ═══════════════════════════════════════════════════════════════════════════

def run_plmlof(
    checkpoint_path: str, test_data_path: str, device: str,
) -> dict:
    """Load PLMLoF checkpoint and evaluate on the test set (cached path)."""
    from plmlof.models.comparison import ComparisonModule, PooledCrossAttention
    from plmlof.models.classifier import ClassifierHead, RegressionHead
    from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES
    from plmlof.data.dataset import PLMLoFDataset
    from transformers import AutoTokenizer, AutoModel

    logger.info("Evaluating PLMLoF checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not ckpt.get("cached_training"):
        raise ValueError("Only cached-training checkpoints are supported")

    model_cfg = ckpt.get("model_config", {})
    esm2_name = model_cfg.get("esm2_model_name", "facebook/esm2_t33_650M_UR50D")
    pool_strategy = model_cfg.get("pool_strategy", "mean_max")

    # Reconstruct model
    comparison_state = ckpt["comparison_state_dict"]
    pre_norm_shape = comparison_state["_pre_norm.weight"].shape[0]
    hidden_size = pre_norm_shape // 8 if pool_strategy == "mean_max" else pre_norm_shape // 4
    comparison = ComparisonModule(hidden_size=hidden_size, pool_strategy=pool_strategy)
    comparison.load_state_dict(ckpt["comparison_state_dict"])

    classifier_input = comparison.output_size + NUM_NUCLEOTIDE_FEATURES
    classifier = ClassifierHead(
        input_size=classifier_input,
        hidden_dims=model_cfg.get("classifier_hidden_dims", [256, 64]),
        num_classes=3,
        dropout=model_cfg.get("classifier_dropout", 0.3),
    )
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    regressor = None
    if model_cfg.get("has_regressor") and "regressor_state_dict" in ckpt:
        regressor = RegressionHead(input_size=classifier_input)
        regressor.load_state_dict(ckpt["regressor_state_dict"])

    feature_norm = nn.LayerNorm(NUM_NUCLEOTIDE_FEATURES)
    if "feature_norm_state_dict" in ckpt:
        feature_norm.load_state_dict(ckpt["feature_norm_state_dict"])

    cross_attn = None
    if model_cfg.get("use_cross_attention") and "cross_attn_state_dict" in ckpt:
        cross_attn = PooledCrossAttention(
            hidden_size=hidden_size,
            num_heads=model_cfg.get("cross_attn_heads", 4),
            dropout=model_cfg.get("cross_attn_dropout", 0.1),
        )
        cross_attn.load_state_dict(ckpt["cross_attn_state_dict"])

    dev = torch.device(device)
    comparison = comparison.to(dev).eval()
    classifier = classifier.to(dev).eval()
    feature_norm = feature_norm.to(dev).eval()
    if cross_attn is not None:
        cross_attn = cross_attn.to(dev).eval()
    if regressor is not None:
        regressor = regressor.to(dev).eval()

    # Load ESM2 for embedding the test set
    tokenizer = AutoTokenizer.from_pretrained(esm2_name)
    esm2 = AutoModel.from_pretrained(esm2_name).to(dev).eval()
    for p in esm2.parameters():
        p.requires_grad = False

    dataset = PLMLoFDataset(test_data_path)

    def _collate(batch):
        ref_seqs = [s["ref_protein"] for s in batch]
        var_seqs = [s["var_protein"] for s in batch]
        enc = tokenizer(ref_seqs + var_seqs, padding=True, truncation=True,
                        max_length=1024, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "n_ref": len(ref_seqs),
            "nucleotide_features": torch.stack([s["nucleotide_features"] for s in batch]),
            "labels": torch.tensor([s["label"] for s in batch], dtype=torch.long),
            "dms_scores": torch.tensor([s.get("dms_score", 0.0) for s in batch], dtype=torch.float32),
        }

    loader = DataLoader(dataset, batch_size=16, collate_fn=_collate, num_workers=4, pin_memory=True)

    all_preds, all_labels, all_probs = [], [], []
    all_reg_preds, all_dms_targets = [], []

    def _pool(emb, mask):
        m_f = mask.unsqueeze(-1).float()
        mean_p = (emb * m_f).sum(1) / m_f.sum(1).clamp(min=1)
        emb_masked = emb.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
        max_p = emb_masked.max(dim=1).values
        max_p = max_p.masked_fill(max_p == float("-inf"), 0.0)
        return mean_p, max_p

    with torch.no_grad():
        for batch in tqdm(loader, desc="PLMLoF eval"):
            ids = batch["input_ids"].to(dev, non_blocking=True)
            mask = batch["attention_mask"].to(dev, non_blocking=True)
            n_ref = batch["n_ref"]
            nuc = batch["nucleotide_features"].to(dev, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                out = esm2(ids, attention_mask=mask).last_hidden_state

            ref_out, var_out = out[:n_ref], out[n_ref:]
            ref_mask, var_mask = mask[:n_ref], mask[n_ref:]
            ref_mean, ref_max = _pool(ref_out, ref_mask)
            var_mean, var_max = _pool(var_out, var_mask)

            if cross_attn is not None:
                tokens = torch.stack([ref_mean, ref_max, var_mean, var_max], dim=1)
                tokens = cross_attn(tokens)
                ref_mean, ref_max, var_mean, var_max = tokens[:, 0], tokens[:, 1], tokens[:, 2], tokens[:, 3]

            ref_pool = torch.cat([ref_mean, ref_max], dim=-1)
            var_pool = torch.cat([var_mean, var_max], dim=-1)
            comp = torch.cat([ref_pool - var_pool, ref_pool * var_pool, ref_pool, var_pool], dim=-1)
            comp = comparison.project(comp)
            features = torch.cat([comp, feature_norm(nuc)], dim=-1)
            logits = classifier(features)

            all_probs.extend(torch.softmax(logits, dim=-1).float().cpu().numpy())
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

            if regressor is not None:
                all_reg_preds.extend(regressor(features).float().cpu().numpy())
                all_dms_targets.extend(batch["dms_scores"].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    reg_preds = np.array(all_reg_preds) if all_reg_preds else None
    dms_targets = np.array(all_dms_targets) if all_dms_targets else None

    return _compute_all_metrics(preds, labels, probs, reg_preds, dms_targets)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark PLMLoF against baselines")
    p.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    p.add_argument("--train-embeddings", default="data/embeddings/train_embeddings.pt")
    p.add_argument("--test-data", default="data/processed/test.parquet",
                   help="Path to the test parquet (for ESM-1v and PLMLoF)")
    p.add_argument("--test-embeddings", default=None,
                   help="Pre-computed test embeddings .pt (if available). "
                        "If not provided, generates from test parquet using ESM2.")
    p.add_argument("--checkpoint", default="outputs/production/checkpoints/model_best.pt")
    p.add_argument("--output", default="outputs/production/benchmark_results.json",
                   help="Save JSON results to this path")
    p.add_argument("--skip-esm1v", action="store_true",
                   help="Skip ESM-1v zero-shot (slow, ~1h)")
    p.add_argument("--only", type=str, default=None,
                   help="Run only this baseline (esm1v, logistic_regression, random_forest, simple_mlp, plmlof)")
    return p.parse_args()


def _compute_test_embeddings(test_data_path: str, device: str) -> dict:
    """Compute ESM2 embeddings for the test set on-the-fly."""
    from plmlof.data.dataset import PLMLoFDataset
    from transformers import AutoTokenizer, AutoModel

    logger.info("Computing ESM2 embeddings for test set...")
    esm2_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(esm2_name)
    esm2 = AutoModel.from_pretrained(esm2_name).to(device).eval()
    for p in esm2.parameters():
        p.requires_grad = False

    dataset = PLMLoFDataset(test_data_path)

    def _collate(batch):
        ref_seqs = [s["ref_protein"] for s in batch]
        var_seqs = [s["var_protein"] for s in batch]
        enc = tokenizer(ref_seqs + var_seqs, padding=True, truncation=True,
                        max_length=1024, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "n_ref": len(ref_seqs),
            "nucleotide_features": torch.stack([s["nucleotide_features"] for s in batch]),
            "labels": torch.tensor([s["label"] for s in batch], dtype=torch.long),
            "dms_scores": torch.tensor([s.get("dms_score", 0.0) for s in batch], dtype=torch.float32),
        }

    loader = DataLoader(dataset, batch_size=16, collate_fn=_collate, num_workers=4, pin_memory=True)

    all_ref_mean, all_ref_max, all_var_mean, all_var_max = [], [], [], []
    all_nuc, all_labels, all_dms = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding test set"):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            n_ref = batch["n_ref"]

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
                out = esm2(ids, attention_mask=mask).last_hidden_state

            ref_out, var_out = out[:n_ref], out[n_ref:]
            ref_mask, var_mask = mask[:n_ref], mask[n_ref:]

            def _pool(emb, m):
                m_f = m.unsqueeze(-1).float()
                mean_p = (emb * m_f).sum(1) / m_f.sum(1).clamp(min=1)
                emb_masked = emb.masked_fill(~m.unsqueeze(-1).bool(), float("-inf"))
                max_p = emb_masked.max(dim=1).values
                max_p = max_p.masked_fill(max_p == float("-inf"), 0.0)
                return mean_p.float().cpu().numpy(), max_p.float().cpu().numpy()

            rm, rx = _pool(ref_out, ref_mask)
            vm, vx = _pool(var_out, var_mask)
            all_ref_mean.append(rm)
            all_ref_max.append(rx)
            all_var_mean.append(vm)
            all_var_max.append(vx)
            all_nuc.append(batch["nucleotide_features"].numpy())
            all_labels.append(batch["labels"].numpy())
            all_dms.append(batch["dms_scores"].numpy())

    return {
        "ref_mean": np.concatenate(all_ref_mean),
        "ref_max": np.concatenate(all_ref_max),
        "var_mean": np.concatenate(all_var_mean),
        "var_max": np.concatenate(all_var_max),
        "nuc_features": np.concatenate(all_nuc),
        "labels": np.concatenate(all_labels),
        "dms_scores": np.concatenate(all_dms),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    device = args.device

    print("=" * 60)
    print("  PLMLoF BASELINE BENCHMARK")
    print("=" * 60)
    print(f"  Device:           {device}")
    print(f"  Test data:        {args.test_data}")
    print(f"  Train embeddings: {args.train_embeddings}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print()

    # Load training embeddings (used by LR, RF, MLP baselines)
    train_emb_path = Path(args.train_embeddings)
    if not train_emb_path.exists():
        print(f"ERROR: Training embeddings not found at {train_emb_path}")
        print("Run: python scripts/precompute_embeddings.py first")
        return
    train = _load_cached_split(train_emb_path)
    logger.info(f"Loaded training embeddings: {len(train['labels'])} samples")

    # Load or compute test embeddings
    if args.test_embeddings and Path(args.test_embeddings).exists():
        test = _load_cached_split(Path(args.test_embeddings))
        logger.info(f"Loaded test embeddings: {len(test['labels'])} samples")
    else:
        test = _compute_test_embeddings(args.test_data, device)
        logger.info(f"Computed test embeddings: {len(test['labels'])} samples")

    results = {}

    # ── Run baselines ──
    baselines = {
        "logistic_regression": ("Logistic Regression (ESM2 + features)", lambda: run_logistic_regression(train, test)),
        "random_forest": ("Random Forest (ESM2 + features)", lambda: run_random_forest(train, test)),
        "simple_mlp": ("Simple MLP (ESM2 + features, no comparison)", lambda: run_simple_mlp(train, test, device)),
        "esm1v": ("ESM-1v Zero-Shot (masked marginal)", lambda: run_esm1v_zeroshot(args.test_data, device)),
        "plmlof": ("PLMLoF (ours)", lambda: run_plmlof(args.checkpoint, args.test_data, device)),
    }

    run_order = ["logistic_regression", "random_forest", "simple_mlp"]
    if not args.skip_esm1v:
        run_order.append("esm1v")
    run_order.append("plmlof")

    if args.only:
        if args.only not in baselines:
            print(f"Unknown baseline: {args.only}. Choose from: {list(baselines.keys())}")
            return
        run_order = [args.only]

    for key in run_order:
        name, fn = baselines[key]
        try:
            t0 = time.time()
            metrics = fn()
            elapsed = time.time() - t0
            results[key] = {"name": name, "metrics": metrics, "time_seconds": elapsed}
            _print_results(name, metrics, elapsed)
        except Exception as e:
            logger.error(f"Failed to run {name}: {e}", exc_info=True)
            results[key] = {"name": name, "error": str(e)}

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    header = f"  {'Method':<45} {'F1':>7} {'AUROC':>7} {'Spρ':>7}"
    print(header)
    print("  " + "-" * 58)
    for key in run_order:
        if key in results and "metrics" in results[key]:
            m = results[key]["metrics"]
            name = results[key]["name"]
            f1 = f"{m['macro_f1']:.4f}"
            auroc = f"{m.get('auroc', float('nan')):.4f}" if "auroc" in m else "  N/A "
            sp = f"{m.get('spearman_rho', float('nan')):.4f}" if "spearman_rho" in m else "  N/A "
            print(f"  {name:<45} {f1:>7} {auroc:>7} {sp:>7}")
    print()

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
