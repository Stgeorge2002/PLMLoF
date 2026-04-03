"""PyTorch Dataset for PLMLoF reference-variant sequence pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from plmlof.data.features import extract_nucleotide_features
from plmlof.utils.sequence_utils import translate_dna


class PLMLoFDataset(Dataset):
    """Dataset of (reference, variant) protein sequence pairs with labels.

    Each sample contains:
        - ref_protein: reference protein sequence (str)
        - var_protein: variant protein sequence (str)
        - nucleotide_features: engineered features (Tensor[12])
        - label: 0=LoF, 1=WT, 2=GoF (int)
        - gene: gene name (str)
        - species: species name (str)
    """

    def __init__(
        self,
        data_path: str | Path,
        max_seq_length: int = 1024,
    ):
        """
        Args:
            data_path: Path to parquet/csv file with columns:
                ref_protein, var_protein, ref_dna, var_dna, label, gene, species
            max_seq_length: Maximum protein sequence length (will be truncated).
        """
        self.max_seq_length = max_seq_length

        path = Path(data_path)
        if path.suffix == ".parquet":
            self.df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            self.df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}. Use .parquet or .csv")

        required_cols = {"ref_protein", "var_protein", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # --- Pre-extract columns as lists (avoid pd.Series alloc per __getitem__) ---
        self._ref_proteins_raw = [
            str(v)[:max_seq_length] for v in self.df["ref_protein"]
        ]
        self._var_proteins_raw = [
            str(v)[:max_seq_length] for v in self.df["var_protein"]
        ]
        self._labels = self.df["label"].astype(int).tolist()

        has_dna_col = "ref_dna" in self.df.columns and "var_dna" in self.df.columns
        self._ref_dnas: list[str] = []
        self._var_dnas: list[str] = []
        if has_dna_col:
            for rd, vd in zip(self.df["ref_dna"], self.df["var_dna"]):
                rd_s = "" if pd.isna(rd) else str(rd)
                vd_s = "" if pd.isna(vd) else str(vd)
                self._ref_dnas.append(rd_s)
                self._var_dnas.append(vd_s)
        else:
            self._ref_dnas = [""] * len(self.df)
            self._var_dnas = [""] * len(self.df)

        has_dms = "dms_zscore" in self.df.columns
        self._dms_scores: list[float] = []
        if has_dms:
            self._dms_scores = [
                float(v) if pd.notna(v) else 0.0 for v in self.df["dms_zscore"]
            ]
        else:
            self._dms_scores = [0.0] * len(self.df)

        self._genes = [str(v) for v in self.df.get("gene", [""] * len(self.df))]
        self._species = [str(v) for v in self.df.get("species", [""] * len(self.df))]

        # --- Pre-compute nucleotide features once (deterministic, no need to redo per epoch) ---
        self._nuc_features = self._precompute_nucleotide_features()

    def _precompute_nucleotide_features(self) -> torch.Tensor:
        """Compute all nucleotide features upfront. Called once at init."""
        features_list = []
        for i in range(len(self.df)):
            ref_dna = self._ref_dnas[i]
            var_dna = self._var_dnas[i]
            ref_prot = self._ref_proteins_raw[i]
            var_prot = self._var_proteins_raw[i]

            if ref_dna and var_dna:
                feat = extract_nucleotide_features(ref_dna, var_dna, ref_prot, var_prot)
            else:
                feat = extract_nucleotide_features("", "", ref_prot, var_prot)
            features_list.append(feat)
        return torch.stack(features_list)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        # Strip "*" for ESM2 tokenization (ESM2 cannot tokenize stop codons)
        ref_protein = self._ref_proteins_raw[idx].replace("*", "")
        var_protein = self._var_proteins_raw[idx].replace("*", "")

        return {
            "ref_protein": ref_protein,
            "var_protein": var_protein,
            "nucleotide_features": self._nuc_features[idx],
            "label": self._labels[idx],
            "dms_score": self._dms_scores[idx],
            "gene": self._genes[idx],
            "species": self._species[idx],
        }


class SyntheticPLMLoFDataset(Dataset):
    """Small synthetic dataset for testing. No file I/O required."""

    def __init__(self, num_samples: int = 20, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)

        self.samples = []
        # Generate balanced classes
        labels = [0] * (num_samples // 3) + [1] * (num_samples // 3) + [2] * (num_samples - 2 * (num_samples // 3))

        # Simple test proteins
        ref_base = "MKTLLLTLVVVTLAALG"
        for i, label in enumerate(labels):
            if label == 0:  # LoF — truncation (premature stop removed for ESM2)
                var_raw = ref_base[:5] + "*" + ref_base[6:]
            elif label == 2:  # GoF — missense in key position
                var_raw = "M" + "R" + ref_base[2:]
            else:  # WT — identical or synonymous
                var_raw = ref_base

            # Compute real features before stripping "*"
            nuc_features = extract_nucleotide_features("", "", ref_base, var_raw)
            # Strip "*" for ESM2 tokenization
            var_clean = var_raw.replace("*", "")

            self.samples.append({
                "ref_protein": ref_base,
                "var_protein": var_clean,
                "nucleotide_features": nuc_features,
                "label": label,
                "dms_score": 0.0,
                "gene": f"test_gene_{i}",
                "species": "test_species",
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class CachedEmbeddingDataset(Dataset):
    """Dataset using pre-computed ESM2 embeddings (no ESM2 forward passes).

    Loads a .pt file created by scripts/precompute_embeddings.py containing
    pre-pooled ref/var mean+max embeddings, nucleotide features, and labels.
    """

    def __init__(self, cache_path: str | Path):
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Cached embeddings not found: {path}. "
                "Run scripts/precompute_embeddings.py first."
            )
        data = torch.load(path, weights_only=True)
        self.ref_mean = data["ref_mean"]
        self.ref_max = data["ref_max"]
        self.var_mean = data["var_mean"]
        self.var_max = data["var_max"]
        self.nuc_features = data["nucleotide_features"]
        self.labels = data["labels"]
        self.dms_scores = data.get("dms_scores", torch.zeros(len(self.labels)))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "ref_mean": self.ref_mean[idx],
            "ref_max": self.ref_max[idx],
            "var_mean": self.var_mean[idx],
            "var_max": self.var_max[idx],
            "nucleotide_features": self.nuc_features[idx],
            "labels": self.labels[idx],
            "dms_scores": self.dms_scores[idx],
        }
