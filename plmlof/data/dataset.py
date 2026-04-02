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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        ref_protein = str(row["ref_protein"]).replace("*", "")[: self.max_seq_length]
        var_protein = str(row["var_protein"]).replace("*", "")[: self.max_seq_length]
        label = int(row["label"])

        # Compute nucleotide features if DNA columns available
        ref_dna = str(row.get("ref_dna", ""))
        var_dna = str(row.get("var_dna", ""))

        if ref_dna and var_dna and ref_dna != "nan" and var_dna != "nan":
            nuc_features = extract_nucleotide_features(
                ref_dna, var_dna, ref_protein, var_protein
            )
        else:
            # Fallback: compute from protein-level differences only
            nuc_features = extract_nucleotide_features(
                "", "", ref_protein, var_protein
            )

        return {
            "ref_protein": ref_protein,
            "var_protein": var_protein,
            "nucleotide_features": nuc_features,
            "label": label,
            "gene": str(row.get("gene", "")),
            "species": str(row.get("species", "")),
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
            if label == 0:  # LoF — premature stop
                var = ref_base[:5] + "*" + ref_base[6:]
            elif label == 2:  # GoF — missense in key position
                var = "M" + "R" + ref_base[2:]
            else:  # WT — identical or synonymous
                var = ref_base

            self.samples.append({
                "ref_protein": ref_base,
                "var_protein": var,
                "nucleotide_features": torch.randn(12, generator=rng),
                "label": label,
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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "ref_mean": self.ref_mean[idx],
            "ref_max": self.ref_max[idx],
            "var_mean": self.var_mean[idx],
            "var_max": self.var_max[idx],
            "nucleotide_features": self.nuc_features[idx],
            "label": self.labels[idx].item(),
        }
