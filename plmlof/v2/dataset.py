"""Parquet + cached-embedding datasets for v2 tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from plmlof.data.features import extract_nucleotide_features


V2_REQUIRED = {"ref_protein", "var_protein", "target", "sample_weight"}


class V2PairDataset(Dataset):
    """On-disk v2 rows (used for embedding scatter and live ESM2)."""

    def __init__(self, data_path: str | Path, max_seq_length: int = 1024):
        self.max_seq_length = max_seq_length
        path = Path(data_path)
        if path.suffix == ".parquet":
            self.df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            self.df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        missing = V2_REQUIRED - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing v2 columns: {missing}")

        self._ref = [str(v).replace("*", "")[:max_seq_length] for v in self.df["ref_protein"]]
        self._var = [str(v).replace("*", "")[:max_seq_length] for v in self.df["var_protein"]]
        self._target = [float(v) for v in self.df["target"]]
        self._weight = [float(v) for v in self.df["sample_weight"]]
        self._genes = [str(v) for v in self.df.get("gene", [""] * len(self.df))]
        self._species = [str(v) for v in self.df.get("species", [""] * len(self.df))]
        self._channel = [str(v) for v in self.df.get("channel", ["unknown"] * len(self.df))]
        self._protein_id = [str(v) for v in self.df.get("protein_id", [""] * len(self.df))]
        z = self.df["dms_zscore"] if "dms_zscore" in self.df.columns else None
        if z is not None:
            self._z = [float(v) if pd.notna(v) else float("nan") for v in z]
        else:
            self._z = [float("nan")] * len(self.df)
        wreck = self.df["is_wreck"] if "is_wreck" in self.df.columns else None
        if wreck is not None:
            self._is_wreck = [bool(v) for v in wreck]
        else:
            self._is_wreck = ["wreck" in c or c == "clear_wreck" for c in self._channel]
        miss = self.df["is_missense"] if "is_missense" in self.df.columns else None
        if miss is not None:
            self._is_missense = [bool(v) for v in miss]
        else:
            self._is_missense = ["missense" in c for c in self._channel]

        self._nuc = torch.stack([
            extract_nucleotide_features(r, v) for r, v in zip(self._ref, self._var)
        ])

    def __len__(self) -> int:
        return len(self._ref)

    def __getitem__(self, idx: int) -> dict:
        return {
            "ref_protein": self._ref[idx],
            "var_protein": self._var[idx],
            "nucleotide_features": self._nuc[idx],
            "target": self._target[idx],
            "sample_weight": self._weight[idx],
            "dms_zscore": self._z[idx],
            "is_wreck": self._is_wreck[idx],
            "is_missense": self._is_missense[idx],
            "channel": self._channel[idx],
            "gene": self._genes[idx],
            "species": self._species[idx],
            "protein_id": self._protein_id[idx],
        }


class V2CachedDataset(Dataset):
    """Pre-pooled ESM2 embeddings plus v2 targets/weights/channels."""

    def __init__(self, cache_path: str | Path):
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"Cached embeddings not found: {path}")
        data = torch.load(path, weights_only=False)
        self.ref_mean = data["ref_mean"]
        self.ref_max = data["ref_max"]
        self.var_mean = data["var_mean"]
        self.var_max = data["var_max"]
        self.nuc_features = data["nucleotide_features"]
        self.targets = data["targets"].float()
        self.weights = data["weights"].float()
        self.z = data.get("dms_zscores", torch.full((len(self.targets),), float("nan")))
        self.is_wreck = data.get("is_wreck", torch.zeros(len(self.targets), dtype=torch.bool))
        self.is_missense = data.get("is_missense", torch.zeros(len(self.targets), dtype=torch.bool))
        self.genes = data.get("genes", [""] * len(self.targets))
        self.protein_ids = data.get("protein_ids", [""] * len(self.targets))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict:
        return {
            "ref_mean": self.ref_mean[idx],
            "ref_max": self.ref_max[idx],
            "var_mean": self.var_mean[idx],
            "var_max": self.var_max[idx],
            "nucleotide_features": self.nuc_features[idx],
            "target": self.targets[idx],
            "sample_weight": self.weights[idx],
            "dms_zscore": self.z[idx],
            "is_wreck": self.is_wreck[idx],
            "is_missense": self.is_missense[idx],
        }
