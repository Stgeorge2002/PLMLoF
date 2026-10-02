"""Parquet + cached-embedding datasets for task heads."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from plmlof.data.features import extract_nucleotide_features


REQUIRED_COLUMNS = {"ref_protein", "var_protein", "target", "sample_weight"}
IDENTITY_TARGET_EPS = 0.05


def lof_train_keep_indices(
    is_wreck: torch.Tensor,
    is_missense: torch.Tensor,
    targets: torch.Tensor,
    *,
    channels: list[str] | None = None,
    seed: int = 0,
    identity_per_missense: float = 1.0,
) -> list[int]:
    """LoF train rows: drop sure wrecks; cap identity WT to ~1× missense count.

    Inference still applies wreck_grade. Training on sure wrecks teaches length.
    Identity WT is `channel == wt` and not missense when channels exist; otherwise
    non-wreck / non-missense with target ≈ 0 (works on older embedding caches).
    """
    wreck = is_wreck.bool()
    miss = is_missense.bool()
    keep = ~wreck
    have_wt = channels is not None and len(channels) == int(keep.numel()) and any(c == "wt" for c in channels)
    if have_wt:
        ident = keep & ~miss & torch.tensor([c == "wt" for c in channels], dtype=torch.bool)
    else:
        ident = keep & ~miss & (targets.abs() < IDENTITY_TARGET_EPS)
    n_miss = int((keep & miss).sum().item())
    cap = max(int(round(n_miss * identity_per_missense)), 1)
    ident_idx = ident.nonzero(as_tuple=False).view(-1)
    other = (keep & ~ident).nonzero(as_tuple=False).view(-1)
    if ident_idx.numel() > cap:
        g = torch.Generator()
        g.manual_seed(int(seed))
        ident_idx = ident_idx[torch.randperm(ident_idx.numel(), generator=g)[:cap]]
    return torch.cat([other, ident_idx]).sort().values.tolist()


class PairDataset(Dataset):
    """On-disk rows (used for embedding scatter and live ESM2)."""

    def __init__(self, data_path: str | Path, max_seq_length: int = 1024):
        self.max_seq_length = max_seq_length
        path = Path(data_path)
        if path.suffix == ".parquet":
            self.df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            self.df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

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


class CachedDataset(Dataset):
    """Pre-pooled ESM2 embeddings plus targets/weights/channels."""

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
        self.channels = data.get("channels", ["unknown"] * len(self.targets))

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
            "gene": self.genes[idx],
            "protein_id": self.protein_ids[idx],
            "channel": self.channels[idx],
        }
