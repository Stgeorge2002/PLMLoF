"""Shared fixtures for PLMLoF test suite.

All tests use the tiny ESM2 model (esm2_t6_8M_UR50D, ~32MB) so they
run on CPU without needing a GPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

TINY_MODEL = "facebook/esm2_t6_8M_UR50D"


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_model_name() -> str:
    """Return the tiny ESM2 model name for CPU testing."""
    return TINY_MODEL


@pytest.fixture(scope="session")
def lora_config() -> dict:
    """Minimal LoRA config for testing."""
    return {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["query", "value"],
    }


@pytest.fixture(scope="session")
def plmlof_model(tiny_model_name, lora_config):
    """Build a tiny PLMLoFModel for testing (session-scoped, reused)."""
    from plmlof.models.plmlof_model import PLMLoFModel

    model = PLMLoFModel(
        esm2_model_name=tiny_model_name,
        freeze_esm2=True,
        lora_config=lora_config,
        pool_strategy="mean_max",
        classifier_hidden_dims=[32, 16],
        classifier_dropout=0.0,
        num_classes=3,
        num_nuc_features=12,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_dataset():
    """A small synthetic dataset (no file I/O, no downloads)."""
    from plmlof.data.dataset import SyntheticPLMLoFDataset

    return SyntheticPLMLoFDataset(num_samples=9, seed=42)


@pytest.fixture(scope="session")
def collator(tiny_model_name):
    """PLMLoFCollator backed by the tiny ESM2 tokenizer."""
    from plmlof.data.collator import PLMLoFCollator

    return PLMLoFCollator(tokenizer_name=tiny_model_name, max_length=64)


@pytest.fixture()
def csv_dataset_path(tmp_path: Path) -> Path:
    """Write a small CSV dataset and return its path."""
    import pandas as pd

    ref = "MKTLLLTLVVVTLAALG"
    rows = [
        {"ref_protein": ref, "var_protein": ref[:5] + "*" + ref[6:], "label": 0,
         "ref_dna": "", "var_dna": "", "gene": "g1", "species": "sp1", "dms_zscore": -1.5},
        {"ref_protein": ref, "var_protein": ref, "label": 1,
         "ref_dna": "", "var_dna": "", "gene": "g2", "species": "sp1", "dms_zscore": 0.0},
        {"ref_protein": ref, "var_protein": "M" + "R" + ref[2:], "label": 2,
         "ref_dna": "", "var_dna": "", "gene": "g3", "species": "sp1", "dms_zscore": 1.8},
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Sequence fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_dna_ref() -> str:
    """Short reference DNA sequence encoding ~6 amino acids."""
    return "ATGAAATTTGGGCCC"  # MKF GP (5 codons → 5 AAs)


@pytest.fixture()
def sample_protein_ref() -> str:
    return "MKTLLLTLVVVTLAALG"


@pytest.fixture()
def sample_protein_var_lof() -> str:
    """Variant with premature stop → LoF."""
    return "MKTLL*TLVVVTLAALG"


@pytest.fixture()
def sample_protein_var_gof() -> str:
    """Variant with missense at position 2 → GoF-like."""
    return "MRTLLLTLVVVTLAALG"


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    return out
