"""Shared fixtures for PLMLoF tests. Heads themselves do not load ESM2."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def sample_protein_ref() -> str:
    return "MKTLLLTLVVVTLAALG"


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    return out
