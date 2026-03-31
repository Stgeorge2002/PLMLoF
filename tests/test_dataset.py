"""Tests for datasets and collator."""

from __future__ import annotations

import torch
import pytest

from plmlof.data.dataset import PLMLoFDataset, SyntheticPLMLoFDataset
from plmlof.data.features import NUM_NUCLEOTIDE_FEATURES


class TestSyntheticDataset:
    def test_length(self, synthetic_dataset):
        assert len(synthetic_dataset) == 9

    def test_sample_keys(self, synthetic_dataset):
        sample = synthetic_dataset[0]
        expected_keys = {"ref_protein", "var_protein", "nucleotide_features", "label", "gene", "species"}
        assert expected_keys == set(sample.keys())

    def test_feature_shape(self, synthetic_dataset):
        sample = synthetic_dataset[0]
        assert sample["nucleotide_features"].shape == (NUM_NUCLEOTIDE_FEATURES,)

    def test_label_range(self, synthetic_dataset):
        for i in range(len(synthetic_dataset)):
            assert synthetic_dataset[i]["label"] in (0, 1, 2)

    def test_balanced_labels(self):
        ds = SyntheticPLMLoFDataset(num_samples=9, seed=0)
        labels = [ds[i]["label"] for i in range(len(ds))]
        assert labels.count(0) == 3
        assert labels.count(1) == 3
        assert labels.count(2) == 3


class TestPLMLoFDataset:
    def test_load_csv(self, csv_dataset_path):
        ds = PLMLoFDataset(csv_dataset_path)
        assert len(ds) == 3

    def test_sample_schema(self, csv_dataset_path):
        ds = PLMLoFDataset(csv_dataset_path)
        s = ds[0]
        assert isinstance(s["ref_protein"], str)
        assert isinstance(s["var_protein"], str)
        assert isinstance(s["label"], int)
        assert s["nucleotide_features"].shape == (NUM_NUCLEOTIDE_FEATURES,)

    def test_missing_column_raises(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"ref_protein": ["MKTL"], "label": [0]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            PLMLoFDataset(path)

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported format"):
            PLMLoFDataset(path)


class TestCollator:
    def test_collated_keys(self, collator, synthetic_dataset):
        batch = collator([synthetic_dataset[0], synthetic_dataset[1]])
        required = {
            "ref_input_ids", "ref_attention_mask",
            "var_input_ids", "var_attention_mask",
            "nucleotide_features", "labels",
        }
        assert required == set(batch.keys())

    def test_collated_shapes(self, collator, synthetic_dataset):
        batch = collator([synthetic_dataset[i] for i in range(3)])
        assert batch["ref_input_ids"].shape[0] == 3
        assert batch["var_input_ids"].shape[0] == 3
        assert batch["nucleotide_features"].shape == (3, NUM_NUCLEOTIDE_FEATURES)
        assert batch["labels"].shape == (3,)

    def test_labels_dtype(self, collator, synthetic_dataset):
        batch = collator([synthetic_dataset[0]])
        assert batch["labels"].dtype == torch.long
