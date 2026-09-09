"""Tests for the core tabular dataset representation."""

import numpy as np
import pandas as pd

from datasets.dataset import TabularDataset


def _create_example_dataset() -> TabularDataset:
    """Create a small dataset used by the unit tests."""
    X = pd.DataFrame(
        {
            "age": [25, 40, 31, 52],
            "income": [30000.0, 52000.0, 41000.0, 68000.0],
            "occupation": ["A", "B", "A", "C"],
        }
    )

    y = pd.Series(
        [0, 1, 0, 1],
        name="target",
    )

    return TabularDataset(
        name="example",
        X=X,
        y=y,
        target_name="target",
        source="test",
        source_id=1,
        source_version=1,
        categorical_features=("occupation",),
        numerical_features=("age", "income"),
    )


def test_dataset_reports_number_of_samples():
    """The sample count should be derived from the feature matrix."""
    dataset = _create_example_dataset()

    assert dataset.n_samples == 4


def test_dataset_reports_number_of_features():
    """The feature count should be derived from the feature matrix."""
    dataset = _create_example_dataset()

    assert dataset.n_features == 3


def test_dataset_reports_classes():
    """Dataset classes should be derived from the target."""
    dataset = _create_example_dataset()

    np.testing.assert_array_equal(
        dataset.classes,
        np.array([0, 1]),
    )


def test_dataset_reports_number_of_classes():
    """The class count should be derived from the target."""
    dataset = _create_example_dataset()

    assert dataset.n_classes == 2


def test_dataset_preserves_feature_types():
    """Categorical and numerical feature metadata should be preserved."""
    dataset = _create_example_dataset()

    assert dataset.categorical_features == ("occupation",)
    assert dataset.numerical_features == ("age", "income")


def test_dataset_preserves_source_information():
    """Dataset source identifiers should remain available."""
    dataset = _create_example_dataset()

    assert dataset.source == "test"
    assert dataset.source_id == 1
    assert dataset.source_version == 1