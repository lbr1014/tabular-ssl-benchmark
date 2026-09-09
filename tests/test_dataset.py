"""Tests for the core tabular dataset representation."""

import numpy as np
import pandas as pd
import pytest

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
    
def test_dataset_rejects_mismatched_sample_counts():
    """Features and target must contain the same number of samples."""
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )
    y = pd.Series([0, 1], name="target")

    with pytest.raises(
        ValueError,
        match="X and y must contain the same number of samples",
    ):
        TabularDataset(
            name="invalid",
            X=X,
            y=y,
            target_name="target",
            source="test",
            numerical_features=("feature",),
        )


def test_dataset_rejects_unknown_feature_metadata():
    """Declared feature names must exist in the feature matrix."""
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="target")

    with pytest.raises(
        ValueError,
        match="Feature type metadata contains unknown columns",
    ):
        TabularDataset(
            name="invalid",
            X=X,
            y=y,
            target_name="target",
            source="test",
            numerical_features=("age", "income"),
        )


def test_dataset_rejects_overlapping_feature_types():
    """A feature cannot be both categorical and numerical."""
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="target")

    with pytest.raises(
        ValueError,
        match="Features cannot be both categorical and numerical",
    ):
        TabularDataset(
            name="invalid",
            X=X,
            y=y,
            target_name="target",
            source="test",
            categorical_features=("age",),
            numerical_features=("age",),
        )


def test_dataset_rejects_unclassified_features():
    """Every feature must have a categorical or numerical type."""
    X = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "occupation": ["A", "B", "A", "C"],
        }
    )
    y = pd.Series([0, 1, 0, 1], name="target")

    with pytest.raises(
        ValueError,
        match="Every feature must be classified",
    ):
        TabularDataset(
            name="invalid",
            X=X,
            y=y,
            target_name="target",
            source="test",
            numerical_features=("age",),
        )


def test_dataset_rejects_single_class_target():
    """Classification datasets must contain at least two classes."""
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )
    y = pd.Series([0, 0, 0, 0], name="target")

    with pytest.raises(
        ValueError,
        match="y must contain at least two classes",
    ):
        TabularDataset(
            name="invalid",
            X=X,
            y=y,
            target_name="target",
            source="test",
            numerical_features=("feature",),
        )
        
def test_dataset_rejects_empty_samples():
    """Datasets without samples should be rejected."""
    X = pd.DataFrame({"feature": pd.Series(dtype="float64")})
    y = pd.Series(dtype="int64", name="target")

    with pytest.raises(
        ValueError,
        match="X must contain at least one sample",
    ):
        TabularDataset(
            name="empty",
            X=X,
            y=y,
            target_name="target",
            source="test",
            numerical_features=("feature",),
        )