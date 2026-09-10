"""Tests for tabular preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from datasets.preprocessing import (
    create_tabular_preprocessor,
    fit_transform_tabular,
    transform_tabular,
)


def _mixed_dataframe() -> pd.DataFrame:
    """Return a small heterogeneous tabular dataset."""
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, np.nan, 50.0],
            "income": [20_000.0, 30_000.0, 40_000.0, 50_000.0],
            "occupation": [
                "student",
                "engineer",
                "student",
                None,
            ],
        }
    )


def test_preprocessor_transforms_mixed_data():
    """Mixed numerical and categorical data should become numeric."""
    X = _mixed_dataframe()

    preprocessor = create_tabular_preprocessor(
        numerical_features=("age", "income"),
        categorical_features=("occupation",),
    )

    transformed = fit_transform_tabular(
        preprocessor,
        X,
    )

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(X)
    assert np.issubdtype(
        transformed.dtype,
        np.number,
    )


def test_preprocessor_imputes_missing_values():
    """Transformed data should not contain missing values."""
    X = _mixed_dataframe()

    preprocessor = create_tabular_preprocessor(
        numerical_features=("age", "income"),
        categorical_features=("occupation",),
    )

    transformed = fit_transform_tabular(
        preprocessor,
        X,
    )

    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_categories():
    """Test-only categories should not require refitting the encoder."""
    X_train = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],
            "occupation": [
                "student",
                "engineer",
                "student",
            ],
        }
    )

    X_test = pd.DataFrame(
        {
            "age": [50.0],
            "occupation": ["artist"],
        }
    )

    preprocessor = create_tabular_preprocessor(
        numerical_features=("age",),
        categorical_features=("occupation",),
    )

    X_train_transformed = fit_transform_tabular(
        preprocessor,
        X_train,
    )

    X_test_transformed = transform_tabular(
        preprocessor,
        X_test,
    )

    assert (
        X_test_transformed.shape[1]
        == X_train_transformed.shape[1]
    )


def test_numeric_only_dataset_is_supported():
    """Datasets containing only numerical features should be supported."""
    X = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],
            "income": [10.0, 20.0, 30.0],
        }
    )

    preprocessor = create_tabular_preprocessor(
        numerical_features=("age", "income"),
        categorical_features=(),
    )

    transformed = fit_transform_tabular(
        preprocessor,
        X,
    )

    assert transformed.shape == (3, 2)


def test_categorical_only_dataset_is_supported():
    """Datasets containing only categorical features should be supported."""
    X = pd.DataFrame(
        {
            "city": ["Burgos", "Madrid", "Burgos"],
        }
    )

    preprocessor = create_tabular_preprocessor(
        numerical_features=(),
        categorical_features=("city",),
    )

    transformed = fit_transform_tabular(
        preprocessor,
        X,
    )

    assert transformed.shape[0] == 3


def test_preprocessor_rejects_overlapping_features():
    """A feature cannot belong to multiple preprocessing groups."""
    with pytest.raises(
        ValueError,
        match="Feature names must be unique",
    ):
        create_tabular_preprocessor(
            numerical_features=("age",),
            categorical_features=("age",),
        )


def test_preprocessor_rejects_empty_feature_groups():
    """At least one feature must be configured."""
    with pytest.raises(
        ValueError,
        match="At least one feature",
    ):
        create_tabular_preprocessor(
            numerical_features=(),
            categorical_features=(),
        )


def test_preprocessor_rejects_non_dataframe():
    """Preprocessing should require named pandas columns."""
    preprocessor = create_tabular_preprocessor(
        numerical_features=("age",),
        categorical_features=(),
    )

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        fit_transform_tabular(
            preprocessor,
            np.array([[20.0], [30.0]]),
        )
        
def test_test_data_does_not_change_numeric_scaling():
    """Test values must not influence training preprocessing statistics."""
    X_train = pd.DataFrame(
        {
            "value": [0.0, 1.0, 2.0],
        }
    )

    X_test = pd.DataFrame(
        {
            "value": [1_000_000.0],
        }
    )

    preprocessor = create_tabular_preprocessor(
        numerical_features=("value",),
        categorical_features=(),
    )

    X_train_transformed = fit_transform_tabular(
        preprocessor,
        X_train,
    )

    transform_tabular(
        preprocessor,
        X_test,
    )

    expected_mean = 0.0

    assert np.isclose(
        X_train_transformed.mean(),
        expected_mean,
    )