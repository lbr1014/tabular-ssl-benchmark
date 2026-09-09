"""Tests for the OpenML dataset loader."""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from datasets.openml_loader import load_openml_dataset


def _create_openml_response() -> SimpleNamespace:
    """Create a representative mocked OpenML response."""
    X = pd.DataFrame(
        {
            "age": pd.Series([25, 40, 31, 52], dtype="int64"),
            "income": pd.Series(
                [30000.0, 52000.0, 41000.0, 68000.0],
                dtype="float64",
            ),
            "occupation": pd.Series(
                ["A", "B", "A", "C"],
                dtype="category",
            ),
            "employed": pd.Series(
                [True, True, False, True],
                dtype="bool",
            ),
        }
    )

    y = pd.Series(
        ["no", "yes", "no", "yes"],
        name="class",
        dtype="category",
    )

    return SimpleNamespace(
        data=X,
        target=y,
        details={
            "id": "123",
            "name": "example-dataset",
            "version": "2",
        },
        DESCR="Example OpenML dataset used for unit testing.",
    )


@patch("datasets.openml_loader.fetch_openml")
def test_loader_requests_dataset_by_id(mock_fetch_openml):
    """The loader should request the exact OpenML dataset ID."""
    mock_fetch_openml.return_value = _create_openml_response()

    load_openml_dataset(123)

    mock_fetch_openml.assert_called_once_with(
        data_id=123,
        data_home=None,
        cache=True,
        as_frame=True,
        return_X_y=False,
    )


@patch("datasets.openml_loader.fetch_openml")
def test_loader_returns_tabular_dataset_metadata(mock_fetch_openml):
    """OpenML metadata should be mapped to the common dataset model."""
    mock_fetch_openml.return_value = _create_openml_response()

    dataset = load_openml_dataset(123)

    assert dataset.name == "example-dataset"
    assert dataset.source == "openml"
    assert dataset.source_id == 123
    assert dataset.source_version == 2
    assert dataset.target_name == "class"


@patch("datasets.openml_loader.fetch_openml")
def test_loader_preserves_features_and_target(mock_fetch_openml):
    """The loader should preserve the feature matrix and target."""
    response = _create_openml_response()
    mock_fetch_openml.return_value = response

    dataset = load_openml_dataset(123)

    pd.testing.assert_frame_equal(
        dataset.X,
        response.data,
    )

    pd.testing.assert_series_equal(
        dataset.y,
        response.target,
    )


@patch("datasets.openml_loader.fetch_openml")
def test_loader_infers_feature_types(mock_fetch_openml):
    """Feature types should be inferred from pandas data types."""
    mock_fetch_openml.return_value = _create_openml_response()

    dataset = load_openml_dataset(123)

    assert dataset.categorical_features == (
        "occupation",
        "employed",
    )

    assert dataset.numerical_features == (
        "age",
        "income",
    )


@patch("datasets.openml_loader.fetch_openml")
def test_loader_preserves_openml_metadata(mock_fetch_openml):
    """Additional OpenML metadata should remain available."""
    mock_fetch_openml.return_value = _create_openml_response()

    dataset = load_openml_dataset(123)

    assert (
        dataset.metadata["description"]
        == "Example OpenML dataset used for unit testing."
    )

    assert dataset.metadata["openml_details"]["version"] == "2"


@patch("datasets.openml_loader.fetch_openml")
def test_loader_passes_custom_cache_configuration(mock_fetch_openml):
    """Custom cache configuration should be forwarded to scikit-learn."""
    mock_fetch_openml.return_value = _create_openml_response()

    load_openml_dataset(
        123,
        data_home="data/cache",
        cache=False,
    )

    mock_fetch_openml.assert_called_once_with(
        data_id=123,
        data_home="data/cache",
        cache=False,
        as_frame=True,
        return_X_y=False,
    )


@pytest.mark.parametrize(
    "invalid_data_id",
    [0, -1, -42],
)
def test_non_positive_data_id_raises_value_error(invalid_data_id):
    """OpenML IDs must be positive integers."""
    with pytest.raises(
        ValueError,
        match="data_id must be positive",
    ):
        load_openml_dataset(invalid_data_id)


@pytest.mark.parametrize(
    "invalid_data_id",
    [1.5, "123", None, True],
)
def test_non_integer_data_id_raises_type_error(invalid_data_id):
    """Non-integer OpenML IDs should be rejected."""
    with pytest.raises(
        TypeError,
        match="data_id must be an integer",
    ):
        load_openml_dataset(invalid_data_id)


@patch("datasets.openml_loader.fetch_openml")
def test_multidimensional_target_raises_value_error(
    mock_fetch_openml,
):
    """Multi-output targets should be rejected by the loader."""
    response = _create_openml_response()

    response.target = pd.DataFrame(
        {
            "target_1": [0, 1, 0, 1],
            "target_2": [1, 0, 1, 0],
        }
    )

    mock_fetch_openml.return_value = response

    with pytest.raises(
        ValueError,
        match="target must be a one-dimensional pandas Series",
    ):
        load_openml_dataset(123)