"""OpenML dataset loading utilities.

This module provides the OpenML-specific data loading layer used by the
benchmark. Remote datasets are converted into the common
(TabularDataset) representation before being consumed by the
remaining benchmark components.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

from datasets.dataset import TabularDataset


def load_openml_dataset(
    data_id: int,
    *,
    data_home: str | Path | None = None,
    cache: bool = True,
) -> TabularDataset:
    """Load a classification dataset from OpenML.

    Args:
        data_id (int): OpenML dataset identifier.
        data_home (str | Path | None, optional): Directory where OpenML
            datasets are cached. If None, the default OpenML cache location
            is used.
        cache (bool, default=True): Whether downloaded OpenML data should
            be cached locally.

    Returns:
        TabularDataset: Normalised benchmark dataset representation.
    """
    _validate_data_id(data_id)

    bunch = fetch_openml(
        data_id=data_id,
        data_home=data_home,
        cache=cache,
        as_frame=True,
        return_X_y=False,
    )

    return _build_tabular_dataset(
        bunch=bunch,
        requested_data_id=data_id,
    )


def _build_tabular_dataset(
    bunch: Any,
    requested_data_id: int,
) -> TabularDataset:
    """Convert an OpenML result into a :class:`TabularDataset`.

    Args:
        bunch (Any): OpenML result object returned by ``fetch_openml``.
        requested_data_id (int): OpenML dataset ID originally requested by the caller.

    Returns:
        TabularDataset: Normalised benchmark dataset representation.
    """
    X = bunch.data
    y = bunch.target

    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            "OpenML feature data must be returned as a pandas DataFrame."
        )

    if not isinstance(y, pd.Series):
        raise ValueError(
            "OpenML target must be a one-dimensional pandas Series."
        )

    details = dict(getattr(bunch, "details", {}) or {})

    categorical_features, numerical_features = _infer_feature_types(X)

    name = str(
        details.get(
            "name",
            f"openml-{requested_data_id}",
        )
    )

    source_id = _parse_optional_int(
        details.get("id"),
        default=requested_data_id,
    )

    source_version = _parse_optional_int(
        details.get("version"),
    )

    target_name = (
        str(y.name)
        if y.name is not None
        else "target"
    )

    metadata = {
        "description": getattr(bunch, "DESCR", None),
        "openml_details": details,
    }

    return TabularDataset(
        name=name,
        X=X,
        y=y,
        target_name=target_name,
        source="openml",
        source_id=source_id,
        source_version=source_version,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        metadata=metadata,
    )


def _infer_feature_types(
    X: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Infer categorical and numerical feature names.

    Args:
        X (pd.DataFrame): Feature matrix.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]:
            Categorical feature names followed by numerical feature names.
    """
    categorical_features: list[str] = []
    numerical_features: list[str] = []

    for column in X.columns:
        dtype = X[column].dtype

        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            categorical_features.append(str(column))
        elif pd.api.types.is_numeric_dtype(dtype):
            numerical_features.append(str(column))

    return (
        tuple(categorical_features),
        tuple(numerical_features),
    )


def _validate_data_id(data_id: int) -> None:
    """Validate an OpenML dataset identifier.

    Args:
        data_id (int): OpenML dataset identifier.
    """
    if not isinstance(data_id, int) or isinstance(data_id, bool):
        raise TypeError("data_id must be an integer.")

    if data_id <= 0:
        raise ValueError("data_id must be positive.")


def _parse_optional_int(
    value: Any,
    *,
    default: int | None = None,
) -> int | None:
    """Convert optional OpenML metadata to an integer.

    Args:
        value (Any): Metadata value returned by OpenML.
        default (int | None, optional): Value returned when the metadata field is missing.

    Returns:
        int | None: Parsed integer value or the default if the metadata is missing.
    """
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer-compatible OpenML metadata, got {value!r}."
        ) from exc