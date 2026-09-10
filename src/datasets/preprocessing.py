"""Leakage-safe preprocessing utilities for tabular datasets.

This module builds preprocessing pipelines for heterogeneous tabular
classification data. Numerical and categorical transformations are kept
separate.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_tabular_preprocessor(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Create the standard preprocessing pipeline used by baselines.
    Numerical features are median-imputed and standardised. Categorical
    features are imputed using their most frequent value and one-hot
    encoded.
    """
    numerical = tuple(numerical_features)
    categorical = tuple(categorical_features)

    _validate_feature_groups(
        numerical_features=numerical,
        categorical_features=categorical,
    )

    transformers = []

    if numerical:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                list(numerical),
            )
        )

    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(categorical),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def fit_transform_tabular(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Fit a tabular preprocessor and transform training data.

    Args:
        preprocessor (ColumnTransformer): Preprocessing pipeline to fit.
        X (pd.DataFrame): Training features used to estimate preprocessing parameters.
    
    Returns:
        np.ndarray: Transformed numerical feature matrix.
    """
    _validate_feature_matrix(X)

    return np.asarray(
        preprocessor.fit_transform(X),
        dtype=float,
    )


def transform_tabular(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Transform data using an already fitted tabular preprocessor.

    Args:
        preprocessor (ColumnTransformer): Previously fitted preprocessing pipeline.
        X (pd.DataFrame): Features to transform.

    Returns:
        np.ndarray: Transformed numerical feature matrix.
    """
    _validate_feature_matrix(X)

    return np.asarray(
        preprocessor.transform(X),
        dtype=float,
    )


def _validate_feature_groups(
    *,
    numerical_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> None:
    """Validate numerical and categorical feature declarations.
    
    Args:
        numerical_features (tuple[str, ...]): Names of numerical features.
        categorical_features (tuple[str, ...]): Names of categorical features.
    """
    if not numerical_features and not categorical_features:
        raise ValueError(
            "At least one feature must be provided."
        )

    all_features = (
        numerical_features
        + categorical_features
    )

    if len(all_features) != len(set(all_features)):
        raise ValueError(
            "Feature names must be unique across preprocessing groups."
        )

    if not all(
        isinstance(feature, str) and feature
        for feature in all_features
    ):
        raise ValueError(
            "Feature names must be non-empty strings."
        )


def _validate_feature_matrix(
    X: pd.DataFrame,
) -> None:
    """Validate a feature matrix before preprocessing.
    
    Args:
        X (pd.DataFrame): Feature matrix to validate.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if len(X) == 0:
        raise ValueError(
            "X must contain at least one sample."
        )