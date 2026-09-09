"""Representation of tabular datasets used by the benchmark.

This module defines the dataset structure shared by all benchmark
components. 
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TabularDataset:
    """Represent a tabular classification dataset and its metadata.
    """
    def __post_init__(self) -> None:
        """Validate the internal consistency of the dataset."""

        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not isinstance(self.y, pd.Series):
            raise TypeError("y must be a pandas Series.")

        if len(self.X) == 0:
            raise ValueError("X must contain at least one sample.")

        if self.X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if len(self.X) != len(self.y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if self.y.nunique(dropna=True) < 2:
            raise ValueError(
                "y must contain at least two classes."
            )

        categorical = set(self.categorical_features)
        numerical = set(self.numerical_features)
        available = set(self.X.columns)

        overlap = categorical & numerical

        if overlap:
            raise ValueError(
                "Features cannot be both categorical and numerical: "
                f"{sorted(overlap)}."
            )

        declared = categorical | numerical
        unknown = declared - available

        if unknown:
            raise ValueError(
                "Feature type metadata contains unknown columns: "
                f"{sorted(unknown)}."
            )

        missing = available - declared

        if missing:
            raise ValueError(
                "Every feature must be classified as categorical or numerical. "
                f"Missing features: {sorted(missing)}."
            )



    name: str
    X: pd.DataFrame
    y: pd.Series
    target_name: str
    source: str

    source_id: int | str | None = None
    source_version: int | str | None = None

    categorical_features: tuple[str, ...] = ()
    numerical_features: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    @property
    def n_features(self) -> int:
        """Return the number of input features."""
        return self.X.shape[1]

    @property
    def classes(self) -> np.ndarray:
        """Return the unique target classes."""
        return np.unique(self.y)

    @property
    def n_classes(self) -> int:
        """Return the number of unique target classes."""
        return len(self.classes)