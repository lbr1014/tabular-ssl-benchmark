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