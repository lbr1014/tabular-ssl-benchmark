"""Shared pytest fixtures for benchmark tests."""

import numpy as np
import pandas as pd
import pytest

from datasets.dataset import TabularDataset


@pytest.fixture
def mixed_dataset() -> TabularDataset:
    """Create a deterministic mixed-type binary classification dataset.

    Returns:
        TabularDataset: Synthetic binary classification dataset for testing.
    """
    n_samples = 100

    X = pd.DataFrame(
        {
            "feature_a": np.linspace(
                0.0,
                10.0,
                n_samples,
            ),
            "feature_b": np.tile(
                [1.0, 2.0, 3.0, 4.0],
                n_samples // 4,
            ),
            "category": np.tile(
                ["a", "b"],
                n_samples // 2,
            ),
        }
    )

    y = pd.Series(
        np.tile(
            [0, 1],
            n_samples // 2,
        ),
        name="target",
    )

    return TabularDataset(
        name="synthetic_mixed",
        X=X,
        y=y,
        target_name="target",
        source="synthetic",
        numerical_features=(
            "feature_a",
            "feature_b",
        ),
        categorical_features=("category",),
    )