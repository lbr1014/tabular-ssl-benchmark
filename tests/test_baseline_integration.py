"""Integration tests for supervised baseline classification.
These tests verify that dataset splitting, leakage-safe preprocessing,
and supervised benchmark classifiers work together through their public
interfaces.
"""

import numpy as np
import pandas as pd
import pytest

from datasets.dataset import TabularDataset
from datasets.preprocessing import (
    create_tabular_preprocessor,
    fit_transform_tabular,
    transform_tabular,
)
from datasets.split import create_ssl_split
from models.sklearn_models import (
    create_logistic_regression,
    create_random_forest,
)


@pytest.fixture
def mixed_dataset() -> TabularDataset:
    """Return a deterministic mixed-type binary classification dataset."""
    n_samples = 60

    X = pd.DataFrame(
        {
            "age": np.tile(
                [20.0, 30.0, 40.0, 50.0, 60.0, np.nan],
                10,
            ),
            "income": np.linspace(
                20_000.0,
                80_000.0,
                n_samples,
            ),
            "sector": np.tile(
                ["public", "private", "academic"],
                20,
            ),
        }
    )

    y = pd.Series(
        np.tile([0, 1], n_samples // 2),
        name="target",
    )

    return TabularDataset(
        name="synthetic_mixed",
        X=X,
        y=y,
        target_name="target",
        source="synthetic",
        numerical_features=("age", "income"),
        categorical_features=("sector",),
    )
    
def _prepare_supervised_split(
    dataset: TabularDataset,
    *,
    label_fraction: float,
    test_size: float,
    seed: int,
):
    """Prepare labelled training and held-out test data."""
    split = create_ssl_split(
        dataset.y,
        label_fraction=label_fraction,
        test_size=test_size,
        seed=seed,
    )

    X_labeled = dataset.X.iloc[
        split.labeled_indices
    ]
    y_labeled = dataset.y.iloc[
        split.labeled_indices
    ]

    X_test = dataset.X.iloc[
        split.test_indices
    ]
    y_test = dataset.y.iloc[
        split.test_indices
    ]

    preprocessor = create_tabular_preprocessor(
        numerical_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
    )

    X_labeled_transformed = fit_transform_tabular(
        preprocessor,
        X_labeled,
    )

    X_test_transformed = transform_tabular(
        preprocessor,
        X_test,
    )

    return (
        X_labeled_transformed,
        y_labeled.to_numpy(),
        X_test_transformed,
        y_test.to_numpy(),
        split,
    )
    
def test_logistic_regression_baseline_end_to_end(
    mixed_dataset: TabularDataset,
):
    """Logistic regression should run through the baseline pipeline."""
    (
        X_train,
        y_train,
        X_test,
        y_test,
        _,
    ) = _prepare_supervised_split(
        mixed_dataset,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    model = create_logistic_regression(seed=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert predictions.shape == y_test.shape

    assert probabilities.shape == (
        len(y_test),
        len(model.classes_),
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )
    
def test_random_forest_baseline_end_to_end(
    mixed_dataset: TabularDataset,
):
    """Random forest should run through the baseline pipeline."""
    (
        X_train,
        y_train,
        X_test,
        y_test,
        _,
    ) = _prepare_supervised_split(
        mixed_dataset,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    model = create_random_forest(seed=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert predictions.shape == y_test.shape

    assert probabilities.shape == (
        len(y_test),
        len(model.classes_),
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )
    
def test_baseline_trains_only_on_labeled_partition(
    mixed_dataset: TabularDataset,
):
    """Baseline training size should equal the labelled split size."""
    (
        X_train,
        y_train,
        _,
        _,
        split,
    ) = _prepare_supervised_split(
        mixed_dataset,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert len(X_train) == len(
        split.labeled_indices
    )

    assert len(y_train) == len(
        split.labeled_indices
    )

    assert len(split.unlabeled_indices) > 0
    
def test_test_partition_is_disjoint_from_baseline_training(
    mixed_dataset: TabularDataset,
):
    """Held-out test samples must never belong to baseline training."""
    split = create_ssl_split(
        mixed_dataset.y,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    labeled = set(
        split.labeled_indices.tolist()
    )
    test = set(
        split.test_indices.tolist()
    )

    assert labeled.isdisjoint(test)
    
def test_baseline_pipeline_is_reproducible(
    mixed_dataset: TabularDataset,
):
    """Equal root seeds should reproduce baseline predictions."""
    first = _prepare_supervised_split(
        mixed_dataset,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    second = _prepare_supervised_split(
        mixed_dataset,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    X_train_1, y_train_1, X_test_1, _, _ = first
    X_train_2, y_train_2, X_test_2, _, _ = second

    first_model = create_random_forest(seed=42)
    second_model = create_random_forest(seed=42)

    first_model.fit(
        X_train_1,
        y_train_1,
    )

    second_model.fit(
        X_train_2,
        y_train_2,
    )

    first_probabilities = first_model.predict_proba(
        X_test_1
    )

    second_probabilities = second_model.predict_proba(
        X_test_2
    )

    assert np.allclose(
        first_probabilities,
        second_probabilities,
    )