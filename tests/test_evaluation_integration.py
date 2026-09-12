"""Integration tests for the benchmark evaluation pipeline.
These tests verify that predictions produced by benchmark classifiers
can be evaluated consistently using classification, probabilistic, and
calibration metrics.
"""

import numpy as np
import pandas as pd

from datasets.dataset import TabularDataset
from datasets.preprocessing import (
    create_tabular_preprocessor,
    fit_transform_tabular,
    transform_tabular,
)
from datasets.split import create_ssl_split
from evaluation.calibration import compute_calibration_metrics
from evaluation.metrics import (
    compute_classification_metrics,
    compute_probabilistic_metrics,
)
from models.sklearn_models import create_logistic_regression

def _create_evaluation_dataset() -> TabularDataset:
    """Create a deterministic mixed-type classification dataset."""
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
        name="evaluation_integration",
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
    
def _run_baseline_pipeline(
    dataset: TabularDataset,
    *,
    seed: int = 42,
):
    """Train a supervised baseline and return test predictions."""
    split = create_ssl_split(
        dataset.y,
        label_fraction=0.5,
        test_size=0.2,
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

    model = create_logistic_regression(
        seed=seed,
    )

    model.fit(
        X_labeled_transformed,
        y_labeled.to_numpy(),
    )

    predictions = model.predict(
        X_test_transformed
    )

    probabilities = model.predict_proba(
        X_test_transformed
    )

    return (
        y_test.to_numpy(),
        predictions,
        probabilities,
        model.classes_,
    )
    
def test_baseline_predictions_support_all_evaluation_metrics():
    """Baseline predictions should support the full evaluation stack."""
    dataset = _create_evaluation_dataset()

    (
        y_test,
        predictions,
        probabilities,
        classes,
    ) = _run_baseline_pipeline(dataset)

    classification_metrics = compute_classification_metrics(
        y_test,
        predictions,
    )

    probabilistic_metrics = compute_probabilistic_metrics(
        y_test,
        probabilities,
        classes,
    )

    calibration_metrics = compute_calibration_metrics(
        y_test,
        probabilities,
        classes,
    )

    assert classification_metrics
    assert probabilistic_metrics
    assert calibration_metrics
    
def test_evaluation_metrics_can_be_combined():
    """All evaluation families should form one result dictionary."""
    dataset = _create_evaluation_dataset()

    (
        y_test,
        predictions,
        probabilities,
        classes,
    ) = _run_baseline_pipeline(dataset)

    metrics = {}

    metrics.update(
        compute_classification_metrics(
            y_test,
            predictions,
        )
    )

    metrics.update(
        compute_probabilistic_metrics(
            y_test,
            probabilities,
            classes,
        )
    )

    metrics.update(
        compute_calibration_metrics(
            y_test,
            probabilities,
            classes,
        )
    )

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "mcc",
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece",
    }

    assert all(
        isinstance(value, float)
        for value in metrics.values()
    )

    assert all(
        np.isfinite(value)
        for value in metrics.values()
    )
    
def test_evaluation_pipeline_is_reproducible():
    """Equal experiment seeds should reproduce evaluation results."""
    dataset = _create_evaluation_dataset()

    first = _run_baseline_pipeline(
        dataset,
        seed=42,
    )

    second = _run_baseline_pipeline(
        dataset,
        seed=42,
    )

    first_y, first_pred, first_proba, first_classes = first
    second_y, second_pred, second_proba, second_classes = second

    assert np.array_equal(
        first_y,
        second_y,
    )

    assert np.array_equal(
        first_pred,
        second_pred,
    )

    assert np.allclose(
        first_proba,
        second_proba,
    )

    assert np.array_equal(
        first_classes,
        second_classes,
    )