"""Tests for label-based classification evaluation metrics."""

import numpy as np
import pytest

from evaluation.metrics import compute_classification_metrics


def test_perfect_predictions_produce_perfect_scores():
    """Perfect predictions should maximize all classification metrics."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)


def test_metrics_return_expected_keys():
    """Classification evaluation should expose stable metric names."""
    metrics = compute_classification_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 1, 1]),
    )

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "mcc",
    }


def test_metrics_are_python_floats():
    """Metric values should be serializable Python floats."""
    metrics = compute_classification_metrics(
        np.array([0, 1]),
        np.array([0, 1]),
    )

    assert all(
        isinstance(value, float)
        for value in metrics.values()
    )


def test_metrics_support_multiclass_targets():
    """Classification metrics should support multiclass problems."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert -1.0 <= metrics["mcc"] <= 1.0


def test_metrics_support_string_labels():
    """Classification metrics should support non-numeric class labels."""
    y_true = np.array(["cat", "dog", "cat", "dog"])
    y_pred = np.array(["cat", "dog", "dog", "dog"])

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)


def test_metrics_reject_different_target_lengths():
    """Ground truth and predictions must contain equal sample counts."""
    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        compute_classification_metrics(
            np.array([0, 1, 0]),
            np.array([0, 1]),
        )


def test_metrics_reject_empty_targets():
    """Evaluation requires at least one sample."""
    with pytest.raises(
        ValueError,
        match="at least one sample",
    ):
        compute_classification_metrics(
            np.array([]),
            np.array([]),
        )


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        (
            np.array([[0], [1]]),
            np.array([0, 1]),
        ),
        (
            np.array([0, 1]),
            np.array([[0], [1]]),
        ),
    ],
)
def test_metrics_reject_multidimensional_targets(
    y_true,
    y_pred,
):
    """Classification targets must be one-dimensional."""
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        compute_classification_metrics(
            y_true,
            y_pred,
        )
        
def test_balanced_accuracy_detects_class_imbalance():
    """Balanced accuracy should expose majority-class-only predictions."""
    y_true = np.array(
        [0] * 90 + [1] * 10
    )

    y_pred = np.zeros(
        100,
        dtype=int,
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)