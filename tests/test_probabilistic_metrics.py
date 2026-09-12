"""Tests for probability-based classification metrics."""

import numpy as np
import pytest

from evaluation.metrics import compute_probabilistic_metrics


def test_perfect_binary_probabilities_produce_perfect_auc():
    """Perfect binary ranking should produce ROC-AUC equal to one."""
    y_true = np.array([0, 0, 1, 1])

    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    metrics = compute_probabilistic_metrics(
        y_true,
        y_proba,
        classes=np.array([0, 1]),
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["log_loss"] >= 0.0
    
def test_probabilistic_metrics_return_expected_keys():
    """Probability evaluation should expose stable metric names."""
    metrics = compute_probabilistic_metrics(
        np.array([0, 1]),
        np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        ),
        classes=np.array([0, 1]),
    )

    assert set(metrics) == {
        "log_loss",
        "roc_auc",
    }


def test_probabilistic_metrics_are_python_floats():
    """Probability metrics should return serializable Python floats."""
    metrics = compute_probabilistic_metrics(
        np.array([0, 1]),
        np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        ),
        classes=np.array([0, 1]),
    )

    assert all(
        isinstance(value, float)
        for value in metrics.values()
    )
    
def test_probabilistic_metrics_support_multiclass():
    """Probability metrics should support multiclass classification."""
    y_true = np.array([0, 1, 2, 0, 1, 2])

    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = compute_probabilistic_metrics(
        y_true,
        y_proba,
        classes=np.array([0, 1, 2]),
    )

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    
def test_probabilistic_metrics_support_multiclass():
    """Probability metrics should support multiclass classification."""
    y_true = np.array([0, 1, 2, 0, 1, 2])

    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = compute_probabilistic_metrics(
        y_true,
        y_proba,
        classes=np.array([0, 1, 2]),
    )

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    
def test_probabilistic_metrics_reject_wrong_probability_columns():
    """Probability matrices must contain one column per class."""
    with pytest.raises(
        ValueError,
        match="one column per class",
    ):
        compute_probabilistic_metrics(
            np.array([0, 1]),
            np.array(
                [
                    [0.5, 0.3, 0.2],
                    [0.1, 0.8, 0.1],
                ]
            ),
            classes=np.array([0, 1]),
        )


def test_probabilistic_metrics_reject_probabilities_outside_range():
    """Class probabilities must remain within the unit interval."""
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        compute_probabilistic_metrics(
            np.array([0, 1]),
            np.array(
                [
                    [1.1, -0.1],
                    [0.2, 0.8],
                ]
            ),
            classes=np.array([0, 1]),
        )


def test_probabilistic_metrics_reject_non_normalized_rows():
    """Each probability row must represent a valid distribution."""
    with pytest.raises(
        ValueError,
        match="sum to 1",
    ):
        compute_probabilistic_metrics(
            np.array([0, 1]),
            np.array(
                [
                    [0.7, 0.7],
                    [0.2, 0.8],
                ]
            ),
            classes=np.array([0, 1]),
        )


def test_probabilistic_metrics_reject_unknown_true_labels():
    """Ground-truth labels must belong to the declared classes."""
    with pytest.raises(
        ValueError,
        match="not present in classes",
    ):
        compute_probabilistic_metrics(
            np.array([0, 2]),
            np.array(
                [
                    [0.8, 0.2],
                    [0.3, 0.7],
                ]
            ),
            classes=np.array([0, 1]),
        )
        
def test_log_loss_penalizes_overconfident_wrong_predictions():
    """Log loss should penalize confident incorrect probabilities."""
    y_true = np.array([0, 1])

    moderate_proba = np.array(
        [
            [0.6, 0.4],
            [0.6, 0.4],
        ]
    )

    overconfident_proba = np.array(
        [
            [0.99, 0.01],
            [0.99, 0.01],
        ]
    )

    moderate = compute_probabilistic_metrics(
        y_true,
        moderate_proba,
        classes=np.array([0, 1]),
    )

    overconfident = compute_probabilistic_metrics(
        y_true,
        overconfident_proba,
        classes=np.array([0, 1]),
    )

    assert (
        overconfident["log_loss"]
        > moderate["log_loss"]
    )