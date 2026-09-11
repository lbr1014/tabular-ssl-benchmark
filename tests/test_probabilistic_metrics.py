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