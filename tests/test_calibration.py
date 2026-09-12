"""Tests for classification calibration metrics."""

import numpy as np
import pytest

from evaluation.calibration import (
    compute_calibration_metrics,
    expected_calibration_error,
    multiclass_brier_score,
)


def test_ece_is_zero_for_perfect_confident_predictions():
    """Perfect and fully confident predictions should have zero ECE."""
    y_true = np.array([0, 1, 0, 1])

    y_proba = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    ece = expected_calibration_error(
        y_true,
        y_proba,
        classes=np.array([0, 1]),
    )

    assert ece == pytest.approx(0.0)
    
def test_ece_matches_hand_computed_example():
    """ECE should match a simple manually computed calibration error."""
    y_true = np.array([0, 1])

    y_proba = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )

    ece = expected_calibration_error(
        y_true,
        y_proba,
        classes=np.array([0, 1]),
        n_bins=10,
    )

    expected = (
        abs(1.0 - 0.8) / 2
        + abs(1.0 - 0.7) / 2
    )

    assert ece == pytest.approx(expected)
    
def test_ece_supports_multiclass_probabilities():
    """Confidence ECE should support multiclass classification."""
    y_true = np.array([0, 1, 2])

    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    ece = expected_calibration_error(
        y_true,
        y_proba,
        classes=np.array([0, 1, 2]),
    )

    assert 0.0 <= ece <= 1.0
    
def test_brier_score_is_zero_for_perfect_predictions():
    """Perfect probabilistic predictions should have zero Brier score."""
    y_true = np.array([0, 1])

    y_proba = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    score = multiclass_brier_score(
        y_true,
        y_proba,
        classes=np.array([0, 1]),
    )

    assert score == pytest.approx(0.0)
    
def test_brier_score_supports_multiclass_targets():
    """Brier score should support multiclass probability vectors."""
    y_true = np.array([0, 1, 2])

    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    score = multiclass_brier_score(
        y_true,
        y_proba,
        classes=np.array([0, 1, 2]),
    )

    assert score >= 0.0
    
def test_brier_score_rewards_better_probabilities():
    """More accurate probability assignments should lower Brier score."""
    y_true = np.array([0, 1])

    good_proba = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    poor_proba = np.array(
        [
            [0.6, 0.4],
            [0.4, 0.6],
        ]
    )

    good_score = multiclass_brier_score(
        y_true,
        good_proba,
        classes=np.array([0, 1]),
    )

    poor_score = multiclass_brier_score(
        y_true,
        poor_proba,
        classes=np.array([0, 1]),
    )

    assert good_score < poor_score
    
def test_calibration_metrics_return_expected_keys():
    """Calibration evaluation should expose stable metric names."""
    metrics = compute_calibration_metrics(
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
        "brier_score",
        "ece",
    }


def test_calibration_metrics_are_python_floats():
    """Calibration metrics should return serializable Python floats."""
    metrics = compute_calibration_metrics(
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
    
@pytest.mark.parametrize(
    "n_bins",
    [0, -1],
)
def test_ece_rejects_non_positive_bin_counts(n_bins):
    """ECE requires a positive number of bins."""
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        expected_calibration_error(
            np.array([0, 1]),
            np.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                ]
            ),
            classes=np.array([0, 1]),
            n_bins=n_bins,
        )


def test_ece_rejects_non_integer_bin_count():
    """ECE bin count must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        expected_calibration_error(
            np.array([0, 1]),
            np.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                ]
            ),
            classes=np.array([0, 1]),
            n_bins=10.5,
        )