"""Calibration metrics for probabilistic classification.
This module provides metrics for evaluating the calibration of predicted
class probabilities. Expected Calibration Error (ECE) is implemented
using top-label confidence and equal-width bins, supporting both binary
and multiclass classification.
"""

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute confidence-based Expected Calibration Error.

    Args:
        y_true (np.ndarray): Ground-truth class labels with shape ``(n_samples,)``.
        y_proba (np.ndarray): Predicted class probabilities with shape ``(n_samples, n_classes)``. Columns must correspond to ``classes``.
        classes (np.ndarray): Class labels corresponding to probability columns.
        n_bins (int, default=10): Number of equal-width confidence bins.

    Returns:
        float: Expected Calibration Error. Lower values indicate better
        confidence calibration.
    """
    
    true = np.asarray(y_true)
    probabilities = np.asarray(y_proba)
    class_labels = np.asarray(classes)

    _validate_calibration_inputs(
        y_true=true,
        y_proba=probabilities,
        classes=class_labels,
        n_bins=n_bins,
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1,
    )

    predictions = class_labels[
        predicted_indices
    ]

    confidences = np.max(
        probabilities,
        axis=1,
    )

    correct = predictions == true

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == n_bins - 1:
            in_bin = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            in_bin = (
                (confidences >= lower)
                & (confidences < upper)
            )

        bin_count = np.sum(in_bin)

        if bin_count == 0:
            continue

        bin_accuracy = np.mean(
            correct[in_bin]
        )

        bin_confidence = np.mean(
            confidences[in_bin]
        )

        bin_weight = (
            bin_count / len(true)
        )

        ece += bin_weight * abs(
            bin_accuracy - bin_confidence
        )

    return float(ece)

def _validate_calibration_inputs(
    *,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int,
) -> None:
    """Validate inputs used by calibration metrics.
    
    Args:
        y_true (np.ndarray): Ground-truth class labels.
        y_proba (np.ndarray): Predicted class probabilities.
        classes (np.ndarray): Class labels.
        n_bins (int): Number of confidence bins.
    """
    
    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if y_proba.ndim != 2:
        raise ValueError(
            "y_proba must be two-dimensional."
        )

    if classes.ndim != 1:
        raise ValueError(
            "classes must be one-dimensional."
        )

    if len(y_true) == 0:
        raise ValueError(
            "y_true must contain at least one sample."
        )

    if len(classes) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    if len(np.unique(classes)) != len(classes):
        raise ValueError(
            "classes must contain unique labels."
        )

    if y_proba.shape[0] != len(y_true):
        raise ValueError(
            "y_true and y_proba must contain the same number of samples."
        )

    if y_proba.shape[1] != len(classes):
        raise ValueError(
            "y_proba must contain one column per class."
        )

    if not np.isfinite(y_proba).all():
        raise ValueError(
            "y_proba must contain only finite values."
        )

    if np.any(y_proba < 0.0) or np.any(y_proba > 1.0):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if not np.allclose(
        y_proba.sum(axis=1),
        1.0,
        rtol=1e-7,
        atol=1e-8,
    ):
        raise ValueError(
            "Probabilities must sum to 1 for each sample."
        )

    if not np.isin(y_true, classes).all():
        raise ValueError(
            "y_true contains labels not present in classes."
        )

    if not isinstance(n_bins, int) or isinstance(n_bins, bool):
        raise TypeError(
            "n_bins must be an integer."
        )

    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )