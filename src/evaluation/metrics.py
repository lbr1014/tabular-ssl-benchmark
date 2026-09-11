"""Classification evaluation metrics for benchmark experiments.
This module provides deterministic metrics computed from ground-truth
and predicted class labels. 
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)


TargetArray = Sequence | np.ndarray


def compute_classification_metrics(
    y_true: TargetArray,
    y_pred: TargetArray,
) -> dict[str, float]:
    """Compute label-based classification metrics.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        
    Returns:
        dict[str, float]: Dictionary containing the computed metrics:
            - "accuracy": Overall accuracy.
            - "balanced_accuracy": Balanced accuracy.
            - "f1_macro": Macro-averaged F1 score.
            - "mcc": Matthews correlation coefficient.
    """
    
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    _validate_targets(
        y_true=true,
        y_pred=predicted,
    )

    return {
        "accuracy": float(
            accuracy_score(true, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(true, predicted)
        ),
        "f1_macro": float(
            f1_score(
                true,
                predicted,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(true, predicted)
        ),
    }


def _validate_targets(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Validate targets used by classification metrics.
    
    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
    """
    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if y_pred.ndim != 1:
        raise ValueError(
            "y_pred must be one-dimensional."
        )

    if len(y_true) == 0:
        raise ValueError(
            "y_true must contain at least one sample."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )