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
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


TargetArray = Sequence | np.ndarray


def compute_classification_metrics(
    y_true: TargetArray,
    y_pred: TargetArray,
) -> dict[str, float]:
    """Compute label-based classification metrics.

    Args:
        y_true (TargetArray): Ground-truth class labels.
        y_pred (TargetArray): Predicted class labels.

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
        y_true (np.ndarray): Ground-truth class labels.
        y_pred (np.ndarray): Predicted class labels.
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
        
def compute_probabilistic_metrics(
    y_true: TargetArray,
    y_proba: np.ndarray,
    classes: TargetArray,
) -> dict[str, float]:
    """Compute probability-based classification metrics.

    Args:
        y_true (TargetArray): Ground-truth class labels.
        y_proba (np.ndarray): Predicted class probabilities with shape ``(n_samples, n_classes)``.
        classes (TargetArray): Class labels corresponding to probability columns.

    Returns:
        dict[str, float]: Dictionary containing the computed metrics.
            - "log_loss": Logarithmic loss.
            - "roc_auc": Area under the receiver operating characteristic curve.    
    """
    true = np.asarray(y_true)
    probabilities = np.asarray(y_proba)
    class_labels = np.asarray(classes)

    _validate_probabilistic_inputs(
        y_true=true,
        y_proba=probabilities,
        classes=class_labels,
    )

    metrics = {
        "log_loss": float(
            log_loss(
                true,
                probabilities,
                labels=class_labels,
            )
        ),
    }

    if len(class_labels) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                true,
                probabilities[:, 1],
                labels=class_labels,
            )
        )
    else:
        metrics["roc_auc"] = float(
            roc_auc_score(
                true,
                probabilities,
                labels=class_labels,
                multi_class="ovr",
                average="macro",
            )
        )

    return metrics

def _validate_probabilistic_inputs(
    *,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> None:
    """Validate inputs used by probability-based metrics.
    
    Args:
        y_true (np.ndarray): Ground-truth class labels.
        y_proba (np.ndarray): Predicted class probabilities with shape ``(n_samples, n_classes)``.
        classes (np.ndarray): Class labels corresponding to probability columns.
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