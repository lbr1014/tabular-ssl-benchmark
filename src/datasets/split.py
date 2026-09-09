"""Utilities for reproducible semi-supervised data splitting.

This module provides the data splitting logic used by the benchmark.
Each dataset is first divided into training and test sets. The training
set is then divided into labelled and unlabelled subsets according to
the requested label fraction.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """Container for a semi-supervised dataset split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    labeled_indices: np.ndarray
    unlabeled_indices: np.ndarray


def create_ssl_split(
    y: np.ndarray,
    label_fraction: float,
    test_size: float,
    seed: int,
) -> DatasetSplit:
    """Create reproducible train, test, labelled, and unlabelled splits.
    
    Args:
        y (np.ndarray): One dimension array of target labels.
        label_fraction (float): Fraction of labelled training samples.
        test_size (float): Fraction of samples reserved for testing.
        seed (int): Random seed for reproducibility.
        
    Returns:
        DatasetSplit: A dataclass containing the indices for the train, 
                      test, labelled, and unlabelled splits.
    """
    
    y = np.asarray(y)

    _validate_split_inputs(
        y=y,
        label_fraction=label_fraction,
        test_size=test_size,
    )

    indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    if label_fraction == 1.0:
        labeled_indices = train_indices.copy()
        unlabeled_indices = np.array([], dtype=int)
    else:
        labeled_indices, unlabeled_indices = train_test_split(
            train_indices,
            train_size=label_fraction,
            random_state=seed,
            stratify=y[train_indices],
        )

    return DatasetSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
    )


def _validate_split_inputs(
    y: np.ndarray,
    label_fraction: float,
    test_size: float,
) -> None:
    """Validate inputs used to construct a semi-supervised split.

    Args:
        y (np.ndarray): One dimension array of target labels.
        label_fraction (float): Fraction of labelled training samples.
        test_size (float): Fraction of samples reserved for testing.

    """
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if len(y) == 0:
        raise ValueError("y must contain at least one sample.")

    if len(np.unique(y)) < 2:
        raise ValueError("y must contain at least two classes.")

    if not 0 < label_fraction <= 1:
        raise ValueError("label_fraction must be in the interval (0, 1].")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in the interval (0, 1).")