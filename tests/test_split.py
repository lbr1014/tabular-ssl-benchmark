"""Tests for semi-supervised dataset splitting utilities."""

import numpy as np
import pytest
from datasets.split import create_ssl_split


@pytest.fixture
def balanced_target():
    """Return a balanced binary target for splitting tests."""
    return np.array([0, 1] * 100)


def test_split_is_reproducible(balanced_target):
    """The same seed should produce identical dataset splits."""
    first = create_ssl_split(
        y=balanced_target,
        label_fraction=0.2,
        test_size=0.2,
        seed=42,
    )

    second = create_ssl_split(
        y=balanced_target,
        label_fraction=0.2,
        test_size=0.2,
        seed=42,
    )

    np.testing.assert_array_equal(
        first.train_indices,
        second.train_indices,
    )
    np.testing.assert_array_equal(
        first.test_indices,
        second.test_indices,
    )
    np.testing.assert_array_equal(
        first.labeled_indices,
        second.labeled_indices,
    )
    np.testing.assert_array_equal(
        first.unlabeled_indices,
        second.unlabeled_indices,
    )


def test_train_and_test_do_not_overlap(balanced_target):
    """Train and test samples should be completely disjoint."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=0.2,
        test_size=0.2,
        seed=42,
    )

    overlap = np.intersect1d(
        split.train_indices,
        split.test_indices,
    )

    assert len(overlap) == 0


def test_labeled_and_unlabeled_do_not_overlap(balanced_target):
    """Labelled and unlabelled training subsets should be disjoint."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=0.2,
        test_size=0.2,
        seed=42,
    )

    overlap = np.intersect1d(
        split.labeled_indices,
        split.unlabeled_indices,
    )

    assert len(overlap) == 0


def test_labeled_and_unlabeled_cover_training_set(balanced_target):
    """Labelled and unlabelled samples should cover the full train set."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=0.2,
        test_size=0.2,
        seed=42,
    )

    reconstructed_train = np.concatenate(
        [split.labeled_indices, split.unlabeled_indices]
    )

    np.testing.assert_array_equal(
        np.sort(reconstructed_train),
        np.sort(split.train_indices),
    )


def test_test_set_has_expected_size(balanced_target):
    """The test set should contain the requested fraction of samples."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=0.2,
        test_size=0.2,
        seed=42,
    )

    assert len(split.test_indices) == 40


def test_label_fraction_has_expected_size(balanced_target):
    """The labelled subset should contain the requested train fraction."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=0.25,
        test_size=0.2,
        seed=42,
    )

    assert len(split.train_indices) == 160
    assert len(split.labeled_indices) == 40
    assert len(split.unlabeled_indices) == 120


def test_full_label_fraction_uses_all_training_samples(balanced_target):
    """A label fraction of one should label the complete training set."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=1.0,
        test_size=0.2,
        seed=42,
    )

    np.testing.assert_array_equal(
        split.labeled_indices,
        split.train_indices,
    )

    assert len(split.unlabeled_indices) == 0


def test_test_samples_are_never_used_for_ssl_training(balanced_target):
    """Test samples should not appear in labelled or unlabelled subsets."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=0.2,
        test_size=0.2,
        seed=42,
    )

    assert len(
        np.intersect1d(split.test_indices, split.labeled_indices)
    ) == 0

    assert len(
        np.intersect1d(split.test_indices, split.unlabeled_indices)
    ) == 0


@pytest.mark.parametrize(
    "label_fraction",
    [0.0, -0.1, 1.1],
)
def test_invalid_label_fraction_raises_value_error(
    balanced_target,
    label_fraction,
):
    """Invalid label fractions should be rejected."""
    with pytest.raises(
        ValueError,
        match="label_fraction must be in the interval",
    ):
        create_ssl_split(
            y=balanced_target,
            label_fraction=label_fraction,
            test_size=0.2,
            seed=42,
        )


@pytest.mark.parametrize(
    "test_size",
    [0.0, -0.1, 1.0, 1.1],
)
def test_invalid_test_size_raises_value_error(
    balanced_target,
    test_size,
):
    """Invalid test sizes should be rejected."""
    with pytest.raises(
        ValueError,
        match="test_size must be in the interval",
    ):
        create_ssl_split(
            y=balanced_target,
            label_fraction=0.2,
            test_size=test_size,
            seed=42,
        )


def test_multidimensional_target_raises_value_error():
    """A multidimensional target should be rejected."""
    y = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="y must be a one-dimensional array",
    ):
        create_ssl_split(
            y=y,
            label_fraction=0.2,
            test_size=0.2,
            seed=42,
        )


def test_single_class_target_raises_value_error():
    """A target containing only one class should be rejected."""
    y = np.zeros(100, dtype=int)

    with pytest.raises(
        ValueError,
        match="y must contain at least two classes",
    ):
        create_ssl_split(
            y=y,
            label_fraction=0.2,
            test_size=0.2,
            seed=42,
        )
        
def test_split_preserves_class_balance(balanced_target):
    """Stratified splitting should preserve class proportions."""
    split = create_ssl_split(
        y=balanced_target,
        label_fraction=0.25,
        test_size=0.2,
        seed=42,
    )

    train_positive_rate = balanced_target[
        split.train_indices
    ].mean()

    test_positive_rate = balanced_target[
        split.test_indices
    ].mean()

    labeled_positive_rate = balanced_target[
        split.labeled_indices
    ].mean()

    assert train_positive_rate == pytest.approx(0.5)
    assert test_positive_rate == pytest.approx(0.5)
    assert labeled_positive_rate == pytest.approx(0.5)