"""Tests for the common benchmark classifier interface."""

import numpy as np
import pytest

from models.base import BenchmarkClassifier


def test_benchmark_classifier_cannot_be_instantiated():
    """The abstract classifier interface must not be instantiated."""
    with pytest.raises(TypeError):
        BenchmarkClassifier()


class IncompleteClassifier(BenchmarkClassifier):
    """Classifier intentionally missing abstract implementations."""

    @property
    def name(self) -> str:
        """Return the classifier identifier."""
        return "incomplete"


def test_incomplete_classifier_cannot_be_instantiated():
    """Subclasses must implement the complete classifier contract."""
    with pytest.raises(TypeError):
        IncompleteClassifier()


class DummyClassifier(BenchmarkClassifier):
    """Minimal complete implementation used to test the interface."""

    @property
    def name(self) -> str:
        """Return the classifier identifier."""
        return "dummy"

    @property
    def classes_(self) -> np.ndarray:
        """Return dummy class labels."""
        return np.array([0, 1])

    def fit(self, X, y):
        """Return the fitted dummy classifier."""
        return self

    def predict(self, X):
        """Return deterministic dummy predictions."""
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        """Return deterministic dummy probabilities."""
        return np.tile(
            np.array([[1.0, 0.0]]),
            (len(X), 1),
        )

    def get_params(self):
        """Return empty dummy parameters."""
        return {}


def test_complete_classifier_can_be_instantiated():
    """A complete implementation should satisfy the interface."""
    classifier = DummyClassifier()

    assert classifier.name == "dummy"
    assert np.array_equal(
        classifier.classes_,
        np.array([0, 1]),
    )


def test_fit_returns_classifier():
    """Classifier fit should support the common fitted-model contract."""
    classifier = DummyClassifier()

    X = np.array([
        [1.0],
        [2.0],
    ])
    y = np.array([0, 1])

    fitted = classifier.fit(X, y)

    assert fitted is classifier


def test_prediction_contract():
    """Predictions should follow the expected output dimensions."""
    classifier = DummyClassifier()

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)

    assert predictions.shape == (3,)
    assert probabilities.shape == (3, 2)