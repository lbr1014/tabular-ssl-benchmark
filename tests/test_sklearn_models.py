"""Tests for supervised scikit-learn benchmark classifiers."""

import numpy as np

from models.base import BenchmarkClassifier
from models.sklearn_models import (
    create_logistic_regression,
    create_random_forest,
)


def _classification_data():
    """Return a small deterministic binary classification dataset."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )

    y = np.array([0, 0, 0, 1, 1, 1])

    return X, y


def test_logistic_regression_implements_classifier_interface():
    """Logistic regression should implement the benchmark contract."""
    model = create_logistic_regression(seed=42)

    assert isinstance(model, BenchmarkClassifier)
    assert model.name == "logistic_regression"


def test_random_forest_implements_classifier_interface():
    """Random forest should implement the benchmark contract."""
    model = create_random_forest(seed=42)

    assert isinstance(model, BenchmarkClassifier)
    assert model.name == "random_forest"


def test_logistic_regression_can_fit_and_predict():
    """Logistic regression should fit and predict labelled data."""
    X, y = _classification_data()

    model = create_logistic_regression(seed=42)
    fitted = model.fit(X, y)

    predictions = model.predict(X)

    assert fitted is model
    assert predictions.shape == (len(X),)


def test_random_forest_can_fit_and_predict():
    """Random forest should fit and predict labelled data."""
    X, y = _classification_data()

    model = create_random_forest(seed=42)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)


def test_predict_proba_matches_number_of_classes():
    """Probability output should match samples and learned classes."""
    X, y = _classification_data()

    model = create_random_forest(seed=42)
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (
        len(X),
        len(model.classes_),
    )


def test_probabilities_sum_to_one():
    """Predicted class probabilities should form valid distributions."""
    X, y = _classification_data()

    model = create_logistic_regression(seed=42)
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_model_seed_is_reproducible():
    """Equal experiment seeds should configure equal model seeds."""
    first = create_random_forest(seed=42)
    second = create_random_forest(seed=42)

    assert (
        first.get_params()["random_state"]
        == second.get_params()["random_state"]
    )


def test_different_root_seeds_produce_different_model_seeds():
    """Different experiment seeds should derive different model seeds."""
    first = create_random_forest(seed=1)
    second = create_random_forest(seed=2)

    assert (
        first.get_params()["random_state"]
        != second.get_params()["random_state"]
    )