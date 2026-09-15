"""Tests for the supervised reference SSL strategy."""

import numpy as np
import pandas as pd

from models.sklearn_models import create_logistic_regression
from ssl_methods.supervised import SupervisedMethod


def test_supervised_method_has_stable_name() -> None:
    """Supervised strategy should expose its stable benchmark identifier."""
    method = SupervisedMethod()

    assert method.name == "supervised"
    
def test_supervised_method_fits_classifier() -> None:
    """Supervised strategy should fit the provided classifier."""
    x_labeled = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0],
        }
    )
    y_labeled = np.array([0, 0, 1, 1])

    x_unlabeled = pd.DataFrame(
        {
            "feature": [10.0, 20.0],
        }
    )

    model = create_logistic_regression(seed=42)
    method = SupervisedMethod()

    fitted_model = method.fit(
        model=model,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        x_unlabeled=x_unlabeled,
    )

    predictions = fitted_model.predict(x_labeled)

    assert fitted_model is model
    assert predictions.shape == (len(x_labeled),)
    
    assert fitted_model is model
    
def test_supervised_method_ignores_unlabeled_data() -> None:
    """Changing unlabeled samples must not affect supervised predictions."""
    x_labeled = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0],
        }
    )
    y_labeled = np.array([0, 0, 1, 1])

    first_unlabeled = pd.DataFrame(
        {
            "feature": [10.0, 20.0],
        }
    )

    second_unlabeled = pd.DataFrame(
        {
            "feature": [-1000.0, 1000.0],
        }
    )

    first_model = create_logistic_regression(seed=42)
    second_model = create_logistic_regression(seed=42)

    method = SupervisedMethod()

    first_fitted = method.fit(
        model=first_model,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        x_unlabeled=first_unlabeled,
    )

    second_fitted = method.fit(
        model=second_model,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        x_unlabeled=second_unlabeled,
    )

    first_probabilities = first_fitted.predict_proba(x_labeled)
    second_probabilities = second_fitted.predict_proba(x_labeled)

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
    )