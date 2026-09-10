"""Supervised scikit-learn baseline classifiers.
Classical scikit-learn classifiers serve as supervised baselines
trained exclusively on the labelled subset of each experimental split.
"""

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from models.base import (
    BenchmarkClassifier,
    FeatureMatrix,
    TargetArray,
)
from utils.seeds import SeedStream, derive_seed, validate_seed


class SklearnClassifier(BenchmarkClassifier):
    """Wrap a scikit-learn classifier using the benchmark interface."""

    def __init__(
        self,
        estimator: ClassifierMixin,
        name: str,
    ) -> None:
        """Initialise the scikit-learn classifier wrapper.
        
        Args:
            estimator (ClassifierMixin): A scikit-learn classifier instance.
            name (str): A stable identifier for the classifier."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if not name.strip():
            raise ValueError("name must not be empty.")

        self._estimator = estimator
        self._name = name

    @property
    def name(self) -> str:
        """Return the stable classifier identifier."""
        return self._name

    @property
    def classes_(self) -> np.ndarray:
        """Return learned classes in probability-column order."""
        return self._estimator.classes_

    def fit(
        self,
        X: FeatureMatrix,
        y: TargetArray,
    ) -> "SklearnClassifier":
        """Fit the underlying classifier on labelled training data.

        Args: 
            X (FeatureMatrix): The feature matrix of the labelled training data.
            y (TargetArray): The target array of the labelled training data.
            
        Returns:
            SklearnClassifier: The fitted classifier instance.
        """
        self._estimator.fit(X, y)
        return self

    def predict(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """Predict class labels using the fitted classifier.
        
        Args:
            X (FeatureMatrix): The feature matrix of the data to predict.
            
        Returns:
            np.ndarray: The predicted class labels.
        """
        return np.asarray(
            self._estimator.predict(X)
        )

    def predict_proba(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """Predict class probabilities using the fitted classifier.
        
        Args:
            X (FeatureMatrix): The feature matrix of the data to predict.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        return np.asarray(
            self._estimator.predict_proba(X)
        )

    def get_params(self) -> dict[str, Any]:
        """Return parameters of the underlying classifier.
        
        Returns:
            dict[str, Any]: A dictionary of parameter names and their values.
        """
        return self._estimator.get_params(deep=True)
    

def create_logistic_regression(
    seed: int,
) -> SklearnClassifier:
    """Create the logistic regression supervised baseline.

    Args:
        seed (int): Root experiment seed. A deterministic model-specific child seed is derived from it.

    Returns:
        SklearnClassifier: Configured logistic regression benchmark classifier.
    """
    validate_seed(seed)

    model_seed = derive_seed(
        seed,
        SeedStream.MODEL,
    )

    estimator = LogisticRegression(
        max_iter=1000,
        random_state=model_seed,
    )

    return SklearnClassifier(
        estimator=estimator,
        name="logistic_regression",
    )


def create_random_forest(
    seed: int,
) -> SklearnClassifier:
    """Create the random forest supervised baseline.

    Args:
        seed (int): Root experiment seed. A deterministic model-specific child seed is derived from it.

    Returns:
        SklearnClassifier: Configured random forest benchmark classifier.
    """
    validate_seed(seed)

    model_seed = derive_seed(
        seed,
        SeedStream.MODEL,
    )

    estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=model_seed,
        n_jobs=1,
    )

    return SklearnClassifier(
        estimator=estimator,
        name="random_forest",
    )