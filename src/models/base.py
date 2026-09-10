"""Common classifier interface used by the benchmark.

This module defines the minimal prediction contract that benchmark
classifiers must implement. Concrete model wrappers expose a uniform
interface regardless of the underlying machine-learning library.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


FeatureMatrix = pd.DataFrame | np.ndarray
TargetArray = pd.Series | np.ndarray


class BenchmarkClassifier(ABC):
    """Abstract interface for classifiers evaluated by the benchmark.
    Implementations wrap concrete classification models while exposing
    a common API to the experiment runner.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable identifier of the classifier."""

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """Return class labels in probability-column order."""

    @abstractmethod
    def fit(
        self,
        X: FeatureMatrix,
        y: TargetArray,
    ) -> "BenchmarkClassifier":
        """Fit the classifier.

       Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
            
        Returns:
            BenchmarkClassifier: Fitted classifier instance.
        """

    @abstractmethod
    def predict(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """Predict class labels for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(
        self,
        X: FeatureMatrix,
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted class probabilities of shape (n_samples, n_classes).
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model parameters required for experiment metadata."""