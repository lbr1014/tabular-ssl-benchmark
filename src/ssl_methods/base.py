"""Common interface for semi-supervised learning methods.

This module defines the contract implemented by semi-supervised learning
strategies evaluated by the benchmark. SSL methods operate on labeled and
unlabeled training data while delegating predictive modeling to a benchmark
classifier.
"""

from abc import ABC, abstractmethod

from models.base import BenchmarkClassifier, FeatureMatrix, TargetArray


class SSLMethod(ABC):
    """Abstract interface for semi-supervised learning strategies.

    An SSL method receives a benchmark classifier together with labeled and
    unlabeled training samples and is responsible for fitting the classifier
    according to its semi-supervised learning strategy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable identifier of the SSL method."""

    @abstractmethod
    def fit(
        self,
        *,
        model: BenchmarkClassifier,
        X_labeled: FeatureMatrix,
        y_labeled: TargetArray,
        X_unlabeled: FeatureMatrix,
    ) -> BenchmarkClassifier:
        """Fit a classifier using labeled and unlabeled training data.

        Args:
            model: Benchmark classifier to train.
            X_labeled: Feature matrix containing labeled training samples.
            y_labeled: Labels associated with the labeled training samples.
            X_unlabeled: Feature matrix containing unlabeled training samples.

        Returns:
            BenchmarkClassifier: Fitted benchmark classifier.
        """
