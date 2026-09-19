"""Self-training strategy for semi-supervised classification.
Implements iterative pseudo-labeling using predictions from
a benchmark classifier. Unlabeled samples whose predicted class
probability exceeds a confidence threshold are added to the labeled
training set and used in subsequent training iterations.
"""

from dataclasses import dataclass

import numpy as np

from models.base import BenchmarkClassifier, FeatureMatrix, TargetArray
from ssl_methods.base import SSLMethod


@dataclass(frozen=True)
class SelfTrainingMethod(SSLMethod):
    """Iterative confidence-based self-training strategy.

    Args:
        confidence_threshold: Minimum predicted class probability required
            to assign a pseudo-label to an unlabeled sample.
        max_iterations: Maximum number of pseudo-labeling iterations.
    """

    confidence_threshold: float = 0.95
    max_iterations: int = 10

    @property
    def name(self) -> str:
        """Return the stable identifier of the learning strategy."""
        return "self_training"
    
    @property
    def params(self) -> dict[str, object]:
        """Return the effective self-training configuration."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "max_iterations": self.max_iterations,
        }
        
    @property
    def requires_base_model(self) -> bool:
        """Return whether the strategy requires a benchmark classifier."""
        return True
    
    def __post_init__(self) -> None:
        """Validate self-training hyperparameters."""
        if (
            isinstance(self.confidence_threshold, bool)
            or not isinstance(self.confidence_threshold, (int, float))
        ):
            raise TypeError(
                "confidence_threshold must be numeric."
            )

        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be in (0, 1]."
            )

        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
        ):
            raise TypeError(
                "max_iterations must be an integer."
            )

        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )
            
    def fit(
        self,
        *,
        model: BenchmarkClassifier,
        x_labeled: FeatureMatrix,
        y_labeled: TargetArray,
        x_unlabeled: FeatureMatrix,
    ) -> BenchmarkClassifier:
        """Fit a classifier using iterative confidence-based pseudo-labeling.
        
        Args:
        model: Benchmark classifier to train.
        x_labeled: Feature matrix containing labeled training samples.
        y_labeled: Ground-truth labels for the labeled training samples.
        x_unlabeled: Feature matrix containing unlabeled training samples.

    Returns:
        BenchmarkClassifier: The fitted classifier instance.
    """

        current_x = np.asarray(x_labeled).copy()
        current_y = np.asarray(y_labeled).copy()
        remaining_x = np.asarray(x_unlabeled).copy()

        model.fit(current_x, current_y)

        for _ in range(self.max_iterations):
            if len(remaining_x) == 0:
                break

            probabilities = model.predict_proba(remaining_x)

            confidence = probabilities.max(axis=1)
            selected = confidence >= self.confidence_threshold

            if not np.any(selected):
                break

            predicted_indices = probabilities[selected].argmax(axis=1)
            pseudo_labels = model.classes_[predicted_indices]

            current_x = np.concatenate(
                (current_x, remaining_x[selected]),
                axis=0,
            )
            current_y = np.concatenate(
                (current_y, pseudo_labels),
                axis=0,
            )

            remaining_x = remaining_x[~selected]

            model.fit(current_x, current_y)

        return model