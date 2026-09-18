"""Supervised benchmark strategy used as the reference condition for evaluating semi-supervised 
learning methods. The strategy trains exclusively on labeled samples and deliberately
ignores the unlabeled training partition.
"""

from models.base import BenchmarkClassifier, FeatureMatrix, TargetArray
from ssl_methods.base import SSLMethod


class SupervisedMethod(SSLMethod):
    """Train a benchmark classifier using labeled samples only."""

    @property
    def name(self) -> str:
        """Return the stable identifier of the supervised strategy.
        
        Returns:
            str: Stable identifier of the supervised strategy.
        """
        return "supervised"
    
    @property
    def params(self) -> dict[str, object]:
        """Return the effective supervised strategy configuration."""
        return {}
    
    @property
    def requires_base_model(self) -> bool:
        return True

    def fit(
        self,
        *,
        model: BenchmarkClassifier,
        x_labeled: FeatureMatrix,
        y_labeled: TargetArray,
        x_unlabeled: FeatureMatrix,
    ) -> BenchmarkClassifier:
        """Fit the classifier exclusively on labeled training samples.

        Args:
            model: Benchmark classifier to train.
            x_labeled: Feature matrix containing labeled training samples.
            y_labeled: Labels associated with the labeled training samples.
            x_unlabeled: Unlabeled training samples, intentionally ignored
                by the supervised reference strategy.

        Returns:
            BenchmarkClassifier: Fitted benchmark classifier.
        """
        del x_unlabeled

        return model.fit(
            x_labeled,
            y_labeled,
        )