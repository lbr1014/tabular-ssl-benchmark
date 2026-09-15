"""Factory utilities for benchmark classifiers.
This module maps stable model identifiers from experiment specifications
to initialized classifiers implementing the benchmark model interface.
"""

from collections.abc import Callable

from models.base import BenchmarkClassifier
from models.sklearn_models import (
    create_logistic_regression,
    create_random_forest,
)


ClassifierFactory = Callable[[int], BenchmarkClassifier]


_CLASSIFIER_FACTORIES: dict[str, ClassifierFactory] = {
    "logistic_regression": create_logistic_regression,
    "random_forest": create_random_forest,
}


def create_classifier(
    model_name: str,
    *,
    seed: int,
) -> BenchmarkClassifier:
    """Create a registered benchmark classifier.

    Args:
        model_name (str): Stable identifier of the classifier to create.
        seed (int): Root experiment seed passed to the classifier factory.

    Returns:
        BenchmarkClassifier: Initialized benchmark classifier.
    """
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")

    if not model_name.strip():
        raise ValueError("model_name must not be empty.")

    try:
        factory = _CLASSIFIER_FACTORIES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark model: {model_name!r}."
        ) from exc

    return factory(seed)