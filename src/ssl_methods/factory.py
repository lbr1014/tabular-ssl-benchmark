"""Factory for benchmark semi-supervised learning methods."""

from ssl_methods.base import SSLMethod
from ssl_methods.supervised import SupervisedMethod
from ssl_methods.self_training import SelfTrainingMethod


def create_ssl_method(name: str, **params: object) -> SSLMethod:
    """Create a semi-supervised learning method by its stable identifier.

    Args:
        name: Stable identifier of the learning strategy.
        **params: Strategy-specific configuration parameters.

    Returns:
        SSLMethod: Configured semi-supervised learning strategy.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if not name.strip():
        raise ValueError("name must not be empty.")

    if name == "supervised":
        if params:
            raise ValueError(
                "Supervised method does not accept configuration parameters."
            )

        return SupervisedMethod()

    if name == "self_training":
        return SelfTrainingMethod(**params)

    raise ValueError(f"Unknown SSL method: {name!r}")