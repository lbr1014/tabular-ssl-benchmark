"""Factory for benchmark semi-supervised learning methods."""

from ssl_methods.base import SSLMethod
from ssl_methods.supervised import SupervisedMethod
from ssl_methods.self_training import SelfTrainingMethod


def create_ssl_method(name: str) -> SSLMethod:
    """Create a semi-supervised learning method by its stable identifier.

    Args:
        name: Stable identifier of the learning strategy.

    Returns:
        SSLMethod: Configured learning strategy.
    """
    if name == "supervised":
        return SupervisedMethod()
    
    if name == "self_training":
        return SelfTrainingMethod()

    raise ValueError(f"Unknown SSL method: {name!r}")