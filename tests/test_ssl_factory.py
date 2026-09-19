"""Tests for the semi-supervised learning method factory."""

import pytest

from ssl_methods.factory import create_ssl_method
from ssl_methods.supervised import SupervisedMethod
from ssl_methods.self_training import SelfTrainingMethod

def test_create_supervised_method() -> None:
    """Factory should create the supervised reference strategy."""
    method = create_ssl_method("supervised")

    assert isinstance(method, SupervisedMethod)
    assert method.name == "supervised"
    
def test_supervised_method_rejects_parameters() -> None:
    """Supervised reference strategy should reject unused parameters."""
    with pytest.raises(
        ValueError,
        match="does not accept configuration parameters",
    ):
        create_ssl_method(
            "supervised",
            unused_parameter=True,
        )
    
def test_create_self_training_method() -> None:
    """Factory should create a configured self-training strategy."""
    method = create_ssl_method(
        "self_training",
        confidence_threshold=0.90,
        max_iterations=5,
    )

    assert isinstance(method, SelfTrainingMethod)
    assert method.name == "self_training"
    assert method.confidence_threshold == 0.90
    assert method.max_iterations == 5
    
def test_unknown_ssl_method_raises_value_error() -> None:
    """Factory should reject unsupported SSL method identifiers."""
    with pytest.raises(ValueError, match="Unknown SSL method"):
        create_ssl_method("unknown")
        
def test_ssl_factory_rejects_non_string_name() -> None:
    """Factory should require string method identifiers."""
    with pytest.raises(TypeError, match="name must be a string"):
        create_ssl_method(123)


def test_ssl_factory_rejects_empty_name() -> None:
    """Factory should reject empty method identifiers."""
    with pytest.raises(ValueError, match="name must not be empty"):
        create_ssl_method(" ")