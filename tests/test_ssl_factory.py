"""Tests for the semi-supervised learning method factory."""

import pytest

from ssl_methods.factory import create_ssl_method
from ssl_methods.supervised import SupervisedMethod


def test_create_supervised_method() -> None:
    """Factory should create the supervised reference strategy."""
    method = create_ssl_method("supervised")

    assert isinstance(method, SupervisedMethod)
    assert method.name == "supervised"
    
def test_unknown_ssl_method_raises_value_error() -> None:
    """Factory should reject unsupported SSL method identifiers."""
    with pytest.raises(ValueError, match="Unknown SSL method"):
        create_ssl_method("unknown")