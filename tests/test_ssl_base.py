"""Tests for the semi-supervised learning method interface."""

import pytest

from ssl_methods.base import SSLMethod


def test_ssl_method_cannot_be_instantiated() -> None:
    """Verify that the abstract SSL interface cannot be instantiated."""
    with pytest.raises(TypeError):
        SSLMethod()
        
def test_incomplete_ssl_method_cannot_be_instantiated() -> None:
    """Verify that subclasses must implement the complete SSL contract."""

    class IncompleteSSLMethod(SSLMethod):
        @property
        def name(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError):
        IncompleteSSLMethod()
        
def test_complete_ssl_method_can_be_instantiated() -> None:
    """Verify that a subclass implementing the contract can be instantiated."""

    class CompleteSSLMethod(SSLMethod):
        @property
        def name(self) -> str:
            return "complete"

        def fit(
            self,
            *,
            model,
            X_labeled,
            y_labeled,
            X_unlabeled,
        ):
            return model

    method = CompleteSSLMethod()

    assert method.name == "complete"