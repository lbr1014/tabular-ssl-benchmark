"""Tests for the semi-supervised learning method interface."""

import pytest

from ssl_methods.base import SSLMethod
from ssl_methods.factory import create_ssl_method
from ssl_methods.self_training import SelfTrainingMethod


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
        
        @property
        def params(self) -> dict[str, object]:
            return {}
        
        @property
        def requires_base_model(self) -> bool:
            return True

        def fit(
            self,
            *,
            model,
            x_labeled,
            y_labeled,
            x_unlabeled,
        ):
            return model

    method = CompleteSSLMethod()

    assert method.name == "complete"
    assert method.requires_base_model is True
    
def test_ssl_method_without_model_requirement_cannot_be_instantiated() -> None:
    """SSL strategies should declare whether they require a base model."""

    class MissingRequirementSSLMethod(SSLMethod):
        @property
        def name(self) -> str:
            return "missing_requirement"

        @property
        def params(self) -> dict[str, object]:
            return {}

        def fit(
            self,
            *,
            model,
            x_labeled,
            y_labeled,
            x_unlabeled,
        ):
            return model

    with pytest.raises(TypeError):
        MissingRequirementSSLMethod()
            
def test_create_self_training_method() -> None:
    """Factory should create the self-training SSL strategy."""
    method = create_ssl_method("self_training")

    assert isinstance(method, SelfTrainingMethod)
    assert method.name == "self_training"
    
def test_ssl_method_without_params_cannot_be_instantiated() -> None:
    """Verify that subclasses must expose their effective parameters."""

    class MissingParamsSSLMethod(SSLMethod):
        @property
        def name(self) -> str:
            return "missing_params"

        def fit(
            self,
            *,
            model,
            x_labeled,
            y_labeled,
            x_unlabeled,
        ):
            return model

    with pytest.raises(TypeError):
        MissingParamsSSLMethod()