
"""Tests for executable benchmark configuration validation."""

import pytest

from config.models import BenchmarkConfig, SSLMethodConfig
from config.validation import validate_benchmark_compatibility
from unittest.mock import Mock

def make_benchmark(
    *,
    models=("logistic_regression",),
    ssl_methods=None,
):
    """Build a minimal benchmark configuration for testing."""
    if ssl_methods is None:
        ssl_methods = (SSLMethodConfig(name="supervised"),)

    return BenchmarkConfig(
        models=models,
        ssl_methods=ssl_methods,
        label_fractions=(0.1,),
        seeds=(1,),
        test_size=0.2,
    )


def test_valid_supervised_configuration():
    """Registered supervised experiments should be accepted."""
    benchmark = make_benchmark()

    validate_benchmark_compatibility(benchmark)


def test_valid_self_training_configuration():
    """Self-training with valid parameters should be accepted."""
    benchmark = make_benchmark(
        ssl_methods=(
            SSLMethodConfig(
                name="self_training",
                params={
                    "confidence_threshold": 0.95,
                    "max_iterations": 10,
                },
            ),
        ),
    )

    validate_benchmark_compatibility(benchmark)


def test_unknown_model_is_rejected():
    """Unknown classifiers should fail validation."""
    benchmark = make_benchmark(
        models=("unknown_model",),
    )

    with pytest.raises(
        ValueError,
        match="Invalid benchmark model",
    ):
        validate_benchmark_compatibility(benchmark)


def test_unknown_ssl_method_is_rejected():
    """Unknown SSL methods should fail validation."""
    benchmark = make_benchmark(
        ssl_methods=(
            SSLMethodConfig(name="unknown_method"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Invalid SSL method",
    ):
        validate_benchmark_compatibility(benchmark)


def test_invalid_ssl_parameters_are_rejected():
    """Invalid strategy parameters should fail validation."""
    benchmark = make_benchmark(
        ssl_methods=(
            SSLMethodConfig(
                name="self_training",
                params={"confidence_threshold": 1.5},
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Invalid SSL method",
    ):
        validate_benchmark_compatibility(benchmark)


def test_supervised_rejects_unexpected_parameters():
    """Supervised learning should not accept SSL hyperparameters."""
    benchmark = make_benchmark(
        ssl_methods=(
            SSLMethodConfig(
                name="supervised",
                params={"max_iterations": 10},
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Invalid SSL method",
    ):
        validate_benchmark_compatibility(benchmark)


def test_invalid_benchmark_type_is_rejected():
    """Validation should require a BenchmarkConfig instance."""
    with pytest.raises(TypeError):
        validate_benchmark_compatibility(None)

def test_valid_standalone_configuration(monkeypatch):
    """Standalone methods with matching configuration should be accepted."""
    method = Mock()
    method.name = "standalone_test"
    method.requires_base_model = False
    factory = Mock(return_value=method)

    monkeypatch.setattr(
        "config.validation.create_ssl_method",
        factory,
    )

    benchmark = make_benchmark(
        ssl_methods=(
            SSLMethodConfig(
                name="standalone_test",
                requires_base_model=False,
            ),
        ),
    )

    validate_benchmark_compatibility(benchmark)

    factory.assert_called_once_with("standalone_test")
    method.fit.assert_not_called()

def test_model_dependent_method_rejects_standalone_configuration():
    """Model-dependent methods must not be declared standalone."""
    benchmark = make_benchmark(
        ssl_methods=(
            SSLMethodConfig(
                name="self_training",
                requires_base_model=False,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Configured requires_base_model does not match",
    ):
        validate_benchmark_compatibility(benchmark)

def test_standalone_method_rejects_base_model_configuration(monkeypatch):
    """Standalone methods must not be declared model-dependent."""
    method = Mock()
    method.name = "standalone_test"
    method.requires_base_model = False

    monkeypatch.setattr(
        "config.validation.create_ssl_method",
        Mock(return_value=method),
    )

    benchmark = make_benchmark(
        ssl_methods=(
            SSLMethodConfig(
                name="standalone_test",
                requires_base_model=True,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Configured requires_base_model does not match",
    ):
        validate_benchmark_compatibility(benchmark)

    method.fit.assert_not_called()