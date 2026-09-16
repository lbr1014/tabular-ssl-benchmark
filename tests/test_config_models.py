"""Tests for typed benchmark configuration models."""

import pytest

from config.models import BenchmarkConfig, DatasetConfig


def test_valid_dataset_config():
    """A valid dataset configuration should be created successfully."""
    config = DatasetConfig(
        name="iris",
        openml_id=61,
    )

    assert config.name == "iris"
    assert config.openml_id == 61
    assert config.enabled is True


@pytest.mark.parametrize("openml_id", [0, -1])
def test_dataset_config_rejects_invalid_openml_id(openml_id):
    """OpenML identifiers must be positive."""
    with pytest.raises(
        ValueError,
        match="openml_id must be positive",
    ):
        DatasetConfig(
            name="iris",
            openml_id=openml_id,
        )


@pytest.mark.parametrize("name", ["", "   "])
def test_dataset_config_rejects_empty_name(name):
    """Dataset names must contain non-whitespace characters."""
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        DatasetConfig(
            name=name,
            openml_id=61,
        )


def test_valid_benchmark_config():
    """A valid benchmark configuration should preserve its values."""
    config = BenchmarkConfig(
        models=("logistic_regression",),
        ssl_methods=("supervised",),
        label_fractions=(0.05, 0.1, 0.2, 0.5),
        seeds=(1, 2, 3),
        test_size=0.2,
    )

    assert config.models == ("logistic_regression",)
    assert config.label_fractions == (0.05, 0.1, 0.2, 0.5)
    assert config.seeds == (1, 2, 3)
    assert config.test_size == 0.2


@pytest.mark.parametrize(
    "label_fraction",
    [0.0, -0.1, 1.1],
)
def test_benchmark_config_rejects_invalid_label_fraction(
    label_fraction,
):
    """Label fractions must belong to the interval (0, 1]."""
    with pytest.raises(
        ValueError,
        match="label_fractions must contain values",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised",),
            label_fractions=(label_fraction,),
            seeds=(42,),
        )


def test_benchmark_config_rejects_duplicate_label_fractions():
    """Duplicate label fractions would generate duplicate experiments."""
    with pytest.raises(
        ValueError,
        match="label_fractions must not contain duplicate",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised",),
            label_fractions=(0.1, 0.1),
            seeds=(42,),
        )


def test_benchmark_config_rejects_duplicate_seeds():
    """Duplicate seeds would generate duplicate experimental runs."""
    with pytest.raises(
        ValueError,
        match="seeds must not contain duplicate",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(42, 42),
        )


def test_benchmark_config_rejects_negative_seed():
    """Experimental root seeds must be non-negative."""
    with pytest.raises(
        ValueError,
        match="seeds must contain non-negative",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(-1,),
        )

@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_benchmark_config_rejects_invalid_test_size(test_size):
    """Test size must belong to the open interval (0, 1)."""
    with pytest.raises(
        ValueError,
        match="test_size must be in the interval",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(42,),
            test_size=test_size,
        )
        
def test_benchmark_config_rejects_empty_models():
    """At least one benchmark model must be configured."""
    with pytest.raises(
        ValueError,
        match="models must contain at least one model",
    ):
        BenchmarkConfig(
            models=(),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(42,),
        )

def test_benchmark_config_rejects_duplicate_models():
    """Duplicate benchmark model identifiers should be rejected."""
    with pytest.raises(
        ValueError,
        match="models must not contain duplicate values",
    ):
        BenchmarkConfig(
            models=(
                "logistic_regression",
                "logistic_regression",
            ),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(42,),
        )

def test_benchmark_config_rejects_non_string_model():
    """Benchmark model identifiers must be strings."""
    with pytest.raises(
        TypeError,
        match="models must contain string values",
    ):
        BenchmarkConfig(
            models=("logistic_regression", 42),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(42,),
        )
        
@pytest.mark.parametrize(
    "model_name",
    ["", "   "],
)
def test_benchmark_config_rejects_empty_model_name(
    model_name,
):
    """Benchmark model identifiers must not be empty."""
    with pytest.raises(
        ValueError,
        match="models must not contain empty names",
    ):
        BenchmarkConfig(
            models=(model_name,),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(42,),
        )
        
def test_benchmark_config_rejects_empty_ssl_methods():
    """At least one SSL method must be configured."""
    with pytest.raises(
        ValueError,
        match="ssl_methods must contain at least one method",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(),
            label_fractions=(0.1,),
            seeds=(42,),
        )


def test_benchmark_config_rejects_duplicate_ssl_methods():
    """Duplicate SSL method identifiers should be rejected."""
    with pytest.raises(
        ValueError,
        match="ssl_methods must not contain duplicate values",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised", "supervised"),
            label_fractions=(0.1,),
            seeds=(42,),
        )


def test_benchmark_config_rejects_non_string_ssl_method():
    """SSL method identifiers must be strings."""
    with pytest.raises(
        TypeError,
        match="ssl_methods must contain string values",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised", 42),
            label_fractions=(0.1,),
            seeds=(42,),
        )


@pytest.mark.parametrize(
    "ssl_method",
    ["", "   "],
)
def test_benchmark_config_rejects_empty_ssl_method_name(
    ssl_method,
):
    """SSL method identifiers must not be empty."""
    with pytest.raises(
        ValueError,
        match="ssl_methods must not contain empty names",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(ssl_method,),
            label_fractions=(0.1,),
            seeds=(42,),
        )