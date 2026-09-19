"""Tests for typed benchmark configuration models."""

import pytest

from config.models import BenchmarkConfig, DatasetConfig, SSLMethodConfig

@pytest.fixture
def supervised_ssl_method():
    """Return a valid supervised SSL method configuration."""
    return SSLMethodConfig(name="supervised")

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


def test_valid_benchmark_config(supervised_ssl_method):
    """A valid benchmark configuration should preserve its values."""
    config = BenchmarkConfig(
        models=("logistic_regression",),
        ssl_methods=(supervised_ssl_method,),
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
    supervised_ssl_method,
):
    """Label fractions must belong to the interval (0, 1]."""
    with pytest.raises(
        ValueError,
        match="label_fractions must contain values",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(label_fraction,),
            seeds=(42,),
        )


def test_benchmark_config_rejects_duplicate_label_fractions(supervised_ssl_method):
    """Duplicate label fractions would generate duplicate experiments."""
    with pytest.raises(
        ValueError,
        match="label_fractions must not contain duplicate",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(0.1, 0.1),
            seeds=(42,),
        )


def test_benchmark_config_rejects_duplicate_seeds(supervised_ssl_method):
    """Duplicate seeds would generate duplicate experimental runs."""
    with pytest.raises(
        ValueError,
        match="seeds must not contain duplicate",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(0.1,),
            seeds=(42, 42),
        )


def test_benchmark_config_rejects_negative_seed(supervised_ssl_method):
    """Experimental root seeds must be non-negative."""
    with pytest.raises(
        ValueError,
        match="seeds must contain non-negative",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(0.1,),
            seeds=(-1,),
        )

@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_benchmark_config_rejects_invalid_test_size(test_size, supervised_ssl_method):
    """Test size must belong to the open interval (0, 1)."""
    with pytest.raises(
        ValueError,
        match="test_size must be in the interval",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(0.1,),
            seeds=(42,),
            test_size=test_size,
        )
        
def test_benchmark_config_rejects_empty_models(supervised_ssl_method):
    """At least one benchmark model must be configured."""
    with pytest.raises(
        ValueError,
        match="models must contain at least one model",
    ):
        BenchmarkConfig(
            models=(),
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(0.1,),
            seeds=(42,),
        )

def test_benchmark_config_rejects_duplicate_models(supervised_ssl_method):
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
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(0.1,),
            seeds=(42,),
        )

def test_benchmark_config_rejects_non_string_model(supervised_ssl_method):
    """Benchmark model identifiers must be strings."""
    with pytest.raises(
        TypeError,
        match="models must contain string values",
    ):
        BenchmarkConfig(
            models=("logistic_regression", 42),
            ssl_methods=(supervised_ssl_method,),
            label_fractions=(0.1,),
            seeds=(42,),
        )
        
@pytest.mark.parametrize(
    "model_name",
    ["", "   "],
)
def test_benchmark_config_rejects_empty_model_name(
    model_name,
    supervised_ssl_method,
):
    """Benchmark model identifiers must not be empty."""
    with pytest.raises(
        ValueError,
        match="models must not contain empty names",
    ):
        BenchmarkConfig(
            models=(model_name,),
            ssl_methods=(supervised_ssl_method,),
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


def test_benchmark_config_rejects_invalid_ssl_method_type():
    """Benchmark SSL methods must use typed configurations."""
    with pytest.raises(
        TypeError,
        match="SSLMethodConfig",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=("supervised",),
            label_fractions=(0.1,),
            seeds=(42,),
        )

       
def test_ssl_method_config_accepts_valid_configuration() -> None:
    """SSL method configuration should preserve valid parameters."""
    config = SSLMethodConfig(
        name="self_training",
        params={
            "confidence_threshold": 0.95,
            "max_iterations": 10,
        },
    )

    assert config.name == "self_training"
    assert config.params == {
        "confidence_threshold": 0.95,
        "max_iterations": 10,
    }


def test_ssl_method_config_defaults_to_empty_parameters() -> None:
    """SSL methods without hyperparameters should use an empty mapping."""
    config = SSLMethodConfig(name="supervised")

    assert config.params == {}

def test_ssl_method_config_rejects_non_mapping_parameters() -> None:
    """SSL method parameters must be provided as a mapping."""
    with pytest.raises(TypeError, match="params"):
        SSLMethodConfig(
            name="self_training",
            params=["invalid"],
        )
        
@pytest.mark.parametrize(
    "name",
    ["", " ", "   "],
)
def test_ssl_method_config_rejects_whitespace_name(name):
    """SSL method names must not contain only whitespace."""
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        SSLMethodConfig(name=name)
        
def test_benchmark_config_accepts_ssl_method_configuration() -> None:
    """Benchmark configuration should preserve SSL method parameters."""
    self_training = SSLMethodConfig(
        name="self_training",
        params={
            "confidence_threshold": 0.95,
            "max_iterations": 10,
        },
    )

    config = BenchmarkConfig(
        models=("logistic_regression",),
        ssl_methods=(self_training,),
        label_fractions=(0.1,),
        seeds=(42,),
    )

    assert config.ssl_methods == (self_training,)
    
def test_benchmark_config_rejects_duplicate_ssl_method_names() -> None:
    """Benchmark configuration should reject duplicate SSL strategies."""
    first_method = SSLMethodConfig(
        name="self_training",
        params={"confidence_threshold": 0.95},
    )
    second_method = SSLMethodConfig(
        name="self_training",
        params={"confidence_threshold": 0.80},
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        BenchmarkConfig(
            models=("logistic_regression",),
            ssl_methods=(first_method, second_method),
            label_fractions=(0.1,),
            seeds=(42,),
        )