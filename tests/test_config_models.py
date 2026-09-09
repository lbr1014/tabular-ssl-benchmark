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
        label_fractions=(0.05, 0.1, 0.2, 0.5),
        seeds=(1, 2, 3),
        test_size=0.2,
    )

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
            label_fractions=(0.1,),
            seeds=(42,),
            test_size=test_size,
        )