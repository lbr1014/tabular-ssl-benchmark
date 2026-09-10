"""Tests for experiment matrix generation."""

import pytest

from config.matrix import (
    ExperimentSpec,
    generate_experiment_matrix,
)
from config.models import BenchmarkConfig, DatasetConfig


@pytest.fixture
def datasets() -> tuple[DatasetConfig, ...]:
    """Return dataset configurations used by matrix tests."""
    return (
        DatasetConfig(
            name="iris",
            openml_id=61,
            enabled=True,
        ),
        DatasetConfig(
            name="adult",
            openml_id=1590,
            enabled=True,
        ),
    )


@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    """Return a small benchmark configuration used by matrix tests."""
    return BenchmarkConfig(
        label_fractions=(0.1, 0.5),
        seeds=(1, 2, 3),
        test_size=0.2,
    )


def test_matrix_has_expected_number_of_experiments(
    datasets,
    benchmark_config,
):
    """Matrix size should equal the Cartesian product dimensions."""
    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    assert len(matrix) == 2 * 2 * 3


def test_matrix_contains_experiment_specs(
    datasets,
    benchmark_config,
):
    """Every generated matrix entry should be an ExperimentSpec."""
    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    assert all(
        isinstance(spec, ExperimentSpec)
        for spec in matrix
    )


def test_matrix_contains_all_combinations(
    datasets,
    benchmark_config,
):
    """The matrix should contain every requested configuration."""
    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    combinations = {
        (
            spec.dataset.name,
            spec.label_fraction,
            spec.seed,
        )
        for spec in matrix
    }

    expected = {
        ("iris", 0.1, 1),
        ("iris", 0.1, 2),
        ("iris", 0.1, 3),
        ("iris", 0.5, 1),
        ("iris", 0.5, 2),
        ("iris", 0.5, 3),
        ("adult", 0.1, 1),
        ("adult", 0.1, 2),
        ("adult", 0.1, 3),
        ("adult", 0.5, 1),
        ("adult", 0.5, 2),
        ("adult", 0.5, 3),
    }

    assert combinations == expected


def test_matrix_excludes_disabled_datasets(
    benchmark_config,
):
    """Disabled datasets should not produce experiment specifications."""
    datasets = (
        DatasetConfig(
            name="iris",
            openml_id=61,
            enabled=True,
        ),
        DatasetConfig(
            name="adult",
            openml_id=1590,
            enabled=False,
        ),
    )

    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    assert len(matrix) == 6

    assert {
        spec.dataset.name
        for spec in matrix
    } == {"iris"}


def test_matrix_preserves_test_size(
    datasets,
    benchmark_config,
):
    """Every specification should use the configured test size."""
    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    assert all(
        spec.test_size == 0.2
        for spec in matrix
    )


def test_matrix_order_is_deterministic(
    datasets,
    benchmark_config,
):
    """Identical configurations should produce identical matrix order."""
    first = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    second = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    assert first == second


def test_spec_id_is_deterministic():
    """Specification IDs should encode their experimental dimensions."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name="iris",
            openml_id=61,
        ),
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    assert spec.spec_id == "iris__lf-0.1__seed-42"


def test_matrix_rejects_no_enabled_datasets(
    benchmark_config,
):
    """At least one dataset must participate in the benchmark."""
    datasets = (
        DatasetConfig(
            name="iris",
            openml_id=61,
            enabled=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="At least one dataset must be enabled",
    ):
        generate_experiment_matrix(
            datasets=datasets,
            benchmark=benchmark_config,
        )
        
def test_matrix_spec_ids_are_unique(
    datasets,
    benchmark_config,
):
    """Every generated specification should have a unique identifier."""
    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark_config,
    )

    spec_ids = [
        spec.spec_id
        for spec in matrix
    ]

    assert len(spec_ids) == len(set(spec_ids))