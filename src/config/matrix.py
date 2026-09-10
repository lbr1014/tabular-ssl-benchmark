"""Experiment matrix generation utilities.

This module expands benchmark-wide configuration values into concrete
dataset-level experimental specifications. The generated specifications describe dataset, label fraction, random
seed, and test split combinations.
"""

from dataclasses import dataclass
from itertools import product

from config.models import BenchmarkConfig, DatasetConfig


@dataclass(frozen=True)
class ExperimentSpec:
    """Describe one dataset-level experimental specification."""

    dataset: DatasetConfig
    label_fraction: float
    seed: int
    test_size: float

    @property
    def spec_id(self) -> str:
        """Return a deterministic human-readable specification ID.
        
        Returns:
            str: Deterministic human-readable specification ID.
        """
        return (
            f"{self.dataset.name}"
            f"__lf-{self.label_fraction:g}"
            f"__seed-{self.seed}"
        )


def generate_experiment_matrix(
    datasets: tuple[DatasetConfig, ...],
    benchmark: BenchmarkConfig,
) -> tuple[ExperimentSpec, ...]:
    """Generate all enabled dataset-level experiment combinations.

    Args:
        datasets (tuple[DatasetConfig, ...]): Dataset configurations.
        benchmark (BenchmarkConfig): Benchmark configuration.
        
    Returns:
        tuple[ExperimentSpec, ...]: All enabled dataset-level experiment combinations.
    """
    
    _validate_matrix_inputs(
        datasets=datasets,
        benchmark=benchmark,
    )

    enabled_datasets = tuple(
        dataset
        for dataset in datasets
        if dataset.enabled
    )

    if not enabled_datasets:
        raise ValueError(
            "At least one dataset must be enabled."
        )

    combinations = product(
        enabled_datasets,
        benchmark.label_fractions,
        benchmark.seeds,
    )

    return tuple(
        ExperimentSpec(
            dataset=dataset,
            label_fraction=label_fraction,
            seed=seed,
            test_size=benchmark.test_size,
        )
        for dataset, label_fraction, seed in combinations
    )


def _validate_matrix_inputs(
    *,
    datasets: tuple[DatasetConfig, ...],
    benchmark: BenchmarkConfig,
) -> None:
    """Validate inputs used to generate the experiment matrix.

    Args:
        datasets (tuple[DatasetConfig, ...]): Dataset configurations to validate.
        benchmark (BenchmarkConfig): Benchmark configuration to validate.
    """
    
    if not isinstance(datasets, tuple):
        raise TypeError("datasets must be a tuple.")

    if not all(
        isinstance(dataset, DatasetConfig)
        for dataset in datasets
    ):
        raise TypeError(
            "datasets must contain only DatasetConfig instances."
        )

    if not isinstance(benchmark, BenchmarkConfig):
        raise TypeError(
            "benchmark must be a BenchmarkConfig instance."
        )