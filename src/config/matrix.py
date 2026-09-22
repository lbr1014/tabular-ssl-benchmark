"""Experiment matrix generation utilities.

This module expands benchmark-wide configuration values into concrete
dataset-level experimental specifications. The generated specifications describe dataset, 
model, SSL method, label fraction, random seed, and test split combinations.
"""

from dataclasses import dataclass
from itertools import product

from config.models import BenchmarkConfig, DatasetConfig, SSLMethodConfig


@dataclass(frozen=True)
class ExperimentSpec:
    """Describe one dataset-level experimental specification."""

    dataset: DatasetConfig
    model_name: str | None
    ssl_method: SSLMethodConfig
    label_fraction: float
    seed: int
    test_size: float

    @property
    def spec_id(self) -> str:
        """Return a deterministic human-readable specification ID.
        
        Returns:
            str: Deterministic human-readable specification ID.
        """
        model_id = self.model_name or "standalone"
        
        return (
            f"{self.dataset.name}"
            f"__model-{model_id}"
            f"__ssl-{self.ssl_method.name}"
            f"__lf-{self.label_fraction:g}"
            f"__test-{self.test_size:g}"
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

    specs: list[ExperimentSpec] = []

    for dataset in enabled_datasets:
        for ssl_method in benchmark.ssl_methods:
            model_names: tuple[str | None, ...]

            if ssl_method.requires_base_model:
                model_names = benchmark.models
            else:
                model_names = (None,)

            for model_name, label_fraction, seed in product(
                model_names,
                benchmark.label_fractions,
                benchmark.seeds,
            ):
                specs.append(
                    ExperimentSpec(
                        dataset=dataset,
                        model_name=model_name,
                        ssl_method=ssl_method,
                        label_fraction=label_fraction,
                        seed=seed,
                        test_size=benchmark.test_size,
                    )
                )

    return tuple(specs)

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