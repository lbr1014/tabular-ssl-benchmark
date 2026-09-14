"""Benchmark orchestration utilities.
This module coordinates dataset loading, experiment matrix generation,
and benchmark execution while keeping those responsibilities separated
from individual experiment execution.
"""

from benchmark.experiment import ExperimentResult
from benchmark.runner import run_benchmark_matrix
from config.matrix import generate_experiment_matrix
from config.models import BenchmarkConfig, DatasetConfig
from datasets.dataset import TabularDataset
from datasets.openml_loader import load_openml_dataset


def _load_enabled_datasets(
    dataset_configs: tuple[DatasetConfig, ...],
) -> dict[str, TabularDataset]:
    """Load all datasets enabled for benchmark execution.
    Each enabled dataset is loaded exactly once and stored by its
    configured benchmark name.

    Args:
        dataset_configs (tuple[DatasetConfig, ...]): Dataset
            configurations available to the benchmark.

    Returns:
        dict[str, TabularDataset]: Loaded datasets keyed by their
        configured dataset names.
    """
    loaded_datasets = {}

    for config in dataset_configs:
        if not config.enabled:
            continue

        dataset = load_openml_dataset(
            config.openml_id,
        )

        loaded_datasets[config.name] = dataset

    return loaded_datasets

def run_benchmark(
    *,
    datasets: tuple[DatasetConfig, ...],
    benchmark: BenchmarkConfig,
) -> tuple[ExperimentResult, ...]:
    """Execute a complete benchmark from validated configuration.
    
    Args:
        datasets (tuple[DatasetConfig, ...]): Dataset configurations
            available to the benchmark.
        benchmark (BenchmarkConfig): Benchmark-wide configuration
            defining models, label fractions, seeds, and test size.

    Returns:
        tuple[ExperimentResult, ...]: Results produced by all benchmark
        experiments in deterministic matrix order.
    """
    loaded_datasets = _load_enabled_datasets(
        datasets,
    )

    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark,
    )

    return run_benchmark_matrix(
        matrix=matrix,
        datasets=loaded_datasets,
    )