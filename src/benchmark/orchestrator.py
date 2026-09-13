"""Benchmark orchestration utilities.
This module coordinates dataset loading, experiment matrix generation,
and benchmark execution while keeping those responsibilities separated
from individual experiment execution.
"""

from config.models import DatasetConfig
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