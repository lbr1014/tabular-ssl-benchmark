"""Serialization utilities for validated benchmark configuration."""

from typing import Any

from config.models import BenchmarkConfig, DatasetConfig


def serialize_benchmark_config(
    *,
    datasets: tuple[DatasetConfig, ...],
    benchmark: BenchmarkConfig,
) -> dict[str, Any]:
    """Serialize the effective benchmark configuration.
    The serialized representation captures the validated configuration
    actually used by the benchmark, including resolved default values.

    Args:
        datasets (tuple[DatasetConfig, ...]): Dataset configurations.
        benchmark (BenchmarkConfig): Benchmark-wide configuration.

    Returns:
        dict[str, Any]: JSON-serializable effective configuration.
    """
    return {
        "datasets": [
            {
                "name": dataset.name,
                "openml_id": dataset.openml_id,
                "enabled": dataset.enabled,
            }
            for dataset in datasets
        ],
        "benchmark": {
            "models": list(benchmark.models),
            "ssl_methods": list(benchmark.ssl_methods),
            "label_fractions": list(
                benchmark.label_fractions
            ),
            "seeds": list(benchmark.seeds),
            "test_size": benchmark.test_size,
        },
    }
    
