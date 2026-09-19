"""Integration tests for the benchmark configuration system."""

from pathlib import Path

from config.loader import (
    load_benchmark_config,
    load_dataset_configs,
)
from config.matrix import generate_experiment_matrix


def test_yaml_configuration_generates_expected_matrix(
    tmp_path: Path,
):
    """YAML configuration should generate the complete experiment matrix."""
    datasets_path = tmp_path / "datasets.yaml"
    benchmark_path = tmp_path / "benchmark.yaml"

    datasets_path.write_text(
        """
datasets:
  - name: iris
    openml_id: 61
    enabled: true

  - name: adult
    openml_id: 1590
    enabled: false
""".strip(),
        encoding="utf-8",
    )

    benchmark_path.write_text(
        """
models:
  - logistic_regression
  - random_forest
  
ssl_methods:
  - name: supervised
  
label_fractions:
  - 0.1
  - 0.5

seeds:
  - 1
  - 2

test_size: 0.25
""".strip(),
        encoding="utf-8",
    )

    datasets = load_dataset_configs(datasets_path)
    benchmark = load_benchmark_config(benchmark_path)

    matrix = generate_experiment_matrix(
        datasets=datasets,
        benchmark=benchmark,
    )

    enabled_datasets = sum(
        dataset.enabled
        for dataset in datasets
    )

    expected_size = (
        enabled_datasets
        * len(benchmark.models)
        * len(benchmark.ssl_methods)
        * len(benchmark.label_fractions)
        * len(benchmark.seeds)
    )

    assert len(matrix) == expected_size

    expected_ids = [
        (
            f"iris"
            f"__model-{model_name}"
            f"__ssl-{ssl_methods.name}"
            f"__lf-{label_fraction:g}"
            f"__test-{benchmark.test_size:g}"
            f"__seed-{seed}"
        )
        for model_name in benchmark.models
        for ssl_methods in benchmark.ssl_methods
        for label_fraction in benchmark.label_fractions
        for seed in benchmark.seeds
    ]

    assert [
        spec.spec_id
        for spec in matrix
    ] == expected_ids

    assert all(
        spec.test_size == 0.25
        for spec in matrix
    )
    
    assert {
        spec.model_name
        for spec in matrix
    } == {
        "logistic_regression",
        "random_forest",
    }