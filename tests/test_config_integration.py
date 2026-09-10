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

    assert len(matrix) == 4

    assert [spec.spec_id for spec in matrix] == [
        "iris__lf-0.1__seed-1",
        "iris__lf-0.1__seed-2",
        "iris__lf-0.5__seed-1",
        "iris__lf-0.5__seed-2",
    ]

    assert all(
        spec.test_size == 0.25
        for spec in matrix
    )