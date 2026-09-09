"""Basic tests for YAML configuration loading."""

from pathlib import Path

from config.loader import (
    load_benchmark_config,
    load_dataset_configs,
)


def test_load_dataset_configs(tmp_path: Path):
    """A valid dataset YAML file should produce typed configurations."""
    config_path = tmp_path / "datasets.yaml"

    config_path.write_text(
        """
datasets:
  - name: iris
    openml_id: 61
    enabled: true
  - name: example
    openml_id: 123
    enabled: false
""".strip(),
        encoding="utf-8",
    )

    datasets = load_dataset_configs(config_path)

    assert len(datasets) == 2

    assert datasets[0].name == "iris"
    assert datasets[0].openml_id == 61
    assert datasets[0].enabled is True

    assert datasets[1].name == "example"
    assert datasets[1].openml_id == 123
    assert datasets[1].enabled is False


def test_load_benchmark_config(tmp_path: Path):
    """A valid benchmark YAML file should produce a typed configuration."""
    config_path = tmp_path / "benchmark.yaml"

    config_path.write_text(
        """
label_fractions:
  - 0.05
  - 0.10
  - 0.20

seeds:
  - 1
  - 2
  - 3

test_size: 0.25
""".strip(),
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)

    assert config.label_fractions == (
        0.05,
        0.10,
        0.20,
    )
    assert config.seeds == (1, 2, 3)
    assert config.test_size == 0.25


def test_benchmark_config_uses_default_test_size(
    tmp_path: Path,
):
    """The benchmark loader should use the model default test size."""
    config_path = tmp_path / "benchmark.yaml"

    config_path.write_text(
        """
label_fractions:
  - 0.10

seeds:
  - 42
""".strip(),
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)

    assert config.test_size == 0.2