"""Basic tests for YAML configuration loading."""

from pathlib import Path

from config.loader import (
    load_benchmark_config,
    load_dataset_configs,
)

import pytest

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
    
def test_loader_rejects_missing_file(tmp_path: Path):
    """Loading a non-existing configuration file should fail clearly."""
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(
        FileNotFoundError,
        match="Configuration file not found",
    ):
        load_benchmark_config(config_path)


def test_loader_rejects_empty_yaml(tmp_path: Path):
    """Empty YAML configuration files should be rejected."""
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Configuration file is empty",
    ):
        load_benchmark_config(config_path)


def test_loader_rejects_non_mapping_root(tmp_path: Path):
    """The root YAML element must be a mapping."""
    config_path = tmp_path / "benchmark.yaml"

    config_path.write_text(
        """
- label_fractions
- seeds
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="root YAML element must be a mapping",
    ):
        load_benchmark_config(config_path)


def test_loader_rejects_malformed_yaml(tmp_path: Path):
    """Malformed YAML should produce a configuration error."""
    config_path = tmp_path / "benchmark.yaml"

    config_path.write_text(
        """
label_fractions:
  - 0.1
seeds:
  - 1
invalid: [
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid YAML configuration",
    ):
        load_benchmark_config(config_path)


def test_benchmark_loader_rejects_missing_required_key(
    tmp_path: Path,
):
    """Required benchmark configuration keys must be present."""
    config_path = tmp_path / "benchmark.yaml"

    config_path.write_text(
        """
label_fractions:
  - 0.1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing required keys",
    ):
        load_benchmark_config(config_path)


def test_benchmark_loader_rejects_unknown_key(
    tmp_path: Path,
):
    """Unknown benchmark keys should fail instead of being ignored."""
    config_path = tmp_path / "benchmark.yaml"

    config_path.write_text(
        """
label_fractions:
  - 0.1

seeds:
  - 42

unknown_setting: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unknown keys",
    ):
        load_benchmark_config(config_path)


def test_dataset_loader_rejects_duplicate_names(
    tmp_path: Path,
):
    """Dataset names must uniquely identify catalogue entries."""
    config_path = tmp_path / "datasets.yaml"

    config_path.write_text(
        """
datasets:
  - name: iris
    openml_id: 61

  - name: iris
    openml_id: 123
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Dataset names must be unique",
    ):
        load_dataset_configs(config_path)


def test_dataset_loader_rejects_unknown_key(
    tmp_path: Path,
):
    """Typos in dataset configuration keys must not be ignored."""
    config_path = tmp_path / "datasets.yaml"

    config_path.write_text(
        """
datasets:
  - name: iris
    openml_id: 61
    enabeld: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unknown keys",
    ):
        load_dataset_configs(config_path)