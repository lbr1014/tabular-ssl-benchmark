"""YAML configuration loading utilities.

This module converts human-editable YAML configuration files into the
typed configuration models used internally by the benchmark.
"""

from pathlib import Path
from typing import Any

import yaml

from config.models import BenchmarkConfig, DatasetConfig


def load_dataset_configs(
    path: str | Path,
) -> tuple[DatasetConfig, ...]:
    """Load dataset configurations from a YAML file.

    Args:
        path (str | Path): Path to the dataset YAML configuration file.
        
    Returns:
        tuple[DatasetConfig, ...]: Validated dataset configurations.
    """
    raw_config = _load_yaml_mapping(path)

    _validate_keys(
        raw_config,
        required={"datasets"},
        allowed={"datasets"},
        context="dataset configuration",
    )

    raw_datasets = raw_config["datasets"]

    if not isinstance(raw_datasets, list):
        raise ValueError("'datasets' must be a list.")

    if not raw_datasets:
        raise ValueError(
            "'datasets' must contain at least one dataset."
        )

    datasets = tuple(
        _parse_dataset_config(item, index=index)
        for index, item in enumerate(raw_datasets)
    )

    names = [dataset.name for dataset in datasets]

    if len(names) != len(set(names)):
        raise ValueError(
            "Dataset names must be unique."
        )

    return datasets


def load_benchmark_config(
    path: str | Path,
) -> BenchmarkConfig:
    """Load benchmark-wide experimental settings from YAML.

    Args:
        path (str | Path): Path to the benchmark YAML configuration file.

    Returns:
        BenchmarkConfig: Validated benchmark configuration.
    """
    raw_config = _load_yaml_mapping(path)

    required = {
        "label_fractions",
        "seeds",
    }

    allowed = required | {"test_size"}

    _validate_keys(
        raw_config,
        required=required,
        allowed=allowed,
        context="benchmark configuration",
    )

    label_fractions = raw_config["label_fractions"]
    seeds = raw_config["seeds"]

    if not isinstance(label_fractions, list):
        raise ValueError(
            "'label_fractions' must be a list."
        )

    if not isinstance(seeds, list):
        raise ValueError(
            "'seeds' must be a list."
        )

    return BenchmarkConfig(
        label_fractions=tuple(label_fractions),
        seeds=tuple(seeds),
        test_size=raw_config.get("test_size", 0.2),
    )


def _load_yaml_mapping(
    path: str | Path,
) -> dict[str, Any]:
    """Read a YAML file whose root element must be a mapping.

    Args:
        path (str | Path): Path to the YAML configuration file.
        
    Returns:
        dict[str, Any]: Parsed YAML mapping.
    """
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration: {config_path}"
        ) from exc

    if data is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            "The root YAML element must be a mapping."
        )

    return data


def _parse_dataset_config(
    raw_dataset: Any,
    *,
    index: int,
) -> DatasetConfig:
    """Convert one raw dataset entry into ``DatasetConfig``.

    Args:
        raw_dataset (Any): Dataset entry obtained from the YAML document.
        index (int): Zero-based position of the entry, used in validation messages.

    Returns:
        DatasetConfig: Validated dataset configuration.
    """
    
    if not isinstance(raw_dataset, dict):
        raise ValueError(
            f"Dataset entry at index {index} must be a mapping."
        )

    required = {
        "name",
        "openml_id",
    }

    allowed = required | {"enabled"}

    _validate_keys(
        raw_dataset,
        required=required,
        allowed=allowed,
        context=f"dataset entry at index {index}",
    )

    return DatasetConfig(
        name=raw_dataset["name"],
        openml_id=raw_dataset["openml_id"],
        enabled=raw_dataset.get("enabled", True),
    )


def _validate_keys(
    mapping: dict[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    context: str,
) -> None:
    """Validate required and allowed keys in a configuration mapping.

    Args:
        mapping (dict[str, Any]): Configuration mapping to validate.
        required (set[str]): Keys that must be present.
        allowed (set[str]): Keys that are permitted.
        context (str): Description of the mapping's context for error messages.
    """
    keys = set(mapping)

    missing = required - keys

    if missing:
        raise ValueError(
            f"Missing required keys in {context}: "
            f"{sorted(missing)}."
        )

    unknown = keys - allowed

    if unknown:
        raise ValueError(
            f"Unknown keys in {context}: "
            f"{sorted(unknown)}."
        )