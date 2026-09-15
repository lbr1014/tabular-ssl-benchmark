"""Persistence utilities for benchmark experiment results."""

import csv
import json
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass

from benchmark.experiment import ExperimentResult
from benchmark.serialization import serialize_experiment_result
from benchmark.metadata import BenchmarkRunMetadata

from config.models import BenchmarkConfig, DatasetConfig
from config.serialization import serialize_benchmark_config

@dataclass(frozen=True)
class BenchmarkRunArtifacts:
    """File artifacts produced by a persisted benchmark run."""

    results_jsonl: Path
    results_csv: Path
    metadata_json: Path
    config_json: Path

def save_benchmark_results(
    results: Iterable[ExperimentResult],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist benchmark results in JSONL and CSV formats.
    Results are serialized into flat records and written both as JSON
    Lines for machine-readable storage and CSV for tabular analysis.

    Args:
        results (Iterable[ExperimentResult]): Experiment results to persist.
        output_dir (str | Path): Directory where result files are written.

    Returns:
        tuple[Path, Path]: Paths to the generated JSONL and CSV files.
    """
    records = [
        serialize_experiment_result(result)
        for result in results
    ]

    if not records:
        raise ValueError(
            "At least one experiment result is required."
        )
        
    _validate_record_schema(records)
    
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )
        
    jsonl_path = output_path / "results.jsonl"
    csv_path = output_path / "results.csv"

    _write_jsonl(
        records,
        jsonl_path,
    )

    _write_csv(
        records,
        csv_path,
    )

    return jsonl_path, csv_path
    
def _write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write serialized experiment records in JSON Lines format.
    
    Args:
        records (list[dict[str, Any]]): Serialized experiment records.
        path (Path): File path to write the JSONL data.
    """
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")
            
def _write_csv(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write serialized experiment records in CSV format.
    
    Args:
        records (list[dict[str, Any]]): Serialized experiment records.
        path (Path): File path to write the CSV data.
    """
    fieldnames = list(records[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)
        
def _validate_record_schema(
    records: list[dict[str, Any]],
) -> None:
    """Validate that all serialized records share the same fields.
    
    Args:
        records (list[dict[str, Any]]): Serialized experiment records.
    """
    expected_fields = set(records[0])

    for index, record in enumerate(
        records[1:],
        start=1,
    ):
        if set(record) != expected_fields:
            raise ValueError(
                "All experiment results must share the same "
                f"serialized fields; mismatch at index {index}."
            )
            
def save_run_metadata(
    metadata: BenchmarkRunMetadata,
    output_dir: str | Path,
) -> Path:
    """Persist benchmark run metadata as JSON.

    Args:
        metadata (BenchmarkRunMetadata): Metadata describing the
            benchmark execution.
        output_dir (str | Path): Directory where the metadata file
            is written.

    Returns:
        Path: Path to the generated metadata JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = output_path / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return metadata_path

def save_benchmark_config(
    *,
    datasets: tuple[DatasetConfig, ...],
    benchmark: BenchmarkConfig,
    output_dir: str | Path,
) -> Path:
    """Persist the effective benchmark configuration as JSON.

    Args:
        datasets (tuple[DatasetConfig, ...]): Dataset configurations
            used by the benchmark.
        benchmark (BenchmarkConfig): Benchmark-wide configuration.
        output_dir (str | Path): Directory where the configuration
            file is written.

    Returns:
        Path: Path to the generated configuration JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = output_path / "config.json"

    config = serialize_benchmark_config(
        datasets=datasets,
        benchmark=benchmark,
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return config_path

def save_benchmark_run(
    *,
    results: tuple[ExperimentResult, ...],
    datasets: tuple[DatasetConfig, ...],
    benchmark: BenchmarkConfig,
    metadata: BenchmarkRunMetadata,
    output_dir: str | Path,
) -> BenchmarkRunArtifacts:
    """Persist all artifacts associated with a benchmark run.

    Args:
        results (tuple[ExperimentResult, ...]): Experiment results
            produced by the benchmark.
        datasets (tuple[DatasetConfig, ...]): Dataset configurations
            used by the benchmark.
        benchmark (BenchmarkConfig): Effective benchmark configuration.
        metadata (BenchmarkRunMetadata): Metadata describing the run.
        output_dir (str | Path): Directory where run artifacts are stored.

    Returns:
        BenchmarkRunArtifacts: Paths to all persisted run artifacts.
    """
    if metadata.n_experiments != len(results):
        raise ValueError(
            "Run metadata experiment count does not match "
            "the number of benchmark results."
        )
        
    jsonl_path, csv_path = save_benchmark_results(
        results,
        output_dir,
    )

    metadata_path = save_run_metadata(
        metadata,
        output_dir,
    )

    config_path = save_benchmark_config(
        datasets=datasets,
        benchmark=benchmark,
        output_dir=output_dir,
    )

    return BenchmarkRunArtifacts(
        results_jsonl=jsonl_path,
        results_csv=csv_path,
        metadata_json=metadata_path,
        config_json=config_path,
    )