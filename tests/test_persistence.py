"""Persistence tests for benchmark results."""

import csv
from dataclasses import replace
import json

import pytest

from benchmark.experiment import ExperimentConfig, ExperimentResult
from benchmark.persistence import save_benchmark_results, save_run_metadata
from benchmark.metadata import BenchmarkRunMetadata

def _create_result(
    *,
    seed: int = 42,
) -> ExperimentResult:
    """Create a representative experiment result for persistence tests."""
    config = ExperimentConfig(
        dataset_name="iris",
        model_name="logistic_regression",
        ssl_method="supervised",
        label_fraction=0.5,
        seed=seed,
        test_size=0.2,
    )

    return ExperimentResult(
        config=config,
        metrics={
            "accuracy": 0.95,
            "f1_macro": 0.94,
        },
        fit_time=0.12,
        predict_time=0.03,
        n_train=120,
        n_labeled=60,
        n_unlabeled=60,
        n_test=30,
        metadata={
            "dataset_source": "openml",
            "dataset_source_id": 61,
            "dataset_source_version": 1,
        },
    )
    
def _create_run_metadata(
    *,
    git_commit: str | None = "abc123",
) -> BenchmarkRunMetadata:
    """Create representative metadata for persistence tests."""
    return BenchmarkRunMetadata(
        created_at="2026-09-14T10:00:00+00:00",
        python_version="3.12.0",
        platform="test-platform",
        git_commit=git_commit,
        n_experiments=10,
    )
    
def test_save_benchmark_results_creates_output_files(
    tmp_path,
):
    """Benchmark persistence should create JSONL and CSV files."""
    results = (
        _create_result(seed=1),
        _create_result(seed=2),
    )

    jsonl_path, csv_path = save_benchmark_results(
        results,
        tmp_path,
    )

    assert jsonl_path.exists()
    assert csv_path.exists()
    assert jsonl_path.name == "results.jsonl"
    assert csv_path.name == "results.csv"
    
def test_save_benchmark_results_writes_jsonl_records(
    tmp_path,
):
    """JSONL output should contain one record per experiment."""
    results = (
        _create_result(seed=1),
        _create_result(seed=2),
    )

    jsonl_path, _ = save_benchmark_results(
        results,
        tmp_path,
    )

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = [
            json.loads(line)
            for line in file
        ]

    assert len(records) == 2
    assert records[0]["seed"] == 1
    assert records[1]["seed"] == 2
    
def test_save_benchmark_results_writes_csv_records(
    tmp_path,
):
    """CSV output should contain one row per experiment."""
    results = (
        _create_result(seed=1),
        _create_result(seed=2),
    )

    _, csv_path = save_benchmark_results(
        results,
        tmp_path,
    )

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        records = list(
            csv.DictReader(file)
        )

    assert len(records) == 2
    assert records[0]["seed"] == "1"
    assert records[1]["seed"] == "2"
    
def test_save_benchmark_results_rejects_empty_results(
    tmp_path,
):
    """Persistence should reject benchmark runs without results."""
    with pytest.raises(
        ValueError,
        match="At least one experiment result",
    ):
        save_benchmark_results(
            (),
            tmp_path,
        )
        
def test_save_benchmark_results_rejects_inconsistent_schema(
    tmp_path,
):
    """All persisted experiment records should share the same schema."""
    first = _create_result(seed=1)

    second = replace(
        _create_result(seed=2),
        metadata={
            "dataset_source": "openml",
            "dataset_source_id": 61,
            "dataset_source_version": 1,
            "extra_field": "unexpected",
        },
    )

    with pytest.raises(
        ValueError,
        match="same serialized fields",
    ):
        save_benchmark_results(
            (first, second),
            tmp_path,
        )
        
    assert not (tmp_path / "results.jsonl").exists()
    assert not (tmp_path / "results.csv").exists()
        
def test_save_run_metadata_creates_metadata_file(
    tmp_path,
):
    """Run metadata persistence should create a JSON file."""
    metadata = _create_run_metadata()

    metadata_path = save_run_metadata(
        metadata,
        tmp_path,
    )

    assert metadata_path.exists()
    assert metadata_path.name == "metadata.json"
    
    
def test_save_run_metadata_writes_metadata_content(
    tmp_path,
):
    """Persisted metadata should preserve all run information."""
    metadata = _create_run_metadata()

    metadata_path = save_run_metadata(
        metadata,
        tmp_path,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        stored_metadata = json.load(file)

    assert stored_metadata == metadata.to_dict()
    
def test_save_run_metadata_creates_output_directory(
    tmp_path,
):
    """Run metadata persistence should create missing directories."""
    output_dir = (
        tmp_path
        / "results"
        / "benchmark-run"
    )

    metadata = _create_run_metadata()

    metadata_path = save_run_metadata(
        metadata,
        output_dir,
    )

    assert output_dir.exists()
    assert metadata_path.exists()
    
def test_save_run_metadata_supports_missing_git_commit(
    tmp_path,
):
    """Metadata persistence should support unavailable Git information."""
    metadata = _create_run_metadata(
        git_commit=None,
    )

    metadata_path = save_run_metadata(
        metadata,
        tmp_path,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        stored_metadata = json.load(file)

    assert stored_metadata["git_commit"] is None