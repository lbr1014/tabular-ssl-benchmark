"""Persistence tests for benchmark results."""

import csv
from dataclasses import replace
import json

import pytest

from benchmark.experiment import ExperimentConfig, ExperimentResult
from benchmark.persistence import save_benchmark_results


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
        
