"""Orchestration tests for benchmark execution.
This module tests the orchestration of benchmark execution, 
including dataset loading, experiment generation, and result collection."""

import json

from benchmark.orchestrator import _load_enabled_datasets, run_and_save_benchmark, run_benchmark
from config.models import BenchmarkConfig, DatasetConfig


def test_load_enabled_datasets_skips_disabled_datasets(
    monkeypatch,
    mixed_dataset,
):
    """Disabled datasets should not be loaded."""
    requested_ids = []

    def fake_loader(data_id):
        requested_ids.append(data_id)
        return mixed_dataset

    monkeypatch.setattr(
        "benchmark.orchestrator.load_openml_dataset",
        fake_loader,
    )

    configs = (
        DatasetConfig(
            name=mixed_dataset.name,
            openml_id=61,
            enabled=True,
        ),
        DatasetConfig(
            name="disabled_dataset",
            openml_id=999,
            enabled=False,
        ),
    )

    loaded = _load_enabled_datasets(configs)

    assert requested_ids == [61]
    assert set(loaded) == {
        mixed_dataset.name,
    }
    
def test_run_benchmark_executes_complete_matrix(
    monkeypatch,
    mixed_dataset,
):
    """A benchmark should execute every generated experiment."""
    monkeypatch.setattr(
        "benchmark.orchestrator.load_openml_dataset",
        lambda data_id: mixed_dataset,
    )

    datasets = (
        DatasetConfig(
            name=mixed_dataset.name,
            openml_id=61,
            enabled=True,
        ),
    )

    benchmark = BenchmarkConfig(
        models=(
            "logistic_regression",
            "random_forest",
        ),
        label_fractions=(
            0.5,
            1.0,
        ),
        seeds=(
            1,
            2,
        ),
        test_size=0.2,
    )

    results = run_benchmark(
        datasets=datasets,
        benchmark=benchmark,
    )

    expected_count = (
        len(benchmark.models)
        * len(benchmark.label_fractions)
        * len(benchmark.seeds)
    )

    assert len(results) == expected_count
    
    actual_combinations = {
        (
            result.config.model_name,
            result.config.label_fraction,
            result.config.seed,
        )
        for result in results
    }

    expected_combinations = {
        (
            model_name,
            label_fraction,
            seed,
        )
        for model_name in benchmark.models
        for label_fraction in benchmark.label_fractions
        for seed in benchmark.seeds
    }

    assert actual_combinations == expected_combinations
    
def test_run_benchmark_loads_each_dataset_once(
    monkeypatch,
    mixed_dataset,
):
    """A dataset should be loaded once regardless of experiment count."""
    requested_ids = []

    def fake_loader(data_id):
        requested_ids.append(data_id)
        return mixed_dataset

    monkeypatch.setattr(
        "benchmark.orchestrator.load_openml_dataset",
        fake_loader,
    )

    datasets = (
        DatasetConfig(
            name=mixed_dataset.name,
            openml_id=61,
            enabled=True,
        ),
    )

    benchmark = BenchmarkConfig(
        models=(
            "logistic_regression",
            "random_forest",
        ),
        label_fractions=(
            0.1,
            0.5,
            1.0,
        ),
        seeds=(
            1,
            2,
            3,
        ),
        test_size=0.2,
    )

    results = run_benchmark(
        datasets=datasets,
        benchmark=benchmark,
    )

    assert len(results) == 18
    assert requested_ids == [61]
    
def test_run_benchmark_is_reproducible(
    monkeypatch,
    mixed_dataset,
):
    """Equal benchmark configurations should reproduce metric results."""
    monkeypatch.setattr(
        "benchmark.orchestrator.load_openml_dataset",
        lambda data_id: mixed_dataset,
    )

    datasets = (
        DatasetConfig(
            name=mixed_dataset.name,
            openml_id=61,
        ),
    )

    benchmark = BenchmarkConfig(
        models=("random_forest",),
        label_fractions=(0.5,),
        seeds=(42,),
        test_size=0.2,
    )

    first = run_benchmark(
        datasets=datasets,
        benchmark=benchmark,
    )

    second = run_benchmark(
        datasets=datasets,
        benchmark=benchmark,
    )

    assert len(first) == len(second)

    for first_result, second_result in zip(
        first,
        second,
        strict=True,
    ):
        assert first_result.config == second_result.config
        assert first_result.metrics == second_result.metrics
        
def test_run_and_save_benchmark_persists_complete_run(
    monkeypatch,
    mixed_dataset,
    tmp_path,
):
    """A complete benchmark run should persist all reproducibility artifacts."""
    monkeypatch.setattr(
        "benchmark.orchestrator.load_openml_dataset",
        lambda data_id: mixed_dataset,
    )

    datasets = (
        DatasetConfig(
            name=mixed_dataset.name,
            openml_id=61,
            enabled=True,
        ),
    )

    benchmark = BenchmarkConfig(
        models=("logistic_regression",),
        label_fractions=(0.5,),
        seeds=(42,),
        test_size=0.2,
    )

    artifacts = run_and_save_benchmark(
        datasets=datasets,
        benchmark=benchmark,
        output_dir=tmp_path,
    )

    assert artifacts.results_jsonl.exists()
    assert artifacts.results_csv.exists()
    assert artifacts.metadata_json.exists()
    assert artifacts.config_json.exists()
    
    with artifacts.metadata_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["n_experiments"] == 1
    
    with artifacts.config_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        stored_config = json.load(file)

    assert stored_config["benchmark"]["test_size"] == 0.2
    assert stored_config["benchmark"]["seeds"] == [42]
    assert stored_config["benchmark"]["models"] == [
        "logistic_regression"
    ]
    
    with artifacts.results_jsonl.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = [
            json.loads(line)
            for line in file
        ]

    assert len(records) == 1
    assert records[0]["model_name"] == "logistic_regression"
    assert records[0]["seed"] == 42
    assert records[0]["label_fraction"] == 0.5
    assert records[0]["test_size"] == 0.2