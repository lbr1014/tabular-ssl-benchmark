"""Tests for the benchmark CLI."""

from pathlib import Path

import pytest

from benchmark.cli import build_parser, main
from benchmark.persistence import BenchmarkRunArtifacts


def test_build_parser_parses_required_arguments():
    """CLI parser should parse benchmark input and output paths."""
    parser = build_parser()

    args = parser.parse_args([
        "--datasets-config",
        "config/datasets.yaml",
        "--benchmark-config",
        "config/benchmark.yaml",
        "--output-dir",
        "results/test-run",
    ])

    assert args.datasets_config == Path(
        "config/datasets.yaml"
    )
    assert args.benchmark_config == Path(
        "config/benchmark.yaml"
    )
    assert args.output_dir == Path(
        "results/test-run"
    )
    
def test_build_parser_requires_configuration_arguments():
    """CLI should require benchmark configuration paths."""
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
        
def test_main_runs_and_persists_benchmark(
    monkeypatch,
    tmp_path,
):
    """CLI should load configuration and execute a persistent run."""
    datasets = object()
    benchmark = object()

    datasets_path = tmp_path / "datasets.yaml"
    benchmark_path = tmp_path / "benchmark.yaml"
    output_dir = tmp_path / "results"

    calls = {}

    monkeypatch.setattr(
        "benchmark.cli.load_dataset_configs",
        lambda path: datasets,
    )

    monkeypatch.setattr(
        "benchmark.cli.load_benchmark_config",
        lambda path: benchmark,
    )

    def fake_run_and_save_benchmark(
        *,
        datasets,
        benchmark,
        output_dir,
    ):
        calls["datasets"] = datasets
        calls["benchmark"] = benchmark
        calls["output_dir"] = output_dir

        return BenchmarkRunArtifacts(
            results_jsonl=output_dir / "results.jsonl",
            results_csv=output_dir / "results.csv",
            metadata_json=output_dir / "metadata.json",
            config_json=output_dir / "config.json",
        )

    monkeypatch.setattr(
        "benchmark.cli.run_and_save_benchmark",
        fake_run_and_save_benchmark,
    )

    exit_code = main([
        "--datasets-config",
        str(datasets_path),
        "--benchmark-config",
        str(benchmark_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    assert calls["datasets"] is datasets
    assert calls["benchmark"] is benchmark
    assert calls["output_dir"] == output_dir