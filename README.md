# Tabular SSL Benchmark

> Reproducible benchmarking framework for classical and modern semi-supervised learning methods on tabular data, with a particular focus on TabPFN.

## Overview

**Tabular SSL Benchmark** is a reproducible experimental framework for evaluating supervised and semi-supervised learning methods on tabular classification problems.

The project is designed to provide a controlled environment for comparing classical machine learning approaches with modern tabular models, with a particular interest in studying TabPFN under limited-label and semi-supervised learning settings.

The benchmark emphasizes:

- Reproducible dataset splitting and experiment execution.
- Controlled labeled-data fractions.
- Consistent evaluation across models and methods.
- Classification, probabilistic, and calibration metrics.
- Experiment provenance and persistent run artifacts.

## Current status

The benchmark infrastructure currently supports reproducible supervised baseline experiments, including:

- YAML-based experiment configuration.
- OpenML dataset loading.
- Deterministic train/test and labeled/unlabeled splits.
- Supervised and semi-supervised baseline model execution.
- Classification, probabilistic, and calibration metrics.
- Experiment result serialization.
- Benchmark-level metadata and configuration persistence.
- CSV and JSONL result export.
- Command-line benchmark execution.
- Automated unit and integration tests.

## Benchmark design

Each benchmark experiment is defined by a combination of dataset, model, semi-supervised learning method, labeled-data fraction, random seed, test-set fraction.

For every experiment, the dataset is first divided into training and held-out test partitions. The training partition is subsequently divided into labeled and unlabeled subsets according to the requested label fraction.

The test set remains isolated from model training and is used exclusively for final evaluation.

## Repository structure

```text
tabular-ssl-benchmark/
├── configs/
│   ├── benchmark.yaml  # Global experimental settings for the benchmark
│   └── datasets.yaml   # Dataset catalogue used by the benchmark   
├── src/
│   ├── benchmark/      # Experiment execution, orchestration and persistence
│   ├── config/         # Validated configuration and experiment matrices
│   ├── datasets/       # Dataset loading and reproducible splitting
│   ├── evaluation/     # Classification and calibration metrics
│   ├── models/         # Benchmark model interfaces and implementations
│   ├── ssl_methods/            # Supervised and semi-supervised learning methods
│   └── utils/          # Shared reproducibility utilities
├── tests/
│   ├── config/
│   │   └── smoke/      # Minimal real benchmark configuration
│   └── ...             # Unit and integration tests
├── pyproject.toml
└── README.md
```

## Installation

Clone the repository and create a Python virtual environment:

```bash
git clone <repository-url>
cd tabular-ssl-benchmark

python -m venv .venv
```

Activate the environment and install the project in editable mode:

```bash
python -m pip install -e .
```

The editable installation also exposes the benchmark command-line interface:

```bash
tabular-ssl-benchmark --help
```

## Configuration

Benchmark execution is controlled through separate YAML files for dataset
selection and experiment configuration.

1) **Dataset configuration**: each enabled dataset is loaded from OpenML using its numeric dataset
identifier.

2) **Benchmark configuration**: the benchmark matrix is generated from the configured datasets, models, label fractions and random seeds. The variable ```test_size``` controls the fraction of the dataset reserved exclusively for final evaluation.

## Running the benchmark

A benchmark run can be executed using this command:

```bash
tabular-ssl-benchmark \
    --datasets-config path/to/datasets.yaml \
    --benchmark-config path/to/benchmark.yaml \
    --output-dir results/my-run
```

The command loads and validates the configuration, executes the complete experiment matrix and persists the resulting artifacts in the selected output directory.

## Output artifacts

Each persisted benchmark run generates the following artifacts in ```results/my-run/```:

- ```config.json```: contains the validated effective configuration used by the benchmark.
- ```metadata.json```: records run-level provenance information such as creation time (UTC), Python version, execution platform, Git commit and number of experiments.
- ```results.jsonl```: stores one serialized record per experiment.
- ```results.csv```: provides the same experiment results in tabular form for subsequent statistical analysis.

## Reproducibility

The benchmark is designed around deterministic and traceable experiment execution.

- Random seeds are propagated through dataset splitting and model creation. For a fixed dataset, configuration and seed, experiments use the same train/test and labeled/unlabeled partitions.

- Each persisted run stores both the effective benchmark configuration and execution metadata, including the Git commit associated with the run. This allows experimental results to be traced back to the configuration and source-code state that produced them.

- The held-out test set is never used for model training.

- Execution timestamps are stored in UTC using timezone-aware ISO 8601 representations.

## Development setup

The project requires Python 3.10 or newer. Continuous integration currently checks Python 3.10 and 3.12.

Create a virtual environment if you have not already created one:

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project and its development tools:

```bash
python -m pip install -e ".[dev]"
```

The development dependencies include pytest for automated tests, Ruff for static analysis, and build for package distribution checks.

## Continuous integration

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests.

Each run checks Python 3.10 and 3.12 independently. For each version, the workflow:

1. Installs the project and development dependencies.
2. Checks installed dependency consistency.
3. Runs Ruff and the unit and integration tests.
4. Checks the development CLI.
5. Builds the wheel and source distribution.
6. Installs the wheel in a separate clean virtual environment.
7. Checks dependency consistency in that environment.
8. Runs the installed CLI from outside the repository.

The separate wheel installation checks that CLI startup works with
the distributed package, without relying on the editable installation or the repository's working directory.

## Testing

Run test suite with:

```bash
python -m pytest -v
```

The automated suite uses synthetic data and mocked OpenML responses.
Real OpenML execution is checked separately using the smoke-test
configuration below.

### Real OpenML smoke test

Run a small end-to-end benchmark using Logistic Regression on Iris:

```bash
tabular-ssl-benchmark \
    --datasets-config tests/config/smoke/test_datasets.yaml \
    --benchmark-config tests/config/smoke/test_benchmark.yaml \
    --output-dir results/smoke-test
```

This command requires network access when the dataset is not already
cached. It is not executed by the CI workflow.

Check that the output directory contains:

- `config.json`
- `metadata.json`
- `results.jsonl`
- `results.csv`
