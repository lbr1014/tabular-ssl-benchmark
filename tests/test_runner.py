"""Tests for single supervised benchmark experiment execution."""

import pytest

from benchmark.experiment import ExperimentResult
from benchmark.runner import run_benchmark_matrix, run_experiment_spec, run_supervised_experiment
from config.matrix import ExperimentSpec
from config.models import DatasetConfig
from models.sklearn_models import (
    create_logistic_regression,
    create_random_forest,
)

def test_supervised_runner_returns_experiment_result(
    mixed_dataset,
):
    """Runner should produce a complete experiment result."""
    model = create_logistic_regression(
        seed=42,
    )

    result = run_supervised_experiment(
        dataset=mixed_dataset,
        model=model,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert isinstance(
        result,
        ExperimentResult,
    )

    assert result.config.dataset_name == mixed_dataset.name
    assert result.config.model_name == model.name
    assert result.config.ssl_method == "supervised"
    
def test_supervised_runner_computes_all_metrics(
    mixed_dataset,
):
    """Runner should compute every benchmark evaluation metric."""
    model = create_logistic_regression(
        seed=42,
    )

    result = run_supervised_experiment(
        dataset=mixed_dataset,
        model=model,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert set(result.metrics) == {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "mcc",
        "log_loss",
        "roc_auc",
        "brier_score",
        "ece",
    }
    
def test_supervised_runner_records_partition_sizes(
    mixed_dataset,
):
    """Runner should record the experimental partition sizes."""
    model = create_logistic_regression(
        seed=42,
    )

    result = run_supervised_experiment(
        dataset=mixed_dataset,
        model=model,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert result.n_train > 0
    assert result.n_labeled > 0
    assert result.n_unlabeled > 0
    assert result.n_test > 0

    assert (
        result.n_labeled
        + result.n_unlabeled
        == result.n_train
    )

    assert (
        result.n_train
        + result.n_test
        == mixed_dataset.n_samples
    )
    
def test_supervised_runner_records_execution_times(
    mixed_dataset,
):
    """Runner should record non-negative fit and prediction times."""
    model = create_logistic_regression(
        seed=42,
    )

    result = run_supervised_experiment(
        dataset=mixed_dataset,
        model=model,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert result.fit_time >= 0.0
    assert result.predict_time >= 0.0
    
def test_supervised_runner_is_reproducible(
    mixed_dataset,
):
    """Equal experiment seeds should reproduce metric values."""
    first_model = create_random_forest(
        seed=42,
    )

    second_model = create_random_forest(
        seed=42,
    )

    first = run_supervised_experiment(
        dataset=mixed_dataset,
        model=first_model,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    second = run_supervised_experiment(
        dataset=mixed_dataset,
        model=second_model,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert first.metrics == second.metrics
    
def test_supervised_runner_supports_fully_labeled_training_data(
    mixed_dataset,
):
    """Runner should support experiments with no unlabeled samples."""
    model = create_logistic_regression(
        seed=42,
    )

    result = run_supervised_experiment(
        dataset=mixed_dataset,
        model=model,
        label_fraction=1.0,
        test_size=0.2,
        seed=42,
    )

    assert result.n_unlabeled == 0
    assert result.n_labeled == result.n_train
    assert result.n_train + result.n_test == mixed_dataset.n_samples
    
def test_run_experiment_spec_returns_result(
    mixed_dataset,
):
    """An experiment specification should execute end to end."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name="logistic_regression",
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    result = run_experiment_spec(
        spec=spec,
        dataset=mixed_dataset,
    )

    assert isinstance(
        result,
        ExperimentResult,
    )

    assert result.config.dataset_name == mixed_dataset.name
    assert result.config.model_name == "logistic_regression"
    assert result.config.label_fraction == 0.5
    assert result.config.seed == 42
    assert result.config.test_size == 0.2
    
@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "random_forest",
    ],
)
def test_run_experiment_spec_supports_registered_models(
    mixed_dataset,
    model_name,
):
    """Experiment specifications should create registered classifiers."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name=model_name,
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    result = run_experiment_spec(
        spec=spec,
        dataset=mixed_dataset,
    )

    assert result.config.model_name == model_name
    
def test_run_experiment_spec_rejects_dataset_mismatch(
    mixed_dataset,
):
    """A specification must not run on a different dataset."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name="different_dataset",
            openml_id=1,
        ),
        model_name="logistic_regression",
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    with pytest.raises(
        ValueError,
        match="Loaded dataset does not match",
    ):
        run_experiment_spec(
            spec=spec,
            dataset=mixed_dataset,
        )
        
def test_run_experiment_spec_is_reproducible(
    mixed_dataset,
):
    """Equal specifications should produce equal experiment results."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name="random_forest",
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    first = run_experiment_spec(
        spec=spec,
        dataset=mixed_dataset,
    )

    second = run_experiment_spec(
        spec=spec,
        dataset=mixed_dataset,
    )

    assert first.metrics == second.metrics
    
def test_run_benchmark_matrix_executes_all_specs(
    mixed_dataset,
):
    """Every matrix specification should produce one result."""
    dataset_config = DatasetConfig(
        name=mixed_dataset.name,
        openml_id=1,
    )

    matrix = (
        ExperimentSpec(
            dataset=dataset_config,
            model_name="logistic_regression",
            label_fraction=0.5,
            seed=1,
            test_size=0.2,
        ),
        ExperimentSpec(
            dataset=dataset_config,
            model_name="random_forest",
            label_fraction=0.5,
            seed=1,
            test_size=0.2,
        ),
    )

    results = run_benchmark_matrix(
        matrix=matrix,
        datasets={
            mixed_dataset.name: mixed_dataset,
        },
    )

    assert len(results) == len(matrix)

    assert all(
        isinstance(result, ExperimentResult)
        for result in results
    )
    
def test_run_benchmark_matrix_preserves_order(
    mixed_dataset,
):
    """Matrix results should preserve specification order."""
    dataset_config = DatasetConfig(
        name=mixed_dataset.name,
        openml_id=1,
    )

    matrix = (
        ExperimentSpec(
            dataset=dataset_config,
            model_name="logistic_regression",
            label_fraction=0.5,
            seed=1,
            test_size=0.2,
        ),
        ExperimentSpec(
            dataset=dataset_config,
            model_name="random_forest",
            label_fraction=0.5,
            seed=2,
            test_size=0.2,
        ),
    )

    results = run_benchmark_matrix(
        matrix=matrix,
        datasets={
            mixed_dataset.name: mixed_dataset,
        },
    )

    assert [
        result.config.model_name
        for result in results
    ] == [
        spec.model_name
        for spec in matrix
    ]

    assert [
        result.config.seed
        for result in results
    ] == [
        spec.seed
        for spec in matrix
    ]
    
def test_run_benchmark_matrix_rejects_missing_dataset(
    mixed_dataset,
):
    """Matrix execution should fail when a dataset is not loaded."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name="missing_dataset",
            openml_id=1,
        ),
        model_name="logistic_regression",
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    with pytest.raises(
        ValueError,
        match="is not loaded",
    ):
        run_benchmark_matrix(
            matrix=(spec,),
            datasets={
                mixed_dataset.name: mixed_dataset,
            },
        )
        
def test_run_benchmark_matrix_accepts_empty_matrix():
    """An empty experiment matrix should produce no results."""
    results = run_benchmark_matrix(
        matrix=(),
        datasets={},
    )

    assert results == ()
    
def test_run_benchmark_matrix_rejects_non_tuple_matrix():
    """Matrix execution should require an immutable tuple."""
    with pytest.raises(
        TypeError,
        match="matrix must be a tuple",
    ):
        run_benchmark_matrix(
            matrix=[],
            datasets={},
        )