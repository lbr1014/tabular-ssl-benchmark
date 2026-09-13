"""Tests for single supervised benchmark experiment execution."""

from benchmark.experiment import ExperimentResult
from benchmark.runner import run_supervised_experiment
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