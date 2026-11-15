"""Tests for single supervised benchmark experiment execution."""

import pytest

from benchmark.experiment import ExperimentResult
from benchmark.runner import run_benchmark_matrix, run_experiment, run_experiment_spec, run_supervised_experiment
from config.matrix import ExperimentSpec
from config.models import DatasetConfig, SSLMethodConfig
from models.sklearn_models import (
    create_logistic_regression,
    create_random_forest,
)
from ssl_methods.supervised import SupervisedMethod
from ssl_methods.base import SSLMethod
from unittest.mock import Mock

class StandaloneTestMethod(SSLMethod):
    """Test-only strategy that creates its own classifier."""

    @property
    def name(self) -> str:
        return "standalone_test"

    @property
    def params(self) -> dict[str, object]:
        return {}

    @property
    def requires_base_model(self) -> bool:
        return False

    def fit(
        self,
        *,
        model,
        x_labeled,
        y_labeled,
        x_unlabeled,
    ):
        assert model is None
        assert len(x_unlabeled) > 0

        classifier = create_logistic_regression(seed=42)
        return classifier.fit(x_labeled, y_labeled)

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
        ssl_method=SSLMethodConfig(name="supervised"),
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
    assert result.config.ssl_method == "supervised"
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
        ssl_method=SSLMethodConfig(name="supervised"),
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
        ssl_method=SSLMethodConfig(name="supervised"),
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
        ssl_method=SSLMethodConfig(name="supervised"),
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
            ssl_method=SSLMethodConfig(name="supervised"),
            label_fraction=0.5,
            seed=1,
            test_size=0.2,
        ),
        ExperimentSpec(
            dataset=dataset_config,
            model_name="random_forest",
            ssl_method=SSLMethodConfig(name="supervised"),
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
            ssl_method=SSLMethodConfig(name="supervised"),
            label_fraction=0.5,
            seed=1,
            test_size=0.2,
        ),
        ExperimentSpec(
            dataset=dataset_config,
            model_name="random_forest",
            ssl_method=SSLMethodConfig(name="supervised"),
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
        ssl_method=SSLMethodConfig(name="supervised"),
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

def test_generic_runner_records_ssl_method(
    mixed_dataset,
):
    """Runner should record the executed learning strategy."""
    model = create_logistic_regression(seed=42)
    method = SupervisedMethod()

    result = run_experiment(
        dataset=mixed_dataset,
        model=model,
        ssl_method=method,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert result.config.ssl_method == "supervised"
    assert result.config.ssl_params == {}

def test_generic_supervised_runner_matches_legacy_runner(
    mixed_dataset,
):
    """Generic supervised execution should preserve previous results."""
    legacy_model = create_logistic_regression(seed=42)
    generic_model = create_logistic_regression(seed=42)

    legacy_result = run_supervised_experiment(
        dataset=mixed_dataset,
        model=legacy_model,
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    generic_result = run_experiment(
        dataset=mixed_dataset,
        model=generic_model,
        ssl_method=SupervisedMethod(),
        label_fraction=0.5,
        test_size=0.2,
        seed=42,
    )

    assert generic_result.metrics == legacy_result.metrics
    assert generic_result.n_train == legacy_result.n_train
    assert generic_result.n_labeled == legacy_result.n_labeled
    assert generic_result.n_unlabeled == legacy_result.n_unlabeled
    assert generic_result.n_test == legacy_result.n_test

def test_generic_runner_supports_empty_unlabeled_partition(
    mixed_dataset,
):
    """Generic runner should support experiments with no unlabeled samples."""
    model = create_logistic_regression(seed=42)
    method = SupervisedMethod()

    result = run_experiment(
        dataset=mixed_dataset,
        model=model,
        ssl_method=method,
        label_fraction=1.0,
        test_size=0.2,
        seed=42,
    )

    assert result.n_unlabeled == 0
    assert result.n_labeled == result.n_train
    assert result.config.ssl_method == "supervised"

def test_run_experiment_spec_rejects_unknown_ssl_method(
    mixed_dataset,
):
    """Experiment execution should reject unregistered SSL methods."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name="logistic_regression",
        ssl_method=SSLMethodConfig(name="unknown_ssl_method"),
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    with pytest.raises(
        ValueError,
        match="Unknown SSL method",
    ):
        run_experiment_spec(
            spec=spec,
            dataset=mixed_dataset,
        )

def test_run_experiment_spec_supports_configured_self_training(
    mixed_dataset,
):
    """Experiment specifications should configure self-training parameters."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name="logistic_regression",
        ssl_method=SSLMethodConfig(
            name="self_training",
            params={
                "confidence_threshold": 0.95,
                "max_iterations": 5,
            },
        ),
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    result = run_experiment_spec(
        spec=spec,
        dataset=mixed_dataset,
    )

    assert isinstance(result, ExperimentResult)
    assert result.config.ssl_method == "self_training"
    assert result.config.ssl_params == {
        "confidence_threshold": 0.95,
        "max_iterations": 5,
    }

def test_run_experiment_spec_records_effective_ssl_defaults(
    mixed_dataset,
):
    """Experiment results should record effective SSL default parameters."""
    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name="logistic_regression",
        ssl_method=SSLMethodConfig(
            name="self_training",
        ),
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    result = run_experiment_spec(
        spec=spec,
        dataset=mixed_dataset,
    )

    assert result.config.ssl_params == {
        "confidence_threshold": 0.95,
        "max_iterations": 10,
    }

def test_standalone_spec_executes_without_creating_base_model(
    monkeypatch,
    mixed_dataset,
):
    """Standalone execution must not invoke the base-model factory."""
    method = StandaloneTestMethod()
    method_factory = Mock(return_value=method)
    classifier_factory = Mock(
        side_effect=AssertionError("Base-model factory must not be called")
    )

    monkeypatch.setattr(
        "benchmark.runner.create_ssl_method",
        method_factory,
    )
    monkeypatch.setattr(
        "benchmark.runner.create_classifier",
        classifier_factory,
    )

    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name=None,
        ssl_method=SSLMethodConfig(
            name="standalone_test",
            requires_base_model=False,
        ),
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    result = run_experiment_spec(spec=spec, dataset=mixed_dataset)

    method_factory.assert_called_once_with("standalone_test")
    classifier_factory.assert_not_called()
    assert isinstance(result, ExperimentResult)
    assert result.config.model_name is None
    assert result.config.ssl_method == "standalone_test"
    assert "__model-standalone__" in result.config.experiment_id
    assert "f1_macro" in result.metrics
    assert result.n_labeled + result.n_unlabeled == result.n_train

@pytest.mark.parametrize(
    "requires_base_model",
    [True, False],
)
def test_runner_rejects_incompatible_base_model_before_splitting(
    monkeypatch,
    mixed_dataset,
    requires_base_model,
):
    """Invalid method/model combinations must fail before splitting."""
    split_mock = Mock(
        side_effect=AssertionError("Dataset splitting must not be called")
    )
    monkeypatch.setattr(
        "benchmark.runner.create_ssl_split",
        split_mock,
    )

    if requires_base_model:
        method = SupervisedMethod()
        model = None
        message = "requires a base model"
    else:
        method = StandaloneTestMethod()
        model = create_logistic_regression(seed=42)
        message = "does not accept a base model"

    with pytest.raises(ValueError, match=message):
        run_experiment(
            dataset=mixed_dataset,
            model=model,
            ssl_method=method,
            label_fraction=0.5,
            test_size=0.2,
            seed=42,
        )

    split_mock.assert_not_called()

@pytest.mark.parametrize(
    "actual_requires_base_model",
    [True, False],
)
def test_spec_rejects_inconsistent_model_requirement(
    monkeypatch,
    mixed_dataset,
    actual_requires_base_model,
):
    """Configured and implemented model requirements must agree."""
    method = (
        SupervisedMethod()
        if actual_requires_base_model
        else StandaloneTestMethod()
    )
    monkeypatch.setattr(
        "benchmark.runner.create_ssl_method",
        Mock(return_value=method),
    )

    classifier_factory = Mock()
    execution_mock = Mock()
    monkeypatch.setattr(
        "benchmark.runner.create_classifier",
        classifier_factory,
    )
    monkeypatch.setattr(
        "benchmark.runner.run_experiment",
        execution_mock,
    )

    spec = ExperimentSpec(
        dataset=DatasetConfig(
            name=mixed_dataset.name,
            openml_id=1,
        ),
        model_name=None if actual_requires_base_model else "logistic_regression",
        ssl_method=SSLMethodConfig(
            name=method.name,
            requires_base_model=not actual_requires_base_model,
        ),
        label_fraction=0.5,
        seed=42,
        test_size=0.2,
    )

    with pytest.raises(
        ValueError,
        match="Configured requires_base_model does not match",
    ):
        run_experiment_spec(spec=spec, dataset=mixed_dataset)

    classifier_factory.assert_not_called()
    execution_mock.assert_not_called()