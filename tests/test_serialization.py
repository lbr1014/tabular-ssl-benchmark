from dataclasses import replace

import pytest

from benchmark.experiment import ExperimentConfig, ExperimentResult
from benchmark.serialization import serialize_experiment_result

@pytest.fixture
def experiment_result() -> ExperimentResult:
    """Create a representative experiment result for serialization tests."""
    config = ExperimentConfig(
        dataset_name="iris",
        model_name="logistic_regression",
        ssl_method="supervised",
        label_fraction=0.5,
        seed=42,
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

def test_serialize_experiment_result_contains_configuration(
    experiment_result,
):
    """Serialized results should preserve experiment configuration."""
    record = serialize_experiment_result(
        experiment_result,
    )

    assert record["experiment_id"] == (
        experiment_result.config.experiment_id
    )
    assert record["dataset_name"] == (
        experiment_result.config.dataset_name
    )
    assert record["model_name"] == (
        experiment_result.config.model_name
    )
    assert record["ssl_method"] == (
        experiment_result.config.ssl_method
    )
    assert record["label_fraction"] == (
        experiment_result.config.label_fraction
    )
    assert record["seed"] == experiment_result.config.seed
    assert record["test_size"] == (
        experiment_result.config.test_size
    )
    
def test_serialize_experiment_result_includes_metrics(
    experiment_result,
):
    """Evaluation metrics should be flattened into the result record."""
    record = serialize_experiment_result(
        experiment_result,
    )

    for metric_name, value in experiment_result.metrics.items():
        assert record[metric_name] == value
        
def test_serialize_experiment_result_includes_execution_data(
    experiment_result,
):
    """Execution times and partition sizes should be preserved."""
    record = serialize_experiment_result(
        experiment_result,
    )

    assert record["fit_time"] == experiment_result.fit_time
    assert record["predict_time"] == experiment_result.predict_time
    assert record["n_train"] == experiment_result.n_train
    assert record["n_labeled"] == experiment_result.n_labeled
    assert record["n_unlabeled"] == experiment_result.n_unlabeled
    assert record["n_test"] == experiment_result.n_test
    
def test_serialize_experiment_result_includes_metadata(
    experiment_result,
):
    """Dataset provenance metadata should be preserved."""
    record = serialize_experiment_result(
        experiment_result,
    )

    assert record["dataset_source"] == "openml"
    assert record["dataset_source_id"] == 61
    assert record["dataset_source_version"] == 1
    
def test_serialize_experiment_result_rejects_metadata_collisions(
    experiment_result,
):
    """Metadata must not overwrite core experiment fields."""
    conflicting_result = replace(
        experiment_result,
        metadata={
            "seed": 999,
        },
    )

    with pytest.raises(
        ValueError,
        match="reserved result fields",
    ):
        serialize_experiment_result(
            conflicting_result,
        )
        
def test_serialize_experiment_result_rejects_metrics_collisions(
    experiment_result,
):
    """Metrics must not overwrite core experiment fields."""
    conflicting_result = replace(
        experiment_result,
        metrics={
            "seed": 0.95,
        },
    )

    with pytest.raises(
        ValueError,
        match="reserved result fields",
    ):
        serialize_experiment_result(
            conflicting_result,
        )