"""Tests for experiment configuration utilities."""

import pytest

from benchmark.experiment import ExperimentConfig


def test_experiment_id_supervised():
    """Supervised experiments should include supervised in their identifier."""
    config = ExperimentConfig(
        dataset_name="adult",
        model_name="tabpfn",
        ssl_method=None,
        label_fraction=0.1,
        seed=42,
    )

    assert (
        config.experiment_id
        == "adult__model-tabpfn__ssl-supervised__lf-0.1__test-0.2__seed-42"
    )


def test_experiment_id_ssl():
    """SSL experiments should include the SSL method in their identifier."""
    config = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        label_fraction=0.05,
        seed=1,
    )

    assert (
        config.experiment_id
        == "adult__model-random_forest__ssl-self_training__lf-0.05__test-0.2__seed-1"
    )


def test_invalid_label_fraction():
    """Label fractions outside the valid interval (0, 1] should be rejected."""
    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset_name="adult",
            model_name="tabpfn",
            ssl_method=None,
            label_fraction=0,
            seed=42,
        )


def test_invalid_test_size():
    """Test sizes outside the valid interval (0, 1) should be rejected."""
    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset_name="adult",
            model_name="tabpfn",
            ssl_method=None,
            label_fraction=0.1,
            seed=42,
            test_size=1.0,
        )
        
def test_experiment_id_changes_with_test_size():
    """Different test sizes should produce different experiment identifiers."""
    first = ExperimentConfig(
        dataset_name="adult",
        model_name="tabpfn",
        ssl_method=None,
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    second = ExperimentConfig(
        dataset_name="adult",
        model_name="tabpfn",
        ssl_method=None,
        label_fraction=0.1,
        seed=42,
        test_size=0.3,
    )

    assert first.experiment_id != second.experiment_id

def test_experiment_id_changes_with_ssl_parameters():
    """Different SSL hyperparameters should produce different IDs."""
    first = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        ssl_params={
            "confidence_threshold": 0.95,
            "max_iterations": 10,
        },
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    second = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        ssl_params={
            "confidence_threshold": 0.80,
            "max_iterations": 10,
        },
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    assert first.experiment_id != second.experiment_id

def test_experiment_id_is_independent_of_ssl_parameter_order():
    """SSL parameter insertion order should not affect experiment IDs."""
    first = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        ssl_params={
            "confidence_threshold": 0.95,
            "max_iterations": 10,
        },
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    second = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        ssl_params={
            "max_iterations": 10,
            "confidence_threshold": 0.95,
        },
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    assert first.experiment_id == second.experiment_id
    
def test_experiment_id_with_ssl_parameters_is_deterministic():
    """Equal SSL configurations should produce identical experiment IDs."""
    params = {
        "confidence_threshold": 0.95,
        "max_iterations": 10,
    }

    first = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        ssl_params=params.copy(),
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    second = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        ssl_params=params.copy(),
        label_fraction=0.1,
        seed=42,
        test_size=0.2,
    )

    assert first.experiment_id == second.experiment_id