import pytest

from benchmark.experiment import ExperimentConfig


def test_experiment_id_supervised():
    config = ExperimentConfig(
        dataset_name="adult",
        model_name="tabpfn",
        ssl_method=None,
        label_fraction=0.1,
        seed=42,
    )

    assert (
        config.experiment_id
        == "adult__tabpfn__supervised__lf-0.1__seed-42"
    )


def test_experiment_id_ssl():
    config = ExperimentConfig(
        dataset_name="adult",
        model_name="random_forest",
        ssl_method="self_training",
        label_fraction=0.05,
        seed=1,
    )

    assert (
        config.experiment_id
        == "adult__random_forest__self_training__lf-0.05__seed-1"
    )


def test_invalid_label_fraction():
    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset_name="adult",
            model_name="tabpfn",
            ssl_method=None,
            label_fraction=0,
            seed=42,
        )


def test_invalid_test_size():
    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset_name="adult",
            model_name="tabpfn",
            ssl_method=None,
            label_fraction=0.1,
            seed=42,
            test_size=1.0,
        )