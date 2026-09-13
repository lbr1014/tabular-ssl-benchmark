"""Tests for benchmark classifier factory utilities."""

import pytest

from models.base import BenchmarkClassifier
from models.factory import create_classifier


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "random_forest",
    ],
)
def test_create_classifier_returns_registered_model(
    model_name,
):
    """Registered model names should create benchmark classifiers."""
    model = create_classifier(
        model_name,
        seed=42,
    )

    assert isinstance(
        model,
        BenchmarkClassifier,
    )

    assert model.name == model_name


def test_create_classifier_is_reproducible():
    """Equal seeds should produce equal classifier configurations."""
    first = create_classifier(
        "random_forest",
        seed=42,
    )

    second = create_classifier(
        "random_forest",
        seed=42,
    )

    assert first.get_params() == second.get_params()


def test_create_classifier_rejects_unknown_model():
    """Unknown model identifiers should fail explicitly."""
    with pytest.raises(
        ValueError,
        match="Unknown benchmark model",
    ):
        create_classifier(
            "unknown_model",
            seed=42,
        )


@pytest.mark.parametrize(
    ("model_name", "exception"),
    [
        (None, TypeError),
        (42, TypeError),
        ("", ValueError),
        ("   ", ValueError),
    ],
)
def test_create_classifier_rejects_invalid_model_names(
    model_name,
    exception,
):
    """Invalid model identifiers should fail explicitly."""
    with pytest.raises(exception):
        create_classifier(
            model_name,
            seed=42,
        )