import pytest
import numpy as np

from ssl_methods.self_training import SelfTrainingMethod
from models.base import BenchmarkClassifier


class FakeClassifier(BenchmarkClassifier):
    """Deterministic classifier used to test self-training behavior."""

    def __init__(
        self,
        probabilities: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        self._probabilities = probabilities
        self._classes = classes
        self.fit_calls: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def name(self) -> str:
        return "fake_classifier"

    @property
    def classes_(self) -> np.ndarray:
        return self._classes

    def fit(self, X, y):
        self.fit_calls.append(
            (
                np.asarray(X).copy(),
                np.asarray(y).copy(),
            )
        )
        return self

    def predict(self, X):
        indices = self.predict_proba(X).argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        return self._probabilities[: len(X)]

    def get_params(self):
        return {}
    
def test_self_training_has_stable_name():
    """Self-training should expose a stable strategy identifier."""
    method = SelfTrainingMethod()

    assert method.name == "self_training"


def test_self_training_uses_expected_defaults():
    """Self-training should expose reproducible default parameters."""
    method = SelfTrainingMethod()

    assert method.confidence_threshold == 0.95
    assert method.max_iterations == 10
    
@pytest.mark.parametrize(
    "threshold",
    ["error", 'c'],
)
def test_self_training_rejects_invalid_type_threshold(
    threshold,
):
    """Confidence threshold should define a valid probability."""
    with pytest.raises(
        TypeError,
        match="confidence_threshold",
    ):
        SelfTrainingMethod(
            confidence_threshold=threshold,
        )
    
@pytest.mark.parametrize(
    "threshold",
    [0.0, -0.1, 1.1],
)
def test_self_training_rejects_invalid_confidence_threshold(
    threshold,
):
    """Confidence threshold should define a valid probability."""
    with pytest.raises(
        ValueError,
        match="confidence_threshold",
    ):
        SelfTrainingMethod(
            confidence_threshold=threshold,
        )
        
@pytest.mark.parametrize(
    "max_iterations",
    [1.1, "error", 'c'],
)
def test_self_training_rejects_invalid_type_max_iterations(
    max_iterations,
):
    """Confidence max_iterations should define a valid integer."""
    with pytest.raises(
        TypeError,
        match="max_iterations",
    ):
        SelfTrainingMethod(
            max_iterations=max_iterations,
        )
        
@pytest.mark.parametrize(
    "max_iterations",
    [-1, 0],
)
def test_self_training_rejects_invalid_max_iterations(
    max_iterations,
):
    """Maximum iterations must be a positive integer."""
    with pytest.raises(
        ValueError,
        match="max_iterations",
    ):
        SelfTrainingMethod(
            max_iterations=max_iterations,
        )
        
def test_self_training_with_no_unlabeled_samples_fits_once():
    """Empty unlabeled data should reduce to supervised training."""
    model = FakeClassifier(
        probabilities=np.empty((0, 2)),
        classes=np.array(["negative", "positive"]),
    )

    method = SelfTrainingMethod()

    x_labeled = np.array([[0.0], [1.0]])
    y_labeled = np.array(["negative", "positive"])
    x_unlabeled = np.empty((0, 1))

    returned = method.fit(
        model=model,
        x_labeled=x_labeled,
        y_labeled=y_labeled,
        x_unlabeled=x_unlabeled,
    )

    assert returned is model
    assert len(model.fit_calls) == 1
    
def test_self_training_stops_when_no_sample_reaches_threshold():
    """Low-confidence predictions should not be pseudo-labeled."""
    model = FakeClassifier(
        probabilities=np.array(
            [
                [0.60, 0.40],
                [0.55, 0.45],
            ]
        ),
        classes=np.array(["negative", "positive"]),
    )

    method = SelfTrainingMethod(
        confidence_threshold=0.95,
    )

    method.fit(
        model=model,
        x_labeled=np.array([[0.0], [1.0]]),
        y_labeled=np.array(["negative", "positive"]),
        x_unlabeled=np.array([[2.0], [3.0]]),
    )

    assert len(model.fit_calls) == 1
    
def test_self_training_uses_classifier_classes_for_pseudo_labels():
    """Pseudo-labels should preserve the classifier's class labels."""
    model = FakeClassifier(
        probabilities=np.array(
            [
                [0.99, 0.01],
                [0.02, 0.98],
            ]
        ),
        classes=np.array(["negative", "positive"]),
    )

    method = SelfTrainingMethod(
        confidence_threshold=0.95,
        max_iterations=1,
    )

    method.fit(
        model=model,
        x_labeled=np.array([[0.0], [1.0]]),
        y_labeled=np.array(["negative", "positive"]),
        x_unlabeled=np.array([[2.0], [3.0]]),
    )

    _, final_y = model.fit_calls[-1]

    assert final_y.tolist() == [
        "negative",
        "positive",
        "negative",
        "positive",
    ]
    
def test_self_training_exposes_effective_parameters() -> None:
    """Self-training should expose its effective configuration."""
    method = SelfTrainingMethod(
        confidence_threshold=0.90,
        max_iterations=5,
    )

    assert method.params == {
        "confidence_threshold": 0.90,
        "max_iterations": 5,
    }
    
def test_self_training_exposes_default_parameters() -> None:
    """Default self-training settings should be part of effective configuration."""
    method = SelfTrainingMethod()

    assert method.params == {
        "confidence_threshold": 0.95,
        "max_iterations": 10,
    }
    
