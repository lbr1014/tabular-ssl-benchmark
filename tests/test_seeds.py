"""Tests for random seed utilities."""

import os
import random

import numpy as np
import pytest

from utils.seeds import derive_seed, set_global_seed


def test_python_random_is_reproducible():
    """Python random should reproduce the same sequence."""
    set_global_seed(42)
    first_sequence = [random.random() for _ in range(5)]

    set_global_seed(42)
    second_sequence = [random.random() for _ in range(5)]

    assert first_sequence == second_sequence


def test_numpy_random_is_reproducible():
    """NumPy should reproduce the same random sequence."""
    set_global_seed(42)
    first_sequence = np.random.random(5)

    set_global_seed(42)
    second_sequence = np.random.random(5)

    np.testing.assert_array_equal(first_sequence, second_sequence)


def test_different_seeds_produce_different_sequences():
    """Different seeds should normally produce different sequences."""
    set_global_seed(42)
    first_sequence = np.random.random(5)

    set_global_seed(123)
    second_sequence = np.random.random(5)

    assert not np.array_equal(first_sequence, second_sequence)


def test_pythonhashseed_environment_variable_is_set():
    """PYTHONHASHSEED should be stored in the environment."""
    set_global_seed(42)

    assert os.environ["PYTHONHASHSEED"] == "42"


@pytest.mark.parametrize("invalid_seed", [-1, -42])
def test_negative_seed_raises_value_error(invalid_seed):
    """Negative seeds should be rejected."""
    with pytest.raises(ValueError, match="seed must be non-negative"):
        set_global_seed(invalid_seed)


@pytest.mark.parametrize(
    "invalid_seed",
    [1.5, "42", None, True],
)
def test_non_integer_seed_raises_type_error(invalid_seed):
    """Non-integer seeds should be rejected."""
    with pytest.raises(TypeError, match="seed must be an integer"):
        set_global_seed(invalid_seed)
        
def test_derived_seed_is_reproducible():
    """The same root seed and stream should produce the same child seed."""
    first_seed = derive_seed(42, stream=0)
    second_seed = derive_seed(42, stream=0)

    assert first_seed == second_seed


def test_different_streams_produce_different_derived_seeds():
    """Different streams should produce different child seeds."""
    first_seed = derive_seed(42, stream=0)
    second_seed = derive_seed(42, stream=1)

    assert first_seed != second_seed


def test_different_root_seeds_produce_different_derived_seeds():
    """Different root seeds should produce different child seeds."""
    first_seed = derive_seed(42, stream=0)
    second_seed = derive_seed(123, stream=0)

    assert first_seed != second_seed


def test_derived_seed_is_valid_uint32():
    """Derived seeds should lie within the unsigned 32-bit range."""
    derived_seed = derive_seed(42, stream=0)

    assert 0 <= derived_seed <= np.iinfo(np.uint32).max


@pytest.mark.parametrize("invalid_stream", [-1, -42])
def test_negative_stream_raises_value_error(invalid_stream):
    """Negative stream identifiers should be rejected."""
    with pytest.raises(
        ValueError,
        match="stream must be non-negative",
    ):
        derive_seed(42, stream=invalid_stream)


@pytest.mark.parametrize(
    "invalid_stream",
    [1.5, "1", None, True],
)
def test_non_integer_stream_raises_type_error(invalid_stream):
    """Non-integer stream identifiers should be rejected."""
    with pytest.raises(
        TypeError,
        match="stream must be an integer",
    ):
        derive_seed(42, stream=invalid_stream)