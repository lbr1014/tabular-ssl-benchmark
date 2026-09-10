"""Utilities for reproducible random number generation.

This module centralises the random seed handling used throughout the
benchmark.
"""
from enum import IntEnum
import os
import random

import numpy as np

class SeedStream(IntEnum):
    """Independent deterninistic random stream used by the benchmark."""
    
    MODEL = 100
    
def set_global_seed(seed: int) -> None:
    """Set the global random seed for reproducibility.

    This function sets the random seed for Python's built-in `random`
    module, NumPy, and any other libraries that rely on these for
    randomness. It also sets the `PYTHONHASHSEED` environment variable
    to ensure consistent hashing of objects across runs.

    Args:
        seed (int): The random seed to set.
    """
    validate_seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def derive_seed(seed: int, stream: int) -> int:
    """Derive a deterministic child seed from a root seed.
    
    Args:
        seed (int): The root random seed
        stream (int): An integer representing the stochastic component.
        
    Returns:
        int: A deterministic child seed derived from the root seed.
    """
    validate_seed(seed)
    validate_seed(stream, name="stream")

    seed_sequence = np.random.SeedSequence(
        entropy=seed,
        spawn_key=(stream,),
    )

    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    
def validate_seed(seed: int, name: str = "seed") -> None:
    """Validate a random seed.

    Args:
        seed (int): The random seed to validate.
        name (str): The name of the seed parameter for error messages.
    """
    
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"{name} must be an integer.")

    if seed < 0:
        raise ValueError(f"{name} must be non-negative.")