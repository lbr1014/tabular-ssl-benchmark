"""Utilities for reproducible random number generation.

This module centralises the random seed handling used throughout the
benchmark.
"""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set the global random seed for reproducibility.

    This function sets the random seed for Python's built-in `random`
    module, NumPy, and any other libraries that rely on these for
    randomness. It also sets the `PYTHONHASHSEED` environment variable
    to ensure consistent hashing of objects across runs.

    Args:
        seed (int): The random seed to set.
    """
    _validate_seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
def _validate_seed(seed: int) -> None:
    """Validate a random seed.

    Args:
        seed (int): The random seed to validate.
    """
    
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")