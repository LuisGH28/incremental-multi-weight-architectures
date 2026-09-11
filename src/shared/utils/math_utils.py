"""
shared/utils/math_utils.py
===========================
Shared mathematical utilities.
"""
from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def dsigmoid_from_output(y: np.ndarray, ospo: float = 0.0) -> np.ndarray:
    """
    Sigmoid derivative computed from the output, with sigmoid prime offset
    (oSPO) to reduce saturation.
    """
    return y * (1.0 - y) + ospo


def make_one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Encode integer labels as one-hot vectors."""
    T = np.zeros((len(y), num_classes), dtype=np.float64)
    T[np.arange(len(y)), y] = 1.0
    return T
