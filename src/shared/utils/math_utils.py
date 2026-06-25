"""
shared/utils/math_utils.py
===========================
Funciones matemáticas compartidas por todo el proyecto.
"""
from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoide numericamente estable."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def dsigmoid_from_output(y: np.ndarray, ospo: float = 0.0) -> np.ndarray:
    """
    Derivada de la sigmoide calculada desde la salida (no la entrada),
    con sigmoid prime offset (oSPO) para prevenir saturación.
    """
    return y * (1.0 - y) + ospo


def make_one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Codifica etiquetas enteras como vectores one-hot."""
    T = np.zeros((len(y), num_classes), dtype=np.float64)
    T[np.arange(len(y)), y] = 1.0
    return T
