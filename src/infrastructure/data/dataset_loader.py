"""
infrastructure/data/dataset_loader.py
=======================================
Cargador unificado de datasets.

Para este experimento (fw²) el dataset principal es OptDigits UCI.
El loader acepta la misma interfaz que el proyecto base para facilitar
la integración futura con otros datasets.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# OptDigits (CSV propio de Bullinaria 2009)
# ─────────────────────────────────────────────────────────────────────────────

def _load_optdigits_csv(
    path: str,
    num_inputs: int = 64,
    scale: float = 16.0,
) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                float(row[0])
            except ValueError:
                continue
            labels.append(int(float(row[-1])))
            feats.append([float(v) for v in row[:num_inputs]])
    X = np.array(feats,  dtype=np.float64) / scale
    y = np.array(labels, dtype=np.int64)
    return X, y


def _load_optdigits(
    data_dir: str,
    tra_path: Optional[str] = None,
    tes_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tra = tra_path or os.path.join(data_dir, "optdigits.tra")
    tes = tes_path or os.path.join(data_dir, "optdigits.tes")
    X_train, y_train = _load_optdigits_csv(tra)
    X_test,  y_test  = _load_optdigits_csv(tes)
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Entrada pública
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    name: str = "optdigits",
    data_dir: str = "./data",
    tra_path: Optional[str] = None,
    tes_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga un dataset y devuelve (X_train, y_train, X_test, y_test).

    name      — identificador del dataset (actualmente solo 'optdigits')
    data_dir  — directorio raíz donde buscar los archivos
    tra_path  — ruta explícita al archivo de entrenamiento (opcional)
    tes_path  — ruta explícita al archivo de prueba (opcional)
    """
    if name == "optdigits":
        return _load_optdigits(data_dir, tra_path, tes_path)
    raise ValueError(
        f"Dataset '{name}' no reconocido. "
        f"Para fw² sólo se usa 'optdigits'."
    )
