"""
infrastructure/data/incremental_splitter.py
============================================
Divide el conjunto de entrenamiento en sesiones incrementales al estilo
de Bullinaria (2009):

  - 6 sesiones de 200 patrones (20 por clase × 10 clases)
  - Un conjunto de validación con el resto de los patrones de entrenamiento
  - Muestreo estratificado y sin reemplazo
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np


def split_incremental(
    X: np.ndarray,
    y: np.ndarray,
    n_sessions:  int = 6,
    ppc:         int = 20,      # patrones por clase por sesión
    num_classes: int = 10,
    rng=None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Divide (X, y) en:
      sessions — lista de n_sessions tuples (X_s, y_s), cada uno con
                 ppc * num_classes patrones balanceados por clase.
      val      — tuple (X_val, y_val) con los patrones restantes.

    Los índices se barajan de forma independiente por clase para
    garantizar que cada sesión tenga exactamente ppc ejemplos de cada dígito.
    """
    if rng is None:
        rng = np.random.default_rng()

    class_idx = [np.where(y == c)[0].copy() for c in range(num_classes)]
    for idx in class_idx:
        rng.shuffle(idx)

    batches_idx = [[] for _ in range(n_sessions)]
    val_idx     = []

    for c in range(num_classes):
        needed = ppc * n_sessions
        pool   = class_idx[c][:needed]
        val_idx.extend(class_idx[c][needed:].tolist())
        for s in range(n_sessions):
            batches_idx[s].extend(pool[s * ppc : (s + 1) * ppc].tolist())

    sessions = []
    for s in range(n_sessions):
        idx = np.array(batches_idx[s], dtype=np.int64)
        rng.shuffle(idx)
        sessions.append((X[idx], y[idx]))

    val = np.array(val_idx, dtype=np.int64)
    rng.shuffle(val)

    return sessions, (X[val], y[val])
