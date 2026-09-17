"""
Split the training set into Bullinaria-style incremental sessions:

  - 6 sessions of 200 patterns each (20 per class x 10 classes)
  - a validation set built from the remaining training patterns
  - stratified sampling without replacement
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np


DEFAULT_N_SESSIONS = 6


def split_incremental(
    X: np.ndarray,
    y: np.ndarray,
    n_sessions:  int = DEFAULT_N_SESSIONS,
    ppc:         int = 20,
    num_classes: int = 10,
    rng=None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Return balanced incremental sessions plus a validation split.

    Class indices are shuffled independently so each session receives exactly
    ppc examples per digit.
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
