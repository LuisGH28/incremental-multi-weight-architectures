"""
infrastructure/data/dataset_loader.py
=======================================
Dataset loader for the fw3 MNIST experiment.

Supported formats:
  1. Binary IDX: official MNIST/Yann LeCun format (.ubyte, .gz)
  2. CSV: one pattern per row, label in the last column, values in 0-255

Constant rationale:
  - n_inputs = 784 (28 x 28 pixels, versus 64 for OptDigits 8 x 8)
  - scale = 255.0 (versus 16.0 for OptDigits)
  Both normalizations map input values into [0, 1]. The MLP architecture and
  evolutionary protocol are otherwise unchanged.
"""
from __future__ import annotations

import csv
import gzip
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Dataset dimensions and normalization constants.
N_INPUTS_MNIST     = 784
N_INPUTS_OPTDIGITS = 64
SCALE_MNIST        = 255.0
SCALE_OPTDIGITS    = 16.0


def _open_idx(path: str):
    """Open an IDX file, with optional gzip compression."""
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _load_idx_images(path: str) -> np.ndarray:
    with _open_idx(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(
                f"Archivo de imágenes IDX inválido (magic={magic}): {path}\n"
                f"Se esperaba magic=2051. ¿Es un archivo de etiquetas?"
            )
        X = (
            np.frombuffer(f.read(), dtype=np.uint8)
            .reshape(n, rows * cols)
            .astype(np.float64)
            / SCALE_MNIST
        )
    return X


def _load_idx_labels(path: str) -> np.ndarray:
    with _open_idx(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(
                f"Archivo de etiquetas IDX inválido (magic={magic}): {path}\n"
                f"Se esperaba magic=2049. ¿Es un archivo de imágenes?"
            )
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    return y


def load_mnist_idx(
    images_path: str,
    labels_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST data from binary IDX files.

    Parameters
    ----------
    images_path : path to the images file (.ubyte or .ubyte.gz)
    labels_path : path to the labels file (.ubyte or .ubyte.gz)

    Returns
    -------
    X : float64 array with shape (N, 784), values in [0, 1]
    y : int64 array with shape (N,), values in {0, ..., 9}
    """
    X = _load_idx_images(images_path)
    y = _load_idx_labels(labels_path)
    return X, y


def _load_csv(
    path: str,
    num_inputs: int,
    scale: float,
    label_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV data with either Kaggle/OptDigits-style trailing labels or
    alternative leading labels.
    """
    feats, labels = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                float(row[0])
            except ValueError:
                continue

            if label_first:
                labels.append(int(float(row[0])))
                feats.append([float(v) for v in row[1: num_inputs + 1]])
            else:
                labels.append(int(float(row[-1])))
                feats.append([float(v) for v in row[:num_inputs]])

    X = np.array(feats,  dtype=np.float64) / scale
    y = np.array(labels, dtype=np.int64)
    return X, y


def load_dataset(
    name: str = "mnist",
    data_dir: str = "./data",
    train_images: Optional[str] = None,
    train_labels: Optional[str] = None,
    test_images:  Optional[str] = None,
    test_labels:  Optional[str] = None,
    tra_path: Optional[str] = None,
    tes_path: Optional[str] = None,
    label_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset and return (X_train, y_train, X_test, y_test).

    Binary IDX mode (official MNIST format):
        load_dataset(
            name="mnist",
            train_images="data/train-images-idx3-ubyte",
            train_labels="data/train-labels-idx1-ubyte",
            test_images= "data/t10k-images-idx3-ubyte",
            test_labels= "data/t10k-labels-idx1-ubyte",
        )

    CSV mode (Kaggle MNIST, last column = label):
        load_dataset(
            name="mnist_csv",
            tra_path="data/mnist_train.csv",
            tes_path="data/mnist_test.csv",
        )

    CSV mode with the label in the first column:
        load_dataset(..., label_first=True)
    """
    if name in ("mnist", "mnist_idx"):
        ti = train_images or str(Path(data_dir) / "train-images-idx3-ubyte")
        tl = train_labels or str(Path(data_dir) / "train-labels-idx1-ubyte")
        ei = test_images  or str(Path(data_dir) / "t10k-images-idx3-ubyte")
        el = test_labels  or str(Path(data_dir) / "t10k-labels-idx1-ubyte")
        X_train, y_train = load_mnist_idx(ti, tl)
        X_test,  y_test  = load_mnist_idx(ei, el)

    elif name in ("mnist_csv", "mnist_kaggle"):
        tra = tra_path or str(Path(data_dir) / "mnist_train.csv")
        tes = tes_path or str(Path(data_dir) / "mnist_test.csv")
        X_train, y_train = _load_csv(tra, N_INPUTS_MNIST, SCALE_MNIST, label_first)
        X_test,  y_test  = _load_csv(tes, N_INPUTS_MNIST, SCALE_MNIST, label_first)

    elif name == "optdigits":
        tra = tra_path or str(Path(data_dir) / "optdigits.tra")
        tes = tes_path or str(Path(data_dir) / "optdigits.tes")
        X_train, y_train = _load_csv(tra, N_INPUTS_OPTDIGITS, SCALE_OPTDIGITS)
        X_test,  y_test  = _load_csv(tes, N_INPUTS_OPTDIGITS, SCALE_OPTDIGITS)

    else:
        raise ValueError(
            f"Dataset '{name}' no reconocido. "
            f"Usa: 'mnist', 'mnist_idx', 'mnist_csv', 'mnist_kaggle', 'optdigits'."
        )

    return X_train, y_train, X_test, y_test
