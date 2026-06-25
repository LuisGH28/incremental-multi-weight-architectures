"""
infrastructure/data/dataset_loader.py
=======================================
Cargador de datos para el experimento fw³-MNIST.

Soporta dos formatos:
  1. IDX binario — formato oficial MNIST/Yann LeCun (.ubyte, .gz)
  2. CSV         — una fila por patrón, última columna = etiqueta, valores 0-255

Justificación de las constantes:
  - n_inputs = 784  (28×28 píxeles, frente a 64 de OptDigits 8×8)
  - scale    = 255.0 (frente a 16.0 de OptDigits)
  Ambas normalizaciones llevan el espacio de entrada a [0, 1].
  La arquitectura del MLP y el protocolo evolutivo no cambian.
"""
from __future__ import annotations

import csv
import gzip
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Dimensiones del dataset
N_INPUTS_MNIST     = 784
N_INPUTS_OPTDIGITS = 64
SCALE_MNIST        = 255.0
SCALE_OPTDIGITS    = 16.0


# ─────────────────────────────────────────────────────────────────────────────
# IDX binario (formato oficial MNIST)
# ─────────────────────────────────────────────────────────────────────────────

def _open_idx(path: str):
    """Abre un archivo IDX, con o sin compresión gzip."""
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
    Carga MNIST en formato IDX binario.

    Parámetros
    ----------
    images_path : ruta al archivo de imágenes (.ubyte o .ubyte.gz)
    labels_path : ruta al archivo de etiquetas (.ubyte o .ubyte.gz)

    Devuelve
    --------
    X : float64 array de forma (N, 784), valores en [0, 1]
    y : int64   array de forma (N,),     valores en {0, …, 9}
    """
    X = _load_idx_images(images_path)
    y = _load_idx_labels(labels_path)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(
    path: str,
    num_inputs: int,
    scale: float,
    label_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga un CSV donde:
      - label_first=False : última columna = etiqueta  (formato OptDigits / Kaggle MNIST)
      - label_first=True  : primera columna = etiqueta (formato MNIST CSV alternativo)
    """
    feats, labels = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                float(row[0])
            except ValueError:
                continue   # saltar encabezado

            if label_first:
                labels.append(int(float(row[0])))
                feats.append([float(v) for v in row[1: num_inputs + 1]])
            else:
                labels.append(int(float(row[-1])))
                feats.append([float(v) for v in row[:num_inputs]])

    X = np.array(feats,  dtype=np.float64) / scale
    y = np.array(labels, dtype=np.int64)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada público
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    name: str = "mnist",
    data_dir: str = "./data",
    # ── Rutas explícitas ──────────────────────────────────────────────────────
    train_images: Optional[str] = None,
    train_labels: Optional[str] = None,
    test_images:  Optional[str] = None,
    test_labels:  Optional[str] = None,
    # ── Rutas CSV (optdigits-style o Kaggle-MNIST) ────────────────────────────
    tra_path: Optional[str] = None,
    tes_path: Optional[str] = None,
    label_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga el dataset y devuelve (X_train, y_train, X_test, y_test).

    Modo IDX binario (formato oficial MNIST):
        load_dataset(
            name="mnist",
            train_images="data/train-images-idx3-ubyte",
            train_labels="data/train-labels-idx1-ubyte",
            test_images= "data/t10k-images-idx3-ubyte",
            test_labels= "data/t10k-labels-idx1-ubyte",
        )

    Modo CSV (Kaggle MNIST, última columna = etiqueta):
        load_dataset(
            name="mnist_csv",
            tra_path="data/mnist_train.csv",
            tes_path="data/mnist_test.csv",
        )

    Modo CSV etiqueta en primera columna:
        load_dataset(..., label_first=True)
    """
    if name in ("mnist", "mnist_idx"):
        # ── IDX binario ───────────────────────────────────────────────────────
        ti = train_images or str(Path(data_dir) / "train-images-idx3-ubyte")
        tl = train_labels or str(Path(data_dir) / "train-labels-idx1-ubyte")
        ei = test_images  or str(Path(data_dir) / "t10k-images-idx3-ubyte")
        el = test_labels  or str(Path(data_dir) / "t10k-labels-idx1-ubyte")
        X_train, y_train = load_mnist_idx(ti, tl)
        X_test,  y_test  = load_mnist_idx(ei, el)

    elif name in ("mnist_csv", "mnist_kaggle"):
        # ── CSV con 784 píxeles ───────────────────────────────────────────────
        tra = tra_path or str(Path(data_dir) / "mnist_train.csv")
        tes = tes_path or str(Path(data_dir) / "mnist_test.csv")
        X_train, y_train = _load_csv(tra, N_INPUTS_MNIST, SCALE_MNIST, label_first)
        X_test,  y_test  = _load_csv(tes, N_INPUTS_MNIST, SCALE_MNIST, label_first)

    elif name == "optdigits":
        # ── OptDigits CSV (compatibilidad) ────────────────────────────────────
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
