#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — fw³-MNIST  (w + fw¹ + fw² + fw³)
==========================================

Uso:
  python run.py cli    [opciones]
  python run.py server [opciones]

Binary IDX format (official MNIST):
  python run.py cli \\
      --train-images data/train-images-idx3-ubyte \\
      --train-labels data/train-labels-idx1-ubyte \\
      --test-images  data/t10k-images-idx3-ubyte  \\
      --test-labels  data/t10k-labels-idx1-ubyte

CSV format (Kaggle MNIST, last column = label):
  python run.py cli --csv --train data/mnist_train.csv --test data/mnist_test.csv

Full replication:
  python run.py cli --full
  python run.py cli --full --workers 0    # multicore

Dashboard:
  python run.py server
  python run.py server --train-images data/... --train-labels data/...
"""

import os
import sys

# Defaults for the standard MNIST IDX directory layout.
CLI_DEFAULTS_IDX = (
    "--train-images data/train-images-idx3-ubyte "
    "--train-labels data/train-labels-idx1-ubyte "
    "--test-images  data/t10k-images-idx3-ubyte  "
    "--test-labels  data/t10k-labels-idx1-ubyte  "
    "--generations 10 --pop-size 20 --max-epochs 50"
)

SERVER_DEFAULTS_IDX = (
    "--train-images data/train-images-idx3-ubyte "
    "--train-labels data/train-labels-idx1-ubyte "
    "--test-images  data/t10k-images-idx3-ubyte  "
    "--test-labels  data/t10k-labels-idx1-ubyte  "
    "--generations 50 --pop-size 100 --max-epochs 500"
)


def run_server(extra_args: str = ""):
    os.system("PYTHONPATH=. python3 src/interfaces/http/server.py " + extra_args)


def run_cli(extra_args: str = ""):
    os.system("PYTHONPATH=. python3 src/interfaces/cli/neuro.py " + extra_args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode  = sys.argv[1]
    extra = " ".join(sys.argv[2:])

    if mode == "server":
        run_server(extra or SERVER_DEFAULTS_IDX)
    elif mode == "cli":
        run_cli(extra or CLI_DEFAULTS_IDX)
    else:
        print(f"Modo desconocido: '{mode}'  —  usa 'server' o 'cli'")
        sys.exit(1)
