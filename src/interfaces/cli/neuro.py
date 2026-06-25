#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interfaces/cli/neuro.py  —  fw³-MNIST (w + fw¹ + fw² + fw³)

Uso con IDX binario (formato oficial MNIST):
    python run.py cli \\
        --train-images data/train-images-idx3-ubyte \\
        --train-labels data/train-labels-idx1-ubyte \\
        --test-images  data/t10k-images-idx3-ubyte  \\
        --test-labels  data/t10k-labels-idx1-ubyte

Uso con CSV (Kaggle MNIST, última col = etiqueta):
    python run.py cli --csv \\
        --train data/mnist_train.csv \\
        --test  data/mnist_test.csv

Uso con CSV (primera col = etiqueta):
    python run.py cli --csv --label-first \\
        --train data/mnist_train.csv --test data/mnist_test.csv

Réplica completa:
    python run.py cli --full
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.infrastructure.data.dataset_loader import load_dataset
from src.infrastructure.events.stdout_event_publisher import (
    EventPublisher, NeuroLogger, make_log_fns,
)
from src.domain.services.evolution_service import run_evolution


def main():
    ap = argparse.ArgumentParser(
        description="fw³-MNIST Neuroevolución — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Modo de entrada ───────────────────────────────────────────────────────
    mode_grp = ap.add_argument_group("Formato de datos")
    mode_grp.add_argument("--csv", action="store_true",
                          help="Leer CSV en lugar de IDX binario")
    mode_grp.add_argument("--label-first", action="store_true",
                          help="En CSV, la primera columna es la etiqueta (no la última)")

    # ── Rutas IDX ─────────────────────────────────────────────────────────────
    idx_grp = ap.add_argument_group("Rutas IDX binario (formato oficial MNIST)")
    idx_grp.add_argument("--train-images", default=None,
                         help="Archivo de imágenes de entrenamiento (.ubyte/.gz)")
    idx_grp.add_argument("--train-labels", default=None,
                         help="Archivo de etiquetas de entrenamiento (.ubyte/.gz)")
    idx_grp.add_argument("--test-images",  default=None,
                         help="Archivo de imágenes de prueba (.ubyte/.gz)")
    idx_grp.add_argument("--test-labels",  default=None,
                         help="Archivo de etiquetas de prueba (.ubyte/.gz)")

    # ── Rutas CSV ─────────────────────────────────────────────────────────────
    csv_grp = ap.add_argument_group("Rutas CSV")
    csv_grp.add_argument("--train", default=None, help="CSV de entrenamiento")
    csv_grp.add_argument("--test",  default=None, help="CSV de prueba")

    # ── Directorio raíz (alternativa a rutas explícitas) ─────────────────────
    ap.add_argument("--data-dir", default="./data",
                    help="Directorio raíz donde buscar los archivos MNIST")

    # ── Evolución ─────────────────────────────────────────────────────────────
    ap.add_argument("--generations",   type=int,   default=50)
    ap.add_argument("--full",          action="store_true",
                    help="1800 gen / pop=100 / epochs=5000")
    ap.add_argument("--pop-size",      type=int,   default=50)
    ap.add_argument("--max-epochs",    type=int,   default=500)
    ap.add_argument("--mutation-std",  type=float, default=0.05)
    ap.add_argument("--top-frac",      type=float, default=0.10)
    ap.add_argument("--no-dual",       action="store_true")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--verbose-indiv", type=int,   default=3)
    ap.add_argument("--workers",       type=int,   default=0,
                    help="CPUs paralelas (0=auto). Desactivado en modo --dashboard")

    # ── Salida ────────────────────────────────────────────────────────────────
    ap.add_argument("--out-prefix", default="fw3_mnist")
    ap.add_argument("--tri-runs",   type=int, default=5)
    ap.add_argument("--dir-data",   default="result")
    ap.add_argument("--dir-plots",  default="plots")
    ap.add_argument("--dashboard",  action="store_true")

    args = ap.parse_args()

    if args.full:
        generations = 1800; pop_size = 100; max_epochs = 5000
    else:
        generations = args.generations
        pop_size    = args.pop_size
        max_epochs  = args.max_epochs

    os.makedirs(args.dir_data,  exist_ok=True)
    os.makedirs(args.dir_plots, exist_ok=True)
    log_path = os.path.join(args.dir_data, f"{args.out_prefix}_run.log")
    logger   = NeuroLogger(log_path)
    log_fn, log_detail_fn = make_log_fns(logger)
    publisher = EventPublisher(dashboard_mode=args.dashboard)

    # ── Carga de datos ────────────────────────────────────────────────────────
    if args.csv:
        log_fn("Formato: CSV")
        X_train, y_train, X_test, y_test = load_dataset(
            name="mnist_csv",
            data_dir=args.data_dir,
            tra_path=args.train,
            tes_path=args.test,
            label_first=args.label_first,
        )
    else:
        log_fn("Formato: IDX binario (MNIST oficial)")
        X_train, y_train, X_test, y_test = load_dataset(
            name="mnist",
            data_dir=args.data_dir,
            train_images=args.train_images,
            train_labels=args.train_labels,
            test_images=args.test_images,
            test_labels=args.test_labels,
        )

    n_in  = X_train.shape[1]
    n_out = len(np.unique(y_train))
    log_fn(f"Dataset: {n_in} entradas | Train={len(X_train)} | Test={len(X_test)}")
    log_fn(f"Gen={generations} | Pop={pop_size} | MaxEpochs={max_epochs} | "
           f"use_dual={not args.no_dual} | seed={args.seed} | workers={args.workers}")

    run_evolution(
        X_train, y_train, X_test, y_test,
        pop_size            = pop_size,
        n_generations       = generations,
        use_dual            = not args.no_dual,
        max_epochs          = max_epochs,
        mutation_std        = args.mutation_std,
        top_frac            = args.top_frac,
        seed                = args.seed,
        verbose_individuals = args.verbose_indiv,
        n_in                = n_in,
        n_out               = n_out,
        out_prefix          = args.out_prefix,
        tri_runs            = args.tri_runs,
        dir_data            = args.dir_data,
        dir_plots           = args.dir_plots,
        n_workers           = args.workers,
        publisher           = publisher,
        log_fn              = log_fn,
        log_detail_fn       = log_detail_fn,
    )

    logger.close()


if __name__ == "__main__":
    main()
