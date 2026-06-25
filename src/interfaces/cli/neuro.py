#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interfaces/cli/neuro.py
========================
CLI compacta para el experimento fw² (w + fw¹ + fw²).

Uso rápido (50 gen, pop=50):
    python run.py cli

Réplica completa (1800 gen, pop=100):
    python run.py cli --full

Con parámetros explícitos:
    python run.py cli --generations 100 --pop-size 50 --max-epochs 500

Solo pesos base (sin fast-weights):
    python run.py cli --no-dual
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from src.infrastructure.data.dataset_loader import load_dataset
from src.infrastructure.events.stdout_event_publisher import (
    EventPublisher,
    NeuroLogger,
    make_log_fns,
)
from src.domain.services.evolution_service import run_evolution


def main():
    ap = argparse.ArgumentParser(
        description="fw² Neuroevolución — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    ap.add_argument("--dataset",  default="optdigits",
                    choices=["optdigits"],
                    help="Dataset a usar")
    ap.add_argument("--data-dir", default="./data",
                    help="Directorio raíz de los datos")
    ap.add_argument("--train",    default=None,
                    help="Ruta explícita a optdigits.tra")
    ap.add_argument("--test",     default=None,
                    help="Ruta explícita a optdigits.tes")

    # ── Evolución ─────────────────────────────────────────────────────────────
    ap.add_argument("--generations",   type=int,   default=50)
    ap.add_argument("--full",          action="store_true",
                    help="Budget completo: 1800 gen / pop=100 / epochs=5000")
    ap.add_argument("--pop-size",      type=int,   default=50)
    ap.add_argument("--max-epochs",    type=int,   default=500)
    ap.add_argument("--mutation-std",  type=float, default=0.05)
    ap.add_argument("--top-frac",      type=float, default=0.10)
    ap.add_argument("--no-dual",       action="store_true",
                    help="Deshabilita los fast-weights (solo pesos base w)")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--verbose-indiv", type=int,   default=3,
                    help="Cuántos individuos por gen reciben log detallado")

    # ── Salida ────────────────────────────────────────────────────────────────
    ap.add_argument("--out-prefix",  default="fw2",
                    help="Prefijo para archivos de salida")
    ap.add_argument("--tri-runs",    type=int, default=5,
                    help="Nº de ejecuciones para la matriz triangular")
    ap.add_argument("--dir-data",    default="result",
                    help="Carpeta para CSVs y archivos de texto")
    ap.add_argument("--dir-plots",   default="plots",
                    help="Carpeta para imágenes PNG")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    ap.add_argument("--dashboard",   action="store_true",
                    help="Activa emisión de eventos JSON para el dashboard SSE")

    args = ap.parse_args()

    # ── Budget ────────────────────────────────────────────────────────────────
    if args.full:
        generations = 1800
        pop_size    = 100
        max_epochs  = 5000
    else:
        generations = args.generations
        pop_size    = args.pop_size
        max_epochs  = args.max_epochs

    # ── Carpetas y logger ─────────────────────────────────────────────────────
    os.makedirs(args.dir_data,  exist_ok=True)
    os.makedirs(args.dir_plots, exist_ok=True)
    log_path = os.path.join(args.dir_data, f"{args.out_prefix}_run.log")
    logger   = NeuroLogger(log_path)
    log_fn, log_detail_fn = make_log_fns(logger)

    # ── Publisher ─────────────────────────────────────────────────────────────
    publisher = EventPublisher(dashboard_mode=args.dashboard)

    # ── Datos ─────────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        tra_path=args.train,
        tes_path=args.test,
    )
    n_in  = X_train.shape[1]
    n_out = len(np.unique(y_train))

    log_fn(f"Train: {len(X_train)} | Test: {len(X_test)}")
    log_fn(
        f"Gen={generations} | Pop={pop_size} | MaxEpochs={max_epochs} | "
        f"use_dual={not args.no_dual} | seed={args.seed}"
    )
    log_fn(f"Salida → datos: {args.dir_data}/   gráficas: {args.dir_plots}/")
    log_fn(f"Log completo:   {log_path}")

    # ── Evolución ─────────────────────────────────────────────────────────────
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
        publisher           = publisher,
        log_fn              = log_fn,
        log_detail_fn       = log_detail_fn,
    )

    logger.close()


if __name__ == "__main__":
    main()
