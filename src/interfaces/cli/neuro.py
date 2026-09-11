#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI runner for the fw3 experiment: w + fw1 + fw2 + fw3.

Quick run:
    python run.py cli

Full replica budget:
    python run.py cli --full

Parallel workers are available only outside dashboard mode:
    python run.py cli --full --workers 0
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
        description="fw³ Neuroevolución — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset",  default="optdigits", choices=["optdigits"])
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--train",    default=None)
    ap.add_argument("--test",     default=None)

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

    ap.add_argument("--out-prefix", default="fw3")
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

    X_train, y_train, X_test, y_test = load_dataset(
        args.dataset, data_dir=args.data_dir,
        tra_path=args.train, tes_path=args.test,
    )
    n_in  = X_train.shape[1]
    n_out = len(np.unique(y_train))

    log_fn(f"Train: {len(X_train)} | Test: {len(X_test)}")
    log_fn(
        f"Gen={generations} | Pop={pop_size} | MaxEpochs={max_epochs} | "
        f"use_dual={not args.no_dual} | seed={args.seed} | workers={args.workers}"
    )

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
