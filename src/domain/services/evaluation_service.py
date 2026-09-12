"""
domain/services/evaluation_service.py
======================================
Individual evaluation and Bullinaria-style triangular matrix generation.
"""
from __future__ import annotations

import csv
from typing import List, Optional, Tuple

import numpy as np

from src.domain.model.genotype import Genotype
from src.domain.model.mlp import MLP
from src.infrastructure.data.incremental_splitter import DEFAULT_N_SESSIONS, split_incremental


def evaluate_individual(
    g: Genotype,
    sessions: list,
    val: Tuple[np.ndarray, np.ndarray],
    use_dual: bool,
    max_epochs: int,
    rng,
    emit_fn=None,
    individual_idx: Optional[int] = None,
    gen: Optional[int] = None,
) -> float:
    """
    Train one genotype across the six incremental sessions and return
    validation accuracy as fitness.

    The caller decides whether emit_fn is None or the real event publisher,
    which keeps detailed dashboard events limited to verbose individuals.
    """
    net = MLP(g, use_dual=use_dual, rng=rng)
    for s_idx, (Xs, ys) in enumerate(sessions):
        net.train_session(
            Xs, ys,
            max_epochs=max_epochs,
            emit_fn=emit_fn,
            individual_idx=individual_idx,
            session_idx=s_idx,
            gen=gen,
        )
    return net.accuracy(val[0], val[1])


def evaluate_triangular(
    top_individuals: List[Genotype],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_dual: bool,
    max_epochs: int,
    master_rng,
    n_runs: int = 5,
    log_fn=None,
) -> Tuple[list, list, float, float]:
    """
    Evaluate the best genotype over independent runs and average results in
    the style of Bullinaria's Tables 3 and 4.
    """
    n_sessions = DEFAULT_N_SESSIONS
    g          = top_individuals[0]
    all_matrices, all_s_accs, all_t_accs = [], [], []

    for run in range(n_runs):
        run_rng  = np.random.default_rng(master_rng.integers(0, 2**31))
        net_rng  = np.random.default_rng(master_rng.integers(0, 2**31))
        sessions, _ = split_incremental(X_train, y_train, rng=run_rng)

        net    = MLP(g, use_dual=use_dual, rng=net_rng)
        matrix = [[None] * n_sessions for _ in range(n_sessions)]
        s_accs = []

        for s_idx, (Xs, ys) in enumerate(sessions):
            net.train_session(Xs, ys, max_epochs=max_epochs)
            for b_idx in range(s_idx + 1):
                Xb, yb       = sessions[b_idx]
                matrix[s_idx][b_idx] = round(net.accuracy(Xb, yb) * 100, 2)
            s_accs.append(round(net.accuracy(X_test, y_test) * 100, 2))

        all_matrices.append(matrix)
        all_s_accs.append(s_accs)
        all_t_accs.append(net.accuracy(X_test, y_test) * 100)

        if log_fn:
            log_fn(f"  [Triangular run {run+1}/{n_runs}] T6_test={s_accs[-1]:.2f}%")

    avg = [[None] * n_sessions for _ in range(n_sessions)]
    for s in range(n_sessions):
        for b in range(n_sessions):
            vals = [
                all_matrices[r][s][b]
                for r in range(n_runs)
                if all_matrices[r][s][b] is not None
            ]
            if vals:
                avg[s][b] = round(float(np.mean(vals)), 2)

    avg_s_accs = [
        round(float(np.mean([all_s_accs[r][s] for r in range(n_runs)])), 2)
        for s in range(n_sessions)
    ]
    mean_acc = round(float(np.mean(all_t_accs)), 2)
    std_acc  = round(float(np.std(all_t_accs)),  2)

    return avg, avg_s_accs, mean_acc, std_acc


def print_triangular_matrix(
    matrix: list,
    session_accs: list,
    mean_acc: float,
    std_acc: float,
    label: str = "fw²",
) -> str:
    n   = len(matrix)
    sep = "─" * 74
    hdr = f"  {'':8s}" + "".join(f"  {'B'+str(i+1):>8s}" for i in range(n))
    lines = [
        "",
        "=" * 74,
        f"  MATRIZ TRIANGULAR — {label}",
        "  Porcentaje de clasificación correcta",
        "  Equivalente a Tablas 3/4 de Bullinaria (2009)",
        "=" * 74,
        hdr,
        f"  {sep}",
    ]
    for s in range(n):
        row = f"  T{str(s+1):<7s}"
        for b in range(n):
            v    = matrix[s][b]
            row += f"  {v:>8.2f}" if v is not None else f"  {'--':>8s}"
        lines.append(row)
    lines.append(f"  {sep}")
    test_row = (
        f"  {'Test':<8s}"
        + "".join(f"  {a:>8.2f}" for a in session_accs)
    )
    lines += [
        test_row,
        f"  {sep}",
        f"  Media (top 10%): {mean_acc:.2f}% ± {std_acc:.2f}%",
        f"  Referencia Bullinaria fw¹:  95.07% ± 0.04%",
        "=" * 74,
        "",
    ]
    output = "\n".join(lines)
    print(output, flush=True)
    return output


def save_triangular_csv(
    matrix: list,
    session_accs: list,
    mean_acc: float,
    std_acc: float,
    path: str,
    label: str,
) -> None:
    n = len(matrix)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Matriz Triangular — {label}"])
        w.writerow([f"# mean={mean_acc:.2f}% std={std_acc:.2f}%"])
        w.writerow(["Sesion"] + [f"B{i+1}" for i in range(n)])
        for s in range(n):
            w.writerow(
                [f"T{s+1}"]
                + [
                    f"{matrix[s][b]:.2f}" if matrix[s][b] is not None else ""
                    for b in range(n)
                ]
            )
        w.writerow(["Test"] + [f"{a:.2f}" for a in session_accs])
        w.writerow(["mean_acc", mean_acc])
        w.writerow(["std_acc",  std_acc])


def compute_confusion(
    net: MLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    preds = net.predict_all(X_test)
    cm    = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(y_test, preds):
        cm[true, pred] += 1
    return cm


def save_confusion_csv(cm: np.ndarray, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"Pred_{i}" for i in range(cm.shape[1])])
        for i, row in enumerate(cm):
            w.writerow([f"True_{i}"] + list(row))
