"""
domain/services/evolution_service.py
======================================
Bucle principal de neuroevolución para la arquitectura fw²
(w + fw¹ + fw²).

Replica el esquema de Bullinaria (2009):
  - Población de 100 redes
  - Cruce con rango + mutación gaussiana
  - Los 50% mejores sobreviven y generan un hijo cada uno
  - Distintos splits de entrenamiento/validación en cada generación

Extensión propuesta:
  - El genotipo incluye (δ₂, σ₂) para una segunda línea de fast-weights
  - Todos los demás parámetros y el protocolo son idénticos a fw¹
"""
from __future__ import annotations

import csv
import os
import time
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np

from src.domain.model.genotype import Genotype
from src.domain.model.mlp import MLP
from src.domain.services.evaluation_service import (
    evaluate_individual,
    evaluate_triangular,
    print_triangular_matrix,
    save_triangular_csv,
    compute_confusion,
    save_confusion_csv,
)
from src.infrastructure.data.incremental_splitter import split_incremental
from src.infrastructure.events.stdout_event_publisher import EventPublisher


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _elapsed_str(seconds: float) -> str:
    td   = timedelta(seconds=int(seconds))
    days = td.days
    h    = td.seconds // 3600
    m    = (td.seconds % 3600) // 60
    s    = td.seconds % 60
    parts = []
    if days: parts.append(f"{days}d")
    if h:    parts.append(f"{h}h")
    if m:    parts.append(f"{m}min")
    parts.append(f"{s}s")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Historial de evolución (para gráficas y CSV de progreso)
# ─────────────────────────────────────────────────────────────────────────────

class EvoHistory:
    def __init__(self):
        self.gens:          List[int]   = []
        self.con_ih:        List[float] = []
        self.con_ho:        List[float] = []
        self.log_eta_ih:    List[float] = []
        self.log_eta_hb:    List[float] = []
        self.log_eta_ho:    List[float] = []
        self.log_eta_ob:    List[float] = []
        self.log_lam:       List[float] = []
        self.log_ospo:      List[float] = []
        self.log_fw1_decay: List[float] = []
        self.log_fw1_scale: List[float] = []
        self.log_fw2_decay: List[float] = []
        self.log_fw2_scale: List[float] = []
        self.tol_t:         List[float] = []
        self.tol_s:         List[float] = []
        self.val_acc:       List[float] = []
        self.test_acc:      List[float] = []

    def record(
        self,
        gen: int,
        survivors: List[Genotype],
        top_fitness: float,
        test_acc_val: Optional[float] = None,
    ) -> None:
        eps = 1e-30

        def mean(fn):  return float(np.mean([fn(g) for g in survivors]))
        def lmean(fn): return float(np.mean([np.log10(max(fn(g), eps)) for g in survivors]))

        self.gens.append(gen)
        self.con_ih.append(mean(lambda g: g.c_ih))
        self.con_ho.append(mean(lambda g: g.c_ho))
        self.log_eta_ih.append(lmean(lambda g: g.eta_ih))
        self.log_eta_hb.append(lmean(lambda g: g.eta_hb))
        self.log_eta_ho.append(lmean(lambda g: g.eta_ho))
        self.log_eta_ob.append(lmean(lambda g: g.eta_ob))
        self.log_lam.append(lmean(lambda g: g.lam))
        self.log_ospo.append(lmean(lambda g: g.ospo))
        self.log_fw1_decay.append(lmean(lambda g: g.fw_decay))
        self.log_fw1_scale.append(lmean(lambda g: g.fw_scale))
        self.log_fw2_decay.append(lmean(lambda g: g.fw2_decay))
        self.log_fw2_scale.append(lmean(lambda g: g.fw2_scale))
        self.tol_t.append(mean(lambda g: g.tol_t))
        self.tol_s.append(mean(lambda g: g.tol_s))
        self.val_acc.append(top_fitness * 100.0)
        self.test_acc.append(
            test_acc_val * 100.0 if test_acc_val is not None else float("nan")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Bucle principal
# ─────────────────────────────────────────────────────────────────────────────

def run_evolution(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    # ── Hiperparámetros evolutivos ─────────────────────────────────────────
    pop_size:            int   = 100,
    n_generations:       int   = 50,
    use_dual:            bool  = True,
    max_epochs:          int   = 500,
    mutation_std:        float = 0.05,
    top_frac:            float = 0.10,
    seed:                int   = 42,
    verbose_individuals: int   = 3,
    n_in:                int   = 64,
    n_out:               int   = 10,
    # ── Salida ────────────────────────────────────────────────────────────
    out_prefix:  str = "fw2",
    tri_runs:    int = 5,
    dir_data:    str = "result",
    dir_plots:   str = "plots",
    # ── Publisher de eventos (dashboard) ──────────────────────────────────
    publisher: Optional[EventPublisher] = None,
    log_fn=None,
    log_detail_fn=None,
):
    """
    Ejecuta la neuroevolución para fw².

    publisher  — instancia de EventPublisher o None (sin dashboard)
    log_fn     — callable(str) para mensajes de terminal/log
    log_detail_fn — callable(str) para mensajes sólo al archivo .log
    """
    run_start = time.time()

    if log_fn        is None: log_fn        = print
    if log_detail_fn is None: log_detail_fn = lambda _: None

    def emit(event_type: str, **kwargs):
        if publisher:
            publisher.emit(event_type, **kwargs)

    label = (
        "fw² — w + fw¹ + fw² (extensión propuesta)"
        if use_dual else "MLP sin fast-weights"
    )

    master_rng = np.random.default_rng(seed)
    history    = EvoHistory()

    os.makedirs(dir_data,  exist_ok=True)
    os.makedirs(dir_plots, exist_ok=True)

    # ── CSV de progreso por generación ────────────────────────────────────────
    csv_progress_path = os.path.join(dir_data, f"{out_prefix}_progress.csv")
    csv_file = open(csv_progress_path, "w", newline="")
    cw = csv.writer(csv_file)
    cw.writerow([
        "gen", "best_fitness_val",
        "T1_test", "T2_test", "T3_test", "T4_test", "T5_test", "T6_test",
        "elapsed_s",
    ])
    csv_file.flush()

    emit("config",
         pop_size=pop_size, n_generations=n_generations,
         use_dual=use_dual, max_epochs=max_epochs,
         n_train=len(X_train), n_test=len(X_test))

    population = [Genotype.random(master_rng) for _ in range(pop_size)]
    fitness    = np.zeros(pop_size)

    log_fn("=" * 60)
    log_fn(f"INICIO EVOLUCIÓN — {label}")
    log_fn(f"Budget: {n_generations} gen | pop={pop_size} | max_epochs={max_epochs}")
    log_fn("=" * 60)

    # ════════════════════════════════════════════════════════════════════
    #  BUCLE GENERACIONAL
    # ════════════════════════════════════════════════════════════════════
    for gen in range(n_generations):
        t0      = time.time()
        gen_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        sessions, val = split_incremental(X_train, y_train, rng=gen_rng)

        emit("gen_start", gen=gen, n_gen=n_generations, pop_size=pop_size)

        for i, g in enumerate(population):
            eval_rng = np.random.default_rng(master_rng.integers(0, 2**31))
            use_emit = (i < verbose_individuals)

            fitness[i] = evaluate_individual(
                g, sessions, val, use_dual, max_epochs, eval_rng,
                emit_fn=emit if use_emit else None,
                individual_idx=i if use_emit else None,
                gen=gen if use_emit else None,
            )
            emit("individual_done",
                 gen=gen, individual=i,
                 fitness=round(float(fitness[i]), 4))

        # ── Selección ──────────────────────────────────────────────────────
        ranked   = np.argsort(fitness)[::-1]
        best_idx = ranked[0]
        best_g   = population[best_idx]
        top_n    = max(1, pop_size // 10)
        top_fit  = float(fitness[ranked[:top_n]].mean())

        # Test rápido del mejor para el dashboard / CSV
        tst_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        tst_ses, _ = split_incremental(
            X_train, y_train,
            rng=np.random.default_rng(master_rng.integers(0, 2**31))
        )
        tst_net = MLP(best_g, use_dual=use_dual, rng=tst_rng)
        t_accs  = []
        for Xs, ys in tst_ses:
            tst_net.train_session(Xs, ys, max_epochs=max_epochs)
            t_accs.append(round(tst_net.accuracy(X_test, y_test) * 100, 2))
        quick_test = tst_net.accuracy(X_test, y_test)

        survivors = [population[i] for i in ranked[: pop_size // 2]]
        history.record(gen, survivors, top_fit, quick_test)

        elapsed_gen = time.time() - t0
        elapsed_tot = time.time() - run_start

        while len(t_accs) < 6:
            t_accs.append("")
        cw.writerow([
            gen,
            round(float(fitness[best_idx]) * 100, 2),
            *t_accs[:6],
            round(elapsed_gen, 1),
        ])
        csv_file.flush()

        log_fn(
            f"[GEN {gen:3d}] best={fitness[best_idx]*100:.2f}%  "
            f"top10={top_fit*100:.2f}%  T6={t_accs[5]}%  "
            f"gen={_elapsed_str(elapsed_gen)}  total={_elapsed_str(elapsed_tot)}"
        )
        log_detail_fn(f"  Mejor genotipo gen {gen}: {best_g.summary()}")
        log_detail_fn(
            f"  Fitness dist: min={fitness.min()*100:.2f}% "
            f"mean={fitness.mean()*100:.2f}% max={fitness.max()*100:.2f}%"
        )

        emit("gen_end",
             gen=gen,
             best_fitness=round(float(fitness[best_idx]), 4),
             top10_mean=round(top_fit, 4),
             pop_mean=round(float(fitness.mean()), 4),
             elapsed=round(elapsed_gen, 2),
             best_genotype=best_g.to_dict(),
             fitness_all=[round(float(f), 4) for f in fitness[ranked]])

        # ── Reproducción ───────────────────────────────────────────────────
        parent_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        children = [
            p.crossover_mutate(
                survivors[parent_rng.integers(0, len(survivors))],
                mutation_std, parent_rng,
            )
            for p in survivors
        ]
        population = survivors + children

    # ════════════════════════════════════════════════════════════════════
    #  EVALUACIÓN FINAL
    # ════════════════════════════════════════════════════════════════════
    log_fn("\n[Evaluación final sobre test set...]")
    emit("final_start")

    final_rng = np.random.default_rng(master_rng.integers(0, 2**31))
    final_sessions, final_val = split_incremental(X_train, y_train, rng=final_rng)
    for i, g in enumerate(population):
        er  = np.random.default_rng(master_rng.integers(0, 2**31))
        net = MLP(g, use_dual=use_dual, rng=er)
        for Xs, ys in final_sessions:
            net.train_session(Xs, ys, max_epochs=max_epochs)
        fitness[i] = net.accuracy(final_val[0], final_val[1])

    ranked     = np.argsort(fitness)[::-1]
    top_n      = max(1, int(top_frac * pop_size))
    top_indivs = [population[i] for i in ranked[:top_n]]

    test_accs    = []
    session_accs = []
    for g in top_indivs:
        ts_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        ts_ses, _ = split_incremental(X_train, y_train, rng=ts_rng)
        net = MLP(
            g, use_dual=use_dual,
            rng=np.random.default_rng(master_rng.integers(0, 2**31)),
        )
        s_accs = []
        for Xs, ys in ts_ses:
            net.train_session(Xs, ys, max_epochs=max_epochs)
            s_accs.append(round(net.accuracy(X_test, y_test) * 100, 2))
        session_accs.append(s_accs)
        test_accs.append(net.accuracy(X_test, y_test))

    mean_acc = float(np.mean(test_accs)) * 100
    std_acc  = float(np.std(test_accs))  * 100

    # Matriz de confusión del mejor individuo
    cm_net   = MLP(
        top_indivs[0], use_dual=use_dual,
        rng=np.random.default_rng(master_rng.integers(0, 2**31)),
    )
    final_ts, _ = split_incremental(
        X_train, y_train,
        rng=np.random.default_rng(master_rng.integers(0, 2**31)),
    )
    for Xs, ys in final_ts:
        cm_net.train_session(Xs, ys, max_epochs=max_epochs)
    cm = compute_confusion(cm_net, X_test, y_test)

    total_elapsed = time.time() - run_start

    log_fn(f"\n{'='*60}")
    log_fn("RESULTADO FINAL")
    log_fn(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    log_fn("  Referencia Bullinaria: 95.07% ± 0.04%")
    log_fn(f"  Tiempo total: {_elapsed_str(total_elapsed)}")
    log_fn(f"{'='*60}")

    ft = session_accs[0] if session_accs else [""] * 6
    while len(ft) < 6:
        ft.append("")
    cw.writerow(["FINAL", round(mean_acc, 2), *ft[:6], round(total_elapsed, 1)])
    csv_file.flush()
    csv_file.close()

    emit("final_result",
         mean_acc=round(mean_acc, 2), std_acc=round(std_acc, 2),
         session_accs=session_accs[0] if session_accs else [],
         best_genotype=top_indivs[0].to_dict(), target_acc=95.07)

    # ── Matriz triangular ─────────────────────────────────────────────────────
    log_fn(f"\n[Generando matriz triangular ({tri_runs} runs)...]")
    tri_matrix, tri_s_accs, tri_mean, tri_std = evaluate_triangular(
        top_indivs, X_train, y_train, X_test, y_test,
        use_dual=use_dual, max_epochs=max_epochs,
        master_rng=master_rng, n_runs=tri_runs,
        log_fn=log_fn,
    )

    print_triangular_matrix(tri_matrix, tri_s_accs, tri_mean, tri_std, label=label)

    tri_csv = os.path.join(dir_data, f"{out_prefix}_triangular.csv")
    tri_txt = os.path.join(dir_data, f"{out_prefix}_triangular.txt")
    save_triangular_csv(tri_matrix, tri_s_accs, tri_mean, tri_std,
                        path=tri_csv, label=label)

    txt_lines = [
        f"MATRIZ TRIANGULAR — {label}\n",
        f"{'Sesion':<8s}" + "".join(f"{'B'+str(i+1):>9s}" for i in range(6)) + "\n",
    ]
    for s in range(6):
        row = f"T{str(s+1):<7s}"
        for b in range(6):
            v    = tri_matrix[s][b]
            row += f"{v:>9.2f}" if v is not None else f"{'--':>9s}"
        txt_lines.append(row + "\n")
    txt_lines += [
        f"{'Test':<8s}" + "".join(f"{a:>9.2f}" for a in tri_s_accs) + "\n",
        f"\nMedia: {tri_mean:.2f}% ± {tri_std:.2f}%\n",
        "Referencia Bullinaria fw¹: 95.07% ± 0.04%\n",
        f"Tiempo total: {_elapsed_str(total_elapsed)}\n",
    ]
    with open(tri_txt, "w") as f:
        f.writelines(txt_lines)

    # ── Matriz de confusión ───────────────────────────────────────────────────
    cm_csv = os.path.join(dir_data, f"{out_prefix}_confusion.csv")
    save_confusion_csv(cm, cm_csv)

    # ── Gráficas (importación diferida — matplotlib opcional) ─────────────────
    log_fn("\n[Generando gráficas...]")
    try:
        from src.shared.utils.plots import (
            plot_evolution,
            plot_triangular_heatmap,
            plot_confusion,
        )
        plot_evolution(
            history, label=label,
            out_path=os.path.join(dir_plots, f"{out_prefix}_evolution.png"),
            log_fn=log_fn,
        )
        plot_triangular_heatmap(
            tri_matrix, tri_s_accs, tri_mean, tri_std, label=label,
            out_path=os.path.join(dir_plots, f"{out_prefix}_heatmap.png"),
            log_fn=log_fn,
        )
        plot_confusion(
            cm, mean_acc, label=label,
            out_path=os.path.join(dir_plots, f"{out_prefix}_confusion.png"),
            log_fn=log_fn,
        )
    except ImportError:
        log_fn("  (matplotlib no disponible — gráficas omitidas)")

    emit("done")

    log_fn(f"\n{'='*60}")
    log_fn("ARCHIVOS GENERADOS")
    log_fn(f"  {dir_data}/")
    log_fn(f"    {out_prefix}_progress.csv    — fitness y T1-T6 por generación")
    log_fn(f"    {out_prefix}_triangular.csv  — matriz triangular (datos)")
    log_fn(f"    {out_prefix}_triangular.txt  — matriz triangular (texto)")
    log_fn(f"    {out_prefix}_confusion.csv   — matriz de confusión")
    log_fn(f"  {dir_plots}/")
    log_fn(f"    {out_prefix}_evolution.png   — Figure 1 réplica Bullinaria")
    log_fn(f"    {out_prefix}_heatmap.png     — heatmap matriz triangular")
    log_fn(f"    {out_prefix}_confusion.png   — matriz de confusión visual")
    log_fn(f"\nTiempo total: {_elapsed_str(total_elapsed)}")
    log_fn(f"{'='*60}")

    return mean_acc, std_acc, tri_matrix, tri_s_accs, history
