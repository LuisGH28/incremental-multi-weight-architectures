"""
Main neuroevolution loop for fw3: w + fw1 + fw2 + fw3.

Multiprocessing is available for CLI runs. Dashboard mode stays single-process
so training progress can be emitted in-order through the SSE telemetry channel.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
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
from src.infrastructure.data.incremental_splitter import DEFAULT_N_SESSIONS, split_incremental
from src.infrastructure.events.stdout_event_publisher import EventPublisher


WEIGHT_LABELS = ["w", "fw1", "fw2", "fw3"]
N_FAST_WEIGHT_LINES = 3

BULLINARIA_OPTDIGITS_REFERENCE = {
    "name": "Bullinaria fw1 OptDigits reference",
    "dataset": "optdigits",
    "target_acc": 95.07,
    "session_accs": [91.62, 93.47, 94.21, 94.60, 94.87, 95.07],
}


def _eval_worker(args):
    idx, g_arr, sessions_xy, val_xy, use_dual, max_epochs, seed = args
    rng      = np.random.default_rng(seed)
    g        = Genotype.from_array(np.array(g_arr))
    net      = MLP(g, use_dual=use_dual, rng=rng)
    sessions = [(np.array(X), np.array(y)) for X, y in sessions_xy]
    val      = (np.array(val_xy[0]), np.array(val_xy[1]))
    for Xs, ys in sessions:
        net.train_session(Xs, ys, max_epochs=max_epochs)
    return idx, net.accuracy(val[0], val[1])


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
        self.log_fw3_decay: List[float] = []
        self.log_fw3_scale: List[float] = []
        self.tol_t:         List[float] = []
        self.tol_s:         List[float] = []
        self.val_acc:       List[float] = []
        self.test_acc:      List[float] = []

    def record(self, gen: int, survivors: List[Genotype],
               top_fitness: float, test_acc_val: Optional[float] = None):
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
        self.log_fw3_decay.append(lmean(lambda g: g.fw3_decay))
        self.log_fw3_scale.append(lmean(lambda g: g.fw3_scale))
        self.tol_t.append(mean(lambda g: g.tol_t))
        self.tol_s.append(mean(lambda g: g.tol_s))
        self.val_acc.append(top_fitness * 100.0)
        self.test_acc.append(
            test_acc_val * 100.0 if test_acc_val is not None else float("nan")
        )


def run_evolution(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
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
    out_prefix:  str = "fw3",
    tri_runs:    int = 5,
    dir_data:    str = "result",
    dir_plots:   str = "plots",
    n_workers:   int = 0,
    publisher: Optional[EventPublisher] = None,
    log_fn=None,
    log_detail_fn=None,
):
    run_start = time.time()

    if log_fn        is None: log_fn        = print
    if log_detail_fn is None: log_detail_fn = lambda _: None

    def emit(event_type: str, **kwargs):
        if publisher:
            publisher.emit(event_type, **kwargs)

    label = (
        "fw³ — w + fw¹ + fw² + fw³ (extensión propuesta)"
        if use_dual else "MLP sin fast-weights"
    )

    n_cpu     = mp.cpu_count()
    n_workers = n_workers if n_workers > 0 else n_cpu
    n_workers = min(n_workers, pop_size)

    master_rng = np.random.default_rng(seed)
    history    = EvoHistory()

    os.makedirs(dir_data,  exist_ok=True)
    os.makedirs(dir_plots, exist_ok=True)

    csv_file = open(os.path.join(dir_data, f"{out_prefix}_progress.csv"), "w", newline="")
    cw = csv.writer(csv_file)
    session_headers = [f"T{i + 1}_test" for i in range(DEFAULT_N_SESSIONS)]
    cw.writerow([
        "gen", "best_fitness_val",
        *session_headers,
        "elapsed_s",
    ])
    csv_file.flush()

    emit("config",
         pop_size=pop_size, n_generations=n_generations,
         use_dual=use_dual, max_epochs=max_epochs,
         n_train=len(X_train), n_test=len(X_test),
         dataset="optdigits", n_inputs=n_in, n_outputs=n_out,
         n_sessions=DEFAULT_N_SESSIONS,
         n_fast_weight_lines=N_FAST_WEIGHT_LINES,
         weight_labels=WEIGHT_LABELS,
         reference=BULLINARIA_OPTDIGITS_REFERENCE)

    population = [Genotype.random(master_rng) for _ in range(pop_size)]
    fitness    = np.zeros(pop_size)

    log_fn("=" * 60)
    log_fn(f"INICIO EVOLUCIÓN — {label}")
    log_fn(f"Budget: {n_generations} gen | pop={pop_size} | max_epochs={max_epochs}")
    log_fn(f"Paralelismo: {n_workers} workers / {n_cpu} CPUs disponibles")
    log_fn("=" * 60)

    for gen in range(n_generations):
        t0      = time.time()
        gen_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        sessions, val = split_incremental(X_train, y_train, rng=gen_rng)

        emit("gen_start", gen=gen, n_gen=n_generations, pop_size=pop_size)

        use_mp = (n_workers > 1 and publisher is None)

        if use_mp:
            # Worker processes cannot emit ordered dashboard telemetry safely.
            sessions_xy = [(Xs.tolist(), ys.tolist()) for Xs, ys in sessions]
            val_xy      = (val[0].tolist(), val[1].tolist())
            worker_args = [
                (i, population[i].as_array().tolist(), sessions_xy, val_xy,
                 use_dual, max_epochs, int(master_rng.integers(0, 2**31)))
                for i in range(len(population))
            ]
            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(_eval_worker, worker_args)
            for idx, fit in results:
                fitness[idx] = fit
        else:
            for i, g in enumerate(population):
                eval_rng  = np.random.default_rng(master_rng.integers(0, 2**31))
                use_emit  = (i < verbose_individuals)
                fitness[i] = evaluate_individual(
                    g, sessions, val, use_dual, max_epochs, eval_rng,
                    emit_fn=emit if use_emit else None,
                    individual_idx=i if use_emit else None,
                    gen=gen if use_emit else None,
                )
                emit("individual_done", gen=gen, individual=i,
                     fitness=round(float(fitness[i]), 4))

        cur_size = len(population)
        ranked   = np.argsort(fitness[:cur_size])[::-1]
        best_idx = ranked[0]
        best_g   = population[best_idx]
        top_n    = max(1, cur_size // 10)
        top_fit  = float(fitness[ranked[:top_n]].mean())

        # Track a quick test-set estimate for plots without changing selection.
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

        n_survivors = max(1, cur_size // 2)
        survivors   = [population[i] for i in ranked[:n_survivors]]
        history.record(gen, survivors, top_fit, quick_test)

        elapsed_gen = time.time() - t0
        elapsed_tot = time.time() - run_start

        while len(t_accs) < DEFAULT_N_SESSIONS: t_accs.append("")
        cw.writerow([gen, round(float(fitness[best_idx]) * 100, 2),
                     *t_accs[:DEFAULT_N_SESSIONS], round(elapsed_gen, 1)])
        csv_file.flush()

        log_fn(
            f"[GEN {gen:3d}] best={fitness[best_idx]*100:.2f}%  "
            f"top10={top_fit*100:.2f}%  T{DEFAULT_N_SESSIONS}={t_accs[DEFAULT_N_SESSIONS - 1]}%  "
            f"gen={_elapsed_str(elapsed_gen)}  total={_elapsed_str(elapsed_tot)}"
        )
        log_detail_fn(f"  Mejor genotipo gen {gen}: {best_g.summary()}")

        emit("gen_end",
             gen=gen,
             best_fitness=round(float(fitness[best_idx]), 4),
             top10_mean=round(top_fit, 4),
             pop_mean=round(float(fitness[:cur_size].mean()), 4),
             elapsed=round(elapsed_gen, 2),
             best_genotype=best_g.to_dict(),
             fitness_all=[round(float(f), 4) for f in fitness[ranked]])

        parent_rng = np.random.default_rng(master_rng.integers(0, 2**31))
        n_children = pop_size - len(survivors)
        children = [
            survivors[i % len(survivors)].crossover_mutate(
                survivors[parent_rng.integers(0, len(survivors))],
                mutation_std, parent_rng,
            )
            for i in range(n_children)
        ]
        population = survivors + children
        fitness    = np.zeros(pop_size)

    log_fn("\n[Evaluación final sobre test set...]")
    emit("final_start")

    final_rng = np.random.default_rng(master_rng.integers(0, 2**31))
    final_sessions, final_val = split_incremental(X_train, y_train, rng=final_rng)
    fitness = np.zeros(len(population))
    for i, g in enumerate(population):
        er  = np.random.default_rng(master_rng.integers(0, 2**31))
        net = MLP(g, use_dual=use_dual, rng=er)
        for Xs, ys in final_sessions:
            net.train_session(Xs, ys, max_epochs=max_epochs)
        fitness[i] = net.accuracy(final_val[0], final_val[1])

    ranked     = np.argsort(fitness)[::-1]
    top_n      = max(1, int(top_frac * pop_size))
    top_indivs = [population[i] for i in ranked[:top_n]]

    test_accs, session_accs = [], []
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

    cm_net = MLP(
        top_indivs[0], use_dual=use_dual,
        rng=np.random.default_rng(master_rng.integers(0, 2**31)),
    )
    cm_ses, _ = split_incremental(
        X_train, y_train,
        rng=np.random.default_rng(master_rng.integers(0, 2**31)),
    )
    for Xs, ys in cm_ses:
        cm_net.train_session(Xs, ys, max_epochs=max_epochs)
    cm = compute_confusion(cm_net, X_test, y_test)

    total_elapsed = time.time() - run_start

    log_fn(f"\n{'='*60}")
    log_fn("RESULTADO FINAL")
    log_fn(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    log_fn(f"  Referencia Bullinaria fw¹: 95.07% ± 0.04%")
    log_fn(f"  Tiempo total: {_elapsed_str(total_elapsed)}")
    log_fn(f"{'='*60}")

    ft = session_accs[0] if session_accs else [""] * DEFAULT_N_SESSIONS
    while len(ft) < DEFAULT_N_SESSIONS: ft.append("")
    cw.writerow(["FINAL", round(mean_acc, 2), *ft[:DEFAULT_N_SESSIONS], round(total_elapsed, 1)])
    csv_file.flush()
    csv_file.close()

    emit("final_result",
         mean_acc=round(mean_acc, 2), std_acc=round(std_acc, 2),
         session_accs=session_accs[0] if session_accs else [],
         best_genotype=top_indivs[0].to_dict(),
         target_acc=95.07,
         dataset="optdigits", n_inputs=n_in, n_outputs=n_out,
         n_sessions=DEFAULT_N_SESSIONS,
         n_fast_weight_lines=N_FAST_WEIGHT_LINES,
         weight_labels=WEIGHT_LABELS)

    log_fn(f"\n[Generando matriz triangular ({tri_runs} runs)...]")
    tri_matrix, tri_s_accs, tri_mean, tri_std = evaluate_triangular(
        top_indivs, X_train, y_train, X_test, y_test,
        use_dual=use_dual, max_epochs=max_epochs,
        master_rng=master_rng, n_runs=tri_runs,
        log_fn=log_fn,
    )

    print_triangular_matrix(tri_matrix, tri_s_accs, tri_mean, tri_std, label=label)

    save_triangular_csv(tri_matrix, tri_s_accs, tri_mean, tri_std,
                        path=os.path.join(dir_data, f"{out_prefix}_triangular.csv"),
                        label=label)

    txt_lines = [
        f"MATRIZ TRIANGULAR — {label}\n",
        f"{'Sesion':<8s}" + "".join(f"{'B'+str(i+1):>9s}" for i in range(DEFAULT_N_SESSIONS)) + "\n",
    ]
    for s in range(DEFAULT_N_SESSIONS):
        row = f"T{str(s+1):<7s}"
        for b in range(DEFAULT_N_SESSIONS):
            v    = tri_matrix[s][b]
            row += f"{v:>9.2f}" if v is not None else f"{'--':>9s}"
        txt_lines.append(row + "\n")
    txt_lines += [
        f"{'Test':<8s}" + "".join(f"{a:>9.2f}" for a in tri_s_accs) + "\n",
        f"\nMedia: {tri_mean:.2f}% ± {tri_std:.2f}%\n",
        "Referencia Bullinaria fw¹: 95.07% ± 0.04%\n",
        f"Tiempo total: {_elapsed_str(total_elapsed)}\n",
    ]
    with open(os.path.join(dir_data, f"{out_prefix}_triangular.txt"), "w") as f:
        f.writelines(txt_lines)

    save_confusion_csv(cm, os.path.join(dir_data, f"{out_prefix}_confusion.csv"))

    log_fn("\n[Generando gráficas...]")
    try:
        from src.shared.utils.plots import (
            plot_evolution, plot_triangular_heatmap, plot_confusion,
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
    log_fn(f"  {dir_data}/  {out_prefix}_progress.csv | _triangular.csv/txt | _confusion.csv")
    log_fn(f"  {dir_plots}/ {out_prefix}_evolution.png | _heatmap.png | _confusion.png")
    log_fn(f"\nTiempo total: {_elapsed_str(total_elapsed)}")
    log_fn(f"{'='*60}")

    return mean_acc, std_acc, tri_matrix, tri_s_accs, history
