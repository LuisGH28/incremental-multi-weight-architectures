"""
Plotting utilities for the fw3 experiment.

The evolution plot extends Bullinaria (2009), Figure 1 with fw1, fw2, and fw3
parameter panels.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_evolution(history, label: str, out_path: str, log_fn=None) -> None:
    G = history.gens
    C = dict(
        cih="#2563EB", cho="#16A34A",
        eta_ih="#DC2626", eta_hb="#D97706", eta_ho="#7C3AED", eta_ob="#0891B2",
        lam="#64748B", ospo="#EC4899",
        fw1d="#F97316", fw1s="#EA580C",
        fw2d="#8B5CF6", fw2s="#6D28D9",
        fw3d="#059669", fw3s="#047857",
        tol_t="#0D9488", tol_s="#7C3AED",
        val="#2563EB", test="#DC2626",
    )

    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def _ax(row, col, title, ylabel):
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("Generation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        return ax

    ax = _ax(0, 0, "Connectivity", "Con.")
    ax.plot(G, history.con_ih, color=C["cih"], lw=1.4, label="conIH")
    ax.plot(G, history.con_ho, color=C["cho"], lw=1.4, label="conHO")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, framealpha=0.7)

    ax = _ax(0, 1, "Learning rates", "log η")
    ax.plot(G, history.log_eta_ho, color=C["eta_ho"], lw=1.4, label="ηHO")
    ax.plot(G, history.log_eta_ih, color=C["eta_ih"], lw=1.4, label="ηIH")
    ax.plot(G, history.log_eta_hb, color=C["eta_hb"], lw=1.4, label="ηHB")
    ax.plot(G, history.log_eta_ob, color=C["eta_ob"], lw=1.4, label="ηOB")
    ax.legend(fontsize=7, framealpha=0.7)

    ax = _ax(1, 0, "Regularization params", "log-param")
    ax.plot(G, history.log_lam,  color=C["lam"],  lw=1.4, label="λ (weight decay)")
    ax.plot(G, history.log_ospo, color=C["ospo"], lw=1.4, label="oSPO")
    ax.legend(fontsize=7, framealpha=0.7)

    ax = _ax(1, 1, "Fast-weight params (fw¹ + fw² + fw³)", "log-fwt")
    ax.plot(G, history.log_fw1_decay, color=C["fw1d"], lw=1.4, label="δ₁ (fw¹ decay)")
    ax.plot(G, history.log_fw1_scale, color=C["fw1s"], lw=1.4, ls="--", label="σ₁ (fw¹ scale)")
    ax.plot(G, history.log_fw2_decay, color=C["fw2d"], lw=1.4, label="δ₂ (fw² decay)")
    ax.plot(G, history.log_fw2_scale, color=C["fw2s"], lw=1.4, ls="--", label="σ₂ (fw² scale)")
    ax.plot(G, history.log_fw3_decay, color=C["fw3d"], lw=1.4, label="δ₃ (fw³ decay)")
    ax.plot(G, history.log_fw3_scale, color=C["fw3s"], lw=1.4, ls="--", label="σ₃ (fw³ scale)")
    ax.legend(fontsize=6, framealpha=0.7)

    ax = _ax(2, 0, "Tolerances", "Tol.")
    ax.plot(G, history.tol_s, color=C["tol_s"], lw=1.4, label="s (stopping)")
    ax.plot(G, history.tol_t, color=C["tol_t"], lw=1.4, label="t (output tol.)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, framealpha=0.7)

    ax = _ax(2, 1, "Generalization performance", "% Correct")
    ax.plot(G, history.val_acc, color=C["val"], lw=1.6, label="Validation set")
    test_clean = [(g, v) for g, v in zip(G, history.test_acc) if not np.isnan(v)]
    if test_clean:
        gx, vx = zip(*test_clean)
        ax.plot(gx, vx, color=C["test"], lw=1.6, ls="--", label="Test set")
    ax.axhline(95.07, color="#6B7280", lw=0.9, ls=":", label="Bullinaria fw¹ 95.07%")
    ax.set_ylim(60, 100); ax.legend(fontsize=7, framealpha=0.7)

    fig.suptitle(
        f"Evolution of parameters and performance — {label}\n"
        f"(replica of Bullinaria 2009, Figure 1)",
        fontsize=10, fontweight="bold", y=0.995,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if log_fn:
        log_fn(f"  Gráfica evolución: {out_path}")


def plot_triangular_heatmap(
    matrix, session_accs, mean_acc, std_acc, label, out_path, log_fn=None,
):
    n    = 6
    data = np.full((n + 1, n), np.nan)
    for s in range(n):
        for b in range(n):
            if matrix[s][b] is not None:
                data[s, b] = matrix[s][b]
    data[n, :] = session_accs

    fig, ax = plt.subplots(figsize=(9, 5))
    im      = ax.imshow(np.ma.masked_invalid(data), aspect="auto",
                        cmap="RdYlGn", vmin=80, vmax=100)
    plt.colorbar(im, ax=ax, label="% Correct", fraction=0.03, pad=0.02)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"B{i+1}" for i in range(n)], fontsize=9)
    ax.set_yticks(range(n + 1))
    ax.set_yticklabels([f"T{i+1}" for i in range(n)] + ["Test"], fontsize=9)
    for s in range(n + 1):
        for b in range(n):
            v = data[s, b]
            if not np.isnan(v):
                ax.text(b, s, f"{v:.1f}", ha="center", va="center",
                        fontsize=7.5, color="white" if v < 88 else "black",
                        fontweight="bold")
    ax.axhline(n - 0.5, color="white", lw=2.5)
    ax.set_title(
        f"Triangular Evaluation Matrix — {label}\n"
        f"Mean test acc: {mean_acc:.2f}% ± {std_acc:.2f}%   "
        f"(Bullinaria fw¹: 95.07% ± 0.04%)",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Batch (Bⱼ)", fontsize=9)
    ax.set_ylabel("Session (Tᵢ)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if log_fn:
        log_fn(f"  Heatmap triangular: {out_path}")


def plot_confusion(cm, mean_acc, label, out_path, log_fn=None):
    fig, ax   = plt.subplots(figsize=(8, 7))
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im        = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Proporción", fraction=0.03, pad=0.02)
    n = cm.shape[0]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)])
    ax.set_yticklabels([str(i) for i in range(n)])
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted digit", fontsize=10)
    ax.set_ylabel("True digit", fontsize=10)
    ax.set_title(
        f"Confusion Matrix — {label}\nTest accuracy: {mean_acc:.2f}%",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if log_fn:
        log_fn(f"  Matriz de confusión: {out_path}")
