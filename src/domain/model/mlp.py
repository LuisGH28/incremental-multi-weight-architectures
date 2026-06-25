"""
domain/model/mlp.py
====================
MLP con arquitectura de cuatro líneas de peso: w + fw¹ + fw² + fw³.

  - w    : pesos permanentes (slow weights)
  - fw¹  : fast-weight corto plazo — Bullinaria (2009), decay δ₁, scale σ₁
  - fw²  : fast-weight medio plazo — extensión previa,  decay δ₂, scale σ₂
  - fw³  : fast-weight semi-largo  — esta extensión,    decay δ₃, scale σ₃

El fenómeno δ₃→0 es la contribución central: la evolución descubre
que la tercera línea debe actuar como memoria permanente de sesión,
creando espontáneamente tres escalas temporales distintas.
"""
from __future__ import annotations

import numpy as np

from src.domain.model.genotype import Genotype
from src.shared.utils.math_utils import sigmoid, dsigmoid_from_output, make_one_hot


class MLP:
    N_IN  = 64
    N_OUT = 10

    def __init__(self, g: Genotype, use_dual: bool = True, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        self.g        = g
        self.use_dual = use_dual
        self.n_hid    = max(1, int(round(g.n_hid)))

        self.mask_ih = (rng.uniform(0, 1, (self.N_IN, self.n_hid)) < g.c_ih).astype(np.float64)
        self.mask_ho = (rng.uniform(0, 1, (self.n_hid, self.N_OUT)) < g.c_ho).astype(np.float64)

        self.w_ih = rng.uniform(-g.l_ih, g.u_ih, (self.N_IN + 1, self.n_hid))
        self.w_ho = rng.uniform(-g.l_ho, g.u_ho, (self.n_hid + 1, self.N_OUT))
        self.w_ih[0] = rng.uniform(-g.l_hb, g.u_hb, self.n_hid)
        self.w_ho[0] = rng.uniform(-g.l_ob, g.u_ob, self.N_OUT)
        self.w_ih[1:] *= self.mask_ih
        self.w_ho[1:] *= self.mask_ho

        if use_dual:
            self.fw_ih  = np.zeros_like(self.w_ih)   # fw¹
            self.fw_ho  = np.zeros_like(self.w_ho)
            self.fw2_ih = np.zeros_like(self.w_ih)   # fw²
            self.fw2_ho = np.zeros_like(self.w_ho)
            self.fw3_ih = np.zeros_like(self.w_ih)   # fw³
            self.fw3_ho = np.zeros_like(self.w_ho)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray):
        if self.use_dual:
            eff_ih = self.w_ih + self.fw_ih + self.fw2_ih + self.fw3_ih
            eff_ho = self.w_ho + self.fw_ho + self.fw2_ho + self.fw3_ho
        else:
            eff_ih = self.w_ih
            eff_ho = self.w_ho

        x_b    = np.concatenate(([1.0], x))
        hidden = sigmoid(x_b @ eff_ih)
        h_b    = np.concatenate(([1.0], hidden))
        out    = sigmoid(h_b @ eff_ho)
        return x_b, hidden, h_b, out

    # ── Entrenamiento de una sesión ───────────────────────────────────────────

    def train_session(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_epochs: int = 5000,
        emit_fn=None,
        individual_idx=None,
        session_idx=None,
        gen=None,
    ) -> int:
        g    = self.g
        T    = make_one_hot(y, self.N_OUT)
        N    = X.shape[0]
        stop = int(np.ceil((1.0 - g.tol_s) * N))

        if emit_fn and individual_idx is not None:
            emit_fn(
                "session_start",
                gen=gen, individual=individual_idx,
                session=session_idx, n_patterns=N,
                arch={"n_hid": self.n_hid, "use_dual": self.use_dual, "n_fw": 3},
            )

        correct  = 0
        total_ce = 0.0

        for epoch in range(max_epochs):
            # ── Decay de fast-weights ─────────────────────────────────────────
            if self.use_dual:
                self.fw_ih  *= (1.0 - g.fw_decay)
                self.fw_ho  *= (1.0 - g.fw_decay)
                self.fw2_ih *= (1.0 - g.fw2_decay)
                self.fw2_ho *= (1.0 - g.fw2_decay)
                self.fw3_ih *= (1.0 - g.fw3_decay)   # δ₃ → 0 = memoria permanente
                self.fw3_ho *= (1.0 - g.fw3_decay)

            if g.lam > 0:
                self.w_ih[1:] *= (1.0 - g.lam)
                self.w_ho[1:] *= (1.0 - g.lam)

            perm     = np.random.permutation(N)
            correct  = 0
            total_ce = 0.0

            for p in perm:
                x_b, hidden, h_b, out = self.forward(X[p])
                t   = T[p]
                eps = 1e-12
                total_ce += -float(
                    np.sum(t * np.log(out + eps) + (1 - t) * np.log(1 - out + eps))
                )

                if np.max(np.abs(out - t)) < g.tol_t:
                    correct += 1
                    continue

                # ── Backpropagation ───────────────────────────────────────────
                delta_o = out - t
                eff_ho  = (self.w_ho + self.fw_ho) if self.use_dual else self.w_ho
                delta_h = (eff_ho[1:] @ delta_o) * dsigmoid_from_output(hidden, g.ospo)

                grad_ho      = np.outer(h_b, delta_o)
                grad_ih      = np.outer(x_b, delta_h)
                grad_ih[1:] *= self.mask_ih
                grad_ho[1:] *= self.mask_ho

                # w — pesos lentos
                self.w_ho    -= g.eta_ho * grad_ho
                self.w_ho[0] -= g.eta_ob * delta_o
                self.w_ih    -= g.eta_ih * grad_ih
                self.w_ih[0] -= g.eta_hb * delta_h

                if self.use_dual:
                    # fw¹ — corto plazo (σ₁·η, decay δ₁)
                    self.fw_ho    -= (g.fw_scale * g.eta_ho) * grad_ho
                    self.fw_ho[0] -= (g.fw_scale * g.eta_ob) * delta_o
                    self.fw_ih    -= (g.fw_scale * g.eta_ih) * grad_ih
                    self.fw_ih[0] -= (g.fw_scale * g.eta_hb) * delta_h
                    self.fw_ih[1:]  *= self.mask_ih
                    self.fw_ho[1:]  *= self.mask_ho

                    # fw² — medio plazo (σ₂·η, decay δ₂)
                    self.fw2_ho    -= (g.fw2_scale * g.eta_ho) * grad_ho
                    self.fw2_ho[0] -= (g.fw2_scale * g.eta_ob) * delta_o
                    self.fw2_ih    -= (g.fw2_scale * g.eta_ih) * grad_ih
                    self.fw2_ih[0] -= (g.fw2_scale * g.eta_hb) * delta_h
                    self.fw2_ih[1:] *= self.mask_ih
                    self.fw2_ho[1:] *= self.mask_ho

                    # fw³ — semi-largo plazo (σ₃·η, decay δ₃→0)
                    self.fw3_ho    -= (g.fw3_scale * g.eta_ho) * grad_ho
                    self.fw3_ho[0] -= (g.fw3_scale * g.eta_ob) * delta_o
                    self.fw3_ih    -= (g.fw3_scale * g.eta_ih) * grad_ih
                    self.fw3_ih[0] -= (g.fw3_scale * g.eta_hb) * delta_h
                    self.fw3_ih[1:] *= self.mask_ih
                    self.fw3_ho[1:] *= self.mask_ho

            if emit_fn and individual_idx is not None and epoch % 50 == 0:
                emit_fn(
                    "epoch",
                    gen=gen, individual=individual_idx,
                    session=session_idx, epoch=epoch,
                    correct=correct, n=N,
                    ce=round(total_ce / N, 5),
                )

            if correct >= stop:
                if emit_fn and individual_idx is not None:
                    emit_fn(
                        "session_end",
                        gen=gen, individual=individual_idx,
                        session=session_idx, epochs_run=epoch + 1,
                    )
                return epoch + 1

        if emit_fn and individual_idx is not None:
            emit_fn(
                "session_end",
                gen=gen, individual=individual_idx,
                session=session_idx, epochs_run=max_epochs,
            )
        return max_epochs

    # ── Métricas ──────────────────────────────────────────────────────────────

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(
            1 for i in range(len(X))
            if np.argmax(self.forward(X[i])[3]) == y[i]
        )
        return correct / len(X)

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            np.argmax(self.forward(X[i])[3]) for i in range(len(X))
        ])
