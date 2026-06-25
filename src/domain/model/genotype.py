"""
domain/model/genotype.py
========================
Genotipo para la arquitectura fw³ (w + fw¹ + fw² + fw³).

Extiende fw² añadiendo un tercer par (fw3_decay δ₃, fw3_scale σ₃)
para la línea de fast-weight semi-lenta.

El fenómeno central de esta extensión es que δ₃ converge a 0.0,
creando una memoria de sesión permanente emergente no programada.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Genotype:
    # ── Topología ─────────────────────────────────────────────────────────────
    n_hid: float
    c_ih:  float;  c_ho:  float

    # ── Tasas de aprendizaje (4 componentes) ──────────────────────────────────
    eta_ih: float;  eta_hb: float
    eta_ho: float;  eta_ob: float

    # ── Distribuciones iniciales de pesos ─────────────────────────────────────
    l_ih: float;  u_ih: float
    l_hb: float;  u_hb: float
    l_ho: float;  u_ho: float
    l_ob: float;  u_ob: float

    # ── Regularización ────────────────────────────────────────────────────────
    ospo:  float;  lam:   float
    tol_t: float;  tol_s: float

    # ── fw¹ — Bullinaria (2009) corto plazo ───────────────────────────────────
    fw_decay:  float;  fw_scale:  float

    # ── fw² — extensión previa, medio plazo ───────────────────────────────────
    fw2_decay: float;  fw2_scale: float

    # ── fw³ — esta extensión, semi-largo plazo ────────────────────────────────
    fw3_decay: float;  fw3_scale: float

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def random(rng) -> "Genotype":
        r = rng.uniform
        return Genotype(
            n_hid=r(1, 100),
            c_ih=r(0, 1),      c_ho=r(0, 1),
            eta_ih=r(0, 1),    eta_hb=r(0, 1),
            eta_ho=r(0, 1),    eta_ob=r(0, 1),
            l_ih=r(0, 1),      u_ih=r(0, 1),
            l_hb=r(0, 1),      u_hb=r(0, 1),
            l_ho=r(0, 1),      u_ho=r(0, 1),
            l_ob=r(0, 1),      u_ob=r(0, 1),
            ospo=r(0, 0.2),    lam=r(0, 0.001),
            tol_t=r(0, 0.5),   tol_s=r(0, 1.0),
            fw_decay=r(0, 0.2),    fw_scale=r(2, 20),    # fw¹ rápido
            fw2_decay=r(0, 0.05),  fw2_scale=r(1, 10),   # fw² medio
            fw3_decay=r(0, 0.01),  fw3_scale=r(0.5, 5),  # fw³ lento
        )

    def as_array(self) -> np.ndarray:
        return np.array([
            self.n_hid,  self.c_ih,    self.c_ho,
            self.eta_ih, self.eta_hb,  self.eta_ho,  self.eta_ob,
            self.l_ih,   self.u_ih,    self.l_hb,    self.u_hb,
            self.l_ho,   self.u_ho,    self.l_ob,    self.u_ob,
            self.ospo,   self.lam,     self.tol_t,   self.tol_s,
            self.fw_decay,   self.fw_scale,
            self.fw2_decay,  self.fw2_scale,
            self.fw3_decay,  self.fw3_scale,
        ], dtype=np.float64)

    @staticmethod
    def from_array(a: np.ndarray) -> "Genotype":
        return Genotype(
            n_hid=a[0],   c_ih=a[1],   c_ho=a[2],
            eta_ih=a[3],  eta_hb=a[4], eta_ho=a[5],  eta_ob=a[6],
            l_ih=a[7],    u_ih=a[8],   l_hb=a[9],    u_hb=a[10],
            l_ho=a[11],   u_ho=a[12],  l_ob=a[13],   u_ob=a[14],
            ospo=a[15],   lam=a[16],   tol_t=a[17],  tol_s=a[18],
            fw_decay=a[19],   fw_scale=a[20],
            fw2_decay=a[21],  fw2_scale=a[22],
            fw3_decay=a[23],  fw3_scale=a[24],
        )

    def crossover_mutate(self, other: "Genotype", std: float, rng) -> "Genotype":
        a, b = self.as_array(), other.as_array()
        child = (
            rng.uniform(np.minimum(a, b), np.maximum(a, b))
            + rng.normal(0, std, len(a))
        )
        child = np.clip(child, 0, None)
        child[20] = max(child[20], 2.0)    # fw_scale  ≥ 2
        child[22] = max(child[22], 1.0)    # fw2_scale ≥ 1
        child[24] = max(child[24], 0.5)    # fw3_scale ≥ 0.5
        return Genotype.from_array(child)

    def to_dict(self) -> dict:
        return {
            "n_hid":      int(round(self.n_hid)),
            "c_ih":       round(self.c_ih,  3),
            "c_ho":       round(self.c_ho,  3),
            "eta_ih":     round(self.eta_ih, 6),
            "eta_hb":     round(self.eta_hb, 6),
            "eta_ho":     round(self.eta_ho, 6),
            "eta_ob":     round(self.eta_ob, 6),
            "ospo":       round(self.ospo,   4),
            "lam":        round(self.lam,    6),
            "tol_t":      round(self.tol_t,  4),
            "tol_s":      round(self.tol_s,  4),
            "fw_decay":   round(self.fw_decay,  4),
            "fw_scale":   round(self.fw_scale,  2),
            "fw2_decay":  round(self.fw2_decay, 4),
            "fw2_scale":  round(self.fw2_scale, 2),
            "fw3_decay":  round(self.fw3_decay, 4),
            "fw3_scale":  round(self.fw3_scale, 2),
        }

    def summary(self) -> str:
        d = self.to_dict()
        return (
            f"n_hid={d['n_hid']}  cIH={d['c_ih']:.2f}  cHO={d['c_ho']:.2f} | "
            f"ηIH={d['eta_ih']:.2e}  ηHB={d['eta_hb']:.2e}  "
            f"ηHO={d['eta_ho']:.2e}  ηOB={d['eta_ob']:.2e} | "
            f"λ={d['lam']:.2e}  oSPO={d['ospo']:.4f} | "
            f"tol_t={d['tol_t']:.3f}  tol_s={d['tol_s']:.3f} | "
            f"δ₁={d['fw_decay']:.4f}  σ₁={d['fw_scale']:.1f} | "
            f"δ₂={d['fw2_decay']:.4f}  σ₂={d['fw2_scale']:.1f} | "
            f"δ₃={d['fw3_decay']:.4f}  σ₃={d['fw3_scale']:.1f}"
        )
