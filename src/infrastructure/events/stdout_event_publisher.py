"""
infrastructure/events/stdout_event_publisher.py
================================================
Publica eventos JSON en stdout para que el servidor SSE los reenvíe al
dashboard, y opcionalmente escribe un archivo .log con el historial completo.

El formato es idéntico al del proyecto base para que dashboard.html
no necesite cambios en su listener SSE.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Publisher de eventos (stdout → SSE → dashboard)
# ─────────────────────────────────────────────────────────────────────────────

class EventPublisher:
    """
    Emite eventos JSON de una sola línea en stdout.

    Cuando dashboard_mode=True cada llamada a emit() imprime:
        {"type": "...", "ts": 1234567890.123, ...}

    El servidor SSE (interfaces/http/server.py) lee stdout línea a línea
    y las reenvía a los clientes conectados.
    """

    def __init__(self, dashboard_mode: bool = False):
        self.dashboard_mode = dashboard_mode

    def emit(self, event_type: str, **kwargs) -> None:
        if not self.dashboard_mode:
            return
        payload = {"type": event_type, "ts": time.time(), **kwargs}
        print(json.dumps(payload), flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logger centralizado (terminal + archivo)
# ─────────────────────────────────────────────────────────────────────────────

class NeuroLogger:
    """
    Wrapper ligero alrededor de logging.Logger.

    - Mensajes INFO y superiores → terminal (stderr)
    - Todos los mensajes (DEBUG incluido) → archivo .log

    También expone handle(line) para que server.py pueda pasarle
    las líneas de stdout del subproceso de neuroevolución.
    """

    def __init__(self, log_path: str):
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._logger = logging.getLogger(f"neuroevo.{log_path}")
        self._logger.setLevel(logging.DEBUG)

        # Handler archivo
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        # Handler terminal
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

    # ── Métodos de log ────────────────────────────────────────────────────────

    def info(self, msg: str)  -> None: self._logger.info(msg)
    def debug(self, msg: str) -> None: self._logger.debug(msg)
    def warn(self, msg: str)  -> None: self._logger.warning(msg)
    def error(self, msg: str) -> None: self._logger.error(msg)

    def handle(self, line: str) -> None:
        """
        Recibe una línea de stdout del subproceso de neuroevolución.
        Si es JSON válido, la registra como DEBUG; si no, como INFO.
        """
        try:
            json.loads(line)          # valida que sea JSON
            self._logger.debug(line)
        except (json.JSONDecodeError, ValueError):
            self._logger.info(line)

    def close(self) -> None:
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de conveniencia para uso en CLI
# ─────────────────────────────────────────────────────────────────────────────

def make_log_fns(logger: NeuroLogger):
    """
    Devuelve (log_fn, log_detail_fn) listas para pasar a run_evolution().
    """
    return logger.info, logger.debug
