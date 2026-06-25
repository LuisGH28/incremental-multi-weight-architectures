#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interfaces/http/server.py
==========================
Servidor HTTP/SSE para el dashboard fw².

Lanza src/interfaces/cli/neuro.py como subproceso y reenvía
su stdout (eventos JSON) al navegador vía Server-Sent Events.

Uso:
    python run.py server
    python run.py server --generations 50 --pop-size 100 --max-epochs 500

Luego abrir:  http://localhost:8765/dashboard.html
"""

import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.infrastructure.events.stdout_event_publisher import NeuroLogger

# ── Estado global compartido ──────────────────────────────────────────────────
_events: list      = []
_events_lock       = threading.Lock()
_subscribers: list = []
_subs_lock         = threading.Lock()
_done              = threading.Event()


def _broadcast(line: str) -> None:
    with _events_lock:
        _events.append(line)
    with _subs_lock:
        for q in list(_subscribers):
            try:
                q.append(line)
            except Exception:
                pass


def _run_script(cmd: list, logger: NeuroLogger) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _broadcast(line)
            logger.handle(line)
    proc.wait()
    _done.set()
    done_line = json.dumps({"type": "server_done"})
    _broadcast(done_line)
    logger.handle(done_line)
    logger.close()


# ── Handler HTTP ──────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silencia el log de acceso

    def do_GET(self):
        path = self.path.split("?")[0]

        # ── Sirve dashboard.html ──────────────────────────────────────────────
        if path in ("/", "/dashboard.html"):
            html_path = Path(__file__).parent / "dashboard.html"
            if not html_path.exists():
                self.send_error(404, "dashboard.html not found")
                return
            data = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # ── SSE /events ───────────────────────────────────────────────────────
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type",  "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q: list = []
            with _subs_lock:
                _subscribers.append(q)

            # Heartbeat inmediato — confirma conexión al browser antes del replay
            try:
                self.wfile.write(": heartbeat\n\n".encode())
                self.wfile.flush()
            except Exception:
                with _subs_lock:
                    try: _subscribers.remove(q)
                    except ValueError: pass
                return

            # Replay de eventos anteriores (clientes que se conectan tarde)
            with _events_lock:
                buffered = list(_events)
            for line in buffered:
                try:
                    self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    break

            # Streaming de nuevos eventos
            last_heartbeat = time.time()
            try:
                while True:
                    if q:
                        line = q.pop(0)
                        self.wfile.write(f"data: {line}\n\n".encode())
                        self.wfile.flush()
                        last_heartbeat = time.time()
                    elif _done.is_set() and not q:
                        # Esperar 3 s extra para que el browser procese server_done
                        time.sleep(3.0)
                        break
                    else:
                        time.sleep(0.05)
                        # Heartbeat cada 15 s para mantener la conexión viva
                        if time.time() - last_heartbeat > 15:
                            self.wfile.write(": heartbeat\n\n".encode())
                            self.wfile.flush()
                            last_heartbeat = time.time()
            except Exception:
                pass
            finally:
                with _subs_lock:
                    try:
                        _subscribers.remove(q)
                    except ValueError:
                        pass
            return

        self.send_error(404)


# ── Punto de entrada ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Dashboard SSE para neuroevolución fw²"
    )
    # ── Dataset ───────────────────────────────────────────────────────────────
    ap.add_argument("--dataset",    default="optdigits",
                    choices=["optdigits"])
    ap.add_argument("--data-dir",   default="./data")
    ap.add_argument("--train",      default=None)
    ap.add_argument("--test",       default=None)
    # ── Evolución ─────────────────────────────────────────────────────────────
    ap.add_argument("--generations",   type=int, default=50)
    ap.add_argument("--pop-size",      type=int, default=100)
    ap.add_argument("--max-epochs",    type=int, default=500)
    ap.add_argument("--no-dual",       action="store_true")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--verbose-indiv", type=int, default=3)
    # ── Servidor ──────────────────────────────────────────────────────────────
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log",  type=str, default="neuroevo.log",
                    help="Ruta del archivo .log")
    args = ap.parse_args()

    script = Path(__file__).parent.parent / "cli" / "neuro.py"

    cmd = [
        sys.executable, str(script),
        "--dataset",       args.dataset,
        "--data-dir",      args.data_dir,
        "--generations",   str(args.generations),
        "--pop-size",      str(args.pop_size),
        "--max-epochs",    str(args.max_epochs),
        "--seed",          str(args.seed),
        "--verbose-indiv", str(args.verbose_indiv),
        "--dashboard",
    ]
    if args.train:
        cmd += ["--train", args.train]
    if args.test:
        cmd += ["--test", args.test]
    if args.no_dual:
        cmd.append("--no-dual")

    logger = NeuroLogger(args.log)
    t = threading.Thread(target=_run_script, args=(cmd, logger), daemon=True)
    t.start()

    server = http.server.HTTPServer(("localhost", args.port), Handler)
    url    = f"http://localhost:{args.port}/dashboard.html"
    print(f"\n  Dashboard: {url}")
    print(f"  Gen: {args.generations} | Pop: {args.pop_size} | MaxEpochs: {args.max_epochs}")
    print(f"  Abre el link en tu navegador.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()
