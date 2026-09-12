#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP/SSE server for the fw2 OptDigits dashboard.

The server launches src/interfaces/cli/neuro.py as a subprocess and forwards
structured JSON telemetry from stdout to the browser through Server-Sent Events.
Human-readable stdout/stderr remains terminal/log output.

Usage:
    python run.py server
    python run.py server --generations 50 --pop-size 100 --max-epochs 500

Then open: http://localhost:8765/dashboard.html
"""

import argparse
import http.server
import json
import mimetypes
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.infrastructure.events.stdout_event_publisher import NeuroLogger

DASHBOARD_DIR = Path(__file__).resolve().parents[3] / "dashboard"

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


def _is_json_event_line(line: str) -> bool:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("type"), str)


def _drain_stderr(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        line = line.rstrip()
        if line:
            print(line, file=sys.stderr, flush=True)


def _run_script(cmd: list, logger: NeuroLogger) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_stderr,
        args=(proc,),
        daemon=True,
    )
    stderr_thread.start()
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            if _is_json_event_line(line):
                _broadcast(line)
            logger.handle(line)
    proc.wait()
    _done.set()
    done_line = json.dumps({"type": "server_done"})
    _broadcast(done_line)
    logger.handle(done_line)
    logger.close()


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/dashboard.html", "/index.html"):
            return self._serve_static(DASHBOARD_DIR / "index.html", "text/html; charset=utf-8")

        if path.startswith("/assets/"):
            requested = (DASHBOARD_DIR / path.lstrip("/")).resolve()
            if DASHBOARD_DIR.resolve() not in requested.parents:
                self.send_error(403)
                return
            return self._serve_static(requested)

        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type",  "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q: list = []
            with _subs_lock:
                _subscribers.append(q)

            # Send a comment heartbeat before replay so the browser sees the stream as open.
            try:
                self.wfile.write(": heartbeat\n\n".encode())
                self.wfile.flush()
            except Exception:
                with _subs_lock:
                    try: _subscribers.remove(q)
                    except ValueError: pass
                return

            # Late dashboard clients need the already-emitted experiment state.
            with _events_lock:
                buffered = list(_events)
            for line in buffered:
                try:
                    self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    break

            last_heartbeat = time.time()
            try:
                while True:
                    if q:
                        line = q.pop(0)
                        self.wfile.write(f"data: {line}\n\n".encode())
                        self.wfile.flush()
                        last_heartbeat = time.time()
                    elif _done.is_set() and not q:
                        # Give the browser a short grace period to process server_done.
                        time.sleep(3.0)
                        break
                    else:
                        time.sleep(0.05)
                        # Comment heartbeats keep long-running experiments from timing out.
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

    def _serve_static(self, file_path: Path, content_type=None):
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, f"{file_path.name} not found")
            return
        data = file_path.read_bytes()
        guessed_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type or guessed_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(
        description="SSE dashboard for fw2 OptDigits neuroevolution"
    )
    ap.add_argument("--dataset",    default="optdigits",
                    choices=["optdigits"])
    ap.add_argument("--data-dir",   default="./data")
    ap.add_argument("--train",      default=None)
    ap.add_argument("--test",       default=None)
    ap.add_argument("--generations",   type=int, default=50)
    ap.add_argument("--pop-size",      type=int, default=100)
    ap.add_argument("--max-epochs",    type=int, default=500)
    ap.add_argument("--mutation-std",  type=float, default=0.05)
    ap.add_argument("--top-frac",      type=float, default=0.10)
    ap.add_argument("--no-dual",       action="store_true")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--verbose-indiv", type=int, default=3)
    ap.add_argument("--out-prefix", default="fw2")
    ap.add_argument("--tri-runs",   type=int, default=5)
    ap.add_argument("--dir-data",   default="result")
    ap.add_argument("--dir-plots",  default="plots")
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
        "--mutation-std",  str(args.mutation_std),
        "--top-frac",      str(args.top_frac),
        "--seed",          str(args.seed),
        "--verbose-indiv", str(args.verbose_indiv),
        "--out-prefix",    args.out_prefix,
        "--tri-runs",      str(args.tri_runs),
        "--dir-data",      args.dir_data,
        "--dir-plots",     args.dir_plots,
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
    print("  Open the link in your browser.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
