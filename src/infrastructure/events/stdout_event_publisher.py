"""
Publish experiment telemetry and logs for CLI/server modes.

Structured JSON events are written to stdout for the SSE server. Human-readable
logs are written to stderr and to the run log file.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Optional


class EventPublisher:
    """
    Emit one JSON event per stdout line when dashboard mode is enabled.

    When dashboard_mode=True, each emit() call prints:
        {"type": "...", "ts": 1234567890.123, ...}
    """

    def __init__(self, dashboard_mode: bool = False):
        self.dashboard_mode = dashboard_mode

    def emit(self, event_type: str, **kwargs) -> None:
        if not self.dashboard_mode:
            return
        payload = {"type": event_type, "ts": time.time(), **kwargs}
        print(json.dumps(payload), flush=True)


class NeuroLogger:
    """
    Lightweight wrapper around logging.Logger.

    - INFO and above are sent to stderr.
    - DEBUG and above are written to the run log file.

    handle(line) records subprocess stdout without forwarding human-readable
    text to the SSE channel.
    """

    def __init__(self, log_path: str):
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._logger = logging.getLogger(f"neuroevo.{log_path}")
        self._logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self._logger.addHandler(ch)

    def info(self, msg: str)  -> None: self._logger.info(msg)
    def debug(self, msg: str) -> None: self._logger.debug(msg)
    def warn(self, msg: str)  -> None: self._logger.warning(msg)
    def error(self, msg: str) -> None: self._logger.error(msg)

    def handle(self, line: str) -> None:
        """
        Record a stdout line from the neuroevolution subprocess.

        JSON telemetry is kept at DEBUG level in the log file; human-readable
        stdout is recorded as INFO.
        """
        try:
            json.loads(line)
            self._logger.debug(line)
        except (json.JSONDecodeError, ValueError):
            self._logger.info(line)

    def close(self) -> None:
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)


def make_log_fns(logger: NeuroLogger):
    return logger.info, logger.debug
