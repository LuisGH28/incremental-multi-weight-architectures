#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-level entry point for the fw3 experiment.

Usage:
  python run.py cli    [options]   -> terminal execution
  python run.py server [options]   -> dashboard at http://localhost:8765

Examples:
  python run.py cli
  python run.py cli --full
  python run.py cli --generations 50 --pop-size 100 --max-epochs 500
  python run.py cli --full --workers 0
  python run.py cli --no-dual

  python run.py server
  python run.py server --generations 50 --pop-size 100 --max-epochs 500
"""

import os
import sys

CLI_DEFAULTS = (
    "--dataset optdigits "
    "--train data/optdigits.tra "
    "--test data/optdigits.tes "
    "--generations 10 "
    "--pop-size 20 "
    "--max-epochs 50"
)

SERVER_DEFAULTS = (
    "--dataset optdigits "
    "--train data/optdigits.tra "
    "--test data/optdigits.tes "
    "--generations 50 "
    "--pop-size 100 "
    "--max-epochs 500"
)


def run_server(extra_args: str = ""):
    os.system("PYTHONPATH=. python3 src/interfaces/http/server.py " + extra_args)


def run_cli(extra_args: str = ""):
    os.system("PYTHONPATH=. python3 src/interfaces/cli/neuro.py " + extra_args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode  = sys.argv[1]
    extra = " ".join(sys.argv[2:])

    if mode == "server":
        run_server(extra or SERVER_DEFAULTS)
    elif mode == "cli":
        run_cli(extra or CLI_DEFAULTS)
    else:
        print(f"Modo desconocido: '{mode}'  —  usa 'server' o 'cli'")
        sys.exit(1)
