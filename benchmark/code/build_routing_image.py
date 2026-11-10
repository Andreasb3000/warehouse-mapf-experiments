#!/usr/bin/env python3
"""Compatibility entry point for the portable native-image builder."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable,
        str(Path(__file__).resolve().parents[1] / 'native/build.py'), *sys.argv[1:]]))
