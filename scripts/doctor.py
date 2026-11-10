#!/usr/bin/env python3
"""Read-only environment checks for the documented experiment workflow."""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys


def main():
    failed = False
    print(f"Python {platform.python_version()} ({platform.system()} / {platform.machine()})")
    if sys.version_info < (3, 12):
        print("FAIL: use Python 3.12 or newer; 3.12 is the tested reference.")
        failed = True
    for name in ("numpy", "scipy", "sklearn", "matplotlib", "joblib", "threadpoolctl", "torch"):
        try:
            module = importlib.import_module(name)
            print(f"OK: {name} {getattr(module, '__version__', '')}")
        except ImportError:
            print(f"FAIL: {name} is missing; install requirements.txt and requirements-ml.txt.")
            failed = True
    if not shutil.which("docker"):
        print("FAIL: Docker CLI is missing. Python smoke checks can still run.")
        return 1
    try:
        result = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        print("FAIL: Docker engine did not respond within 15 seconds.")
        return 1
    if result.returncode:
        print("FAIL: Docker engine is unavailable. Start it and check your Docker context or DOCKER_HOST.")
        failed = True
    else:
        print(f"OK: Docker engine {result.stdout.strip()}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
