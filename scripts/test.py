#!/usr/bin/env python3
"""Run the source-only Python tests; native C++ checks run during image build."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = ROOT / "simulator"
BENCHMARK = ROOT / "benchmark"

SUITES = {
    "simulator": (SIMULATOR, "tests"),
    "junction": (SIMULATOR, "experiments/junction"),
    "junction-benchmark": (SIMULATOR, "experiments/junction/benchmark"),
    "zones": (SIMULATOR, "experiments/zones"),
    "prediction": (SIMULATOR, "experiments/prediction"),
    "benchmark": (BENCHMARK, "code"),
}


def environment():
    """Prefer this checkout, and avoid unnecessary numerical worker pools."""
    env = os.environ.copy()
    search = [str(SIMULATOR), str(BENCHMARK / "code")]
    if env.get("PYTHONPATH"):
        search.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(search)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def run_checks(checks):
    """Run independent suites in separate interpreters to avoid name collisions."""
    failed = []
    started = time.perf_counter()
    for label, directory, arguments in checks:
        print(f"\n{label}", flush=True)
        result = subprocess.run([sys.executable, "-m", "unittest", *arguments],
                                cwd=directory, env=environment())
        if result.returncode:
            failed.append(label)
    elapsed = time.perf_counter() - started
    if failed:
        print(f"\nFailed: {', '.join(failed)} ({elapsed:.1f} s)", file=sys.stderr)
        return 1
    print(f"\nPassed {len(checks)} checks in {elapsed:.1f} s.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", choices=tuple(SUITES),
                        help="Run only the named suite; may be repeated.")
    parser.add_argument("--quiet", action="store_true", help="Show suite summaries.")
    args = parser.parse_args()
    checks = []
    for name in args.suite or SUITES:
        directory, tests = SUITES[name]
        checks.append((name, directory, ["discover", "-s", tests, "-p", "test_*.py",
                                         "-q" if args.quiet else "-v"]))
    return run_checks(checks)


if __name__ == "__main__":
    raise SystemExit(main())
