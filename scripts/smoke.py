#!/usr/bin/env python3
"""Fast CPU checks from generated fixtures; no recordings or Docker required.

This checks the Python pipeline, not native simulator execution or scientific
performance. Use the documented native smoke and full experiment commands for
those. --learning adds small CPU neural-model training and causality checks.
"""
from __future__ import annotations

import argparse

from test import BENCHMARK, SIMULATOR, run_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning", action="store_true",
                        help="Also check ConvLSTM models and tiny CPU training (requires torch).")
    parser.add_argument("--verbose", action="store_true", help="List individual checks.")
    args = parser.parse_args()
    verbosity = "-v" if args.verbose else "-q"
    checks = [
        ("27/54/81-robot layouts and causal feature extraction", BENCHMARK,
         [verbosity, "test_fleet_cases", "test_features"]),
        ("Joint motion, safety fields and dependency waiting", SIMULATOR,
         [verbosity, "experiments.prediction.test_joint"]),
        ("Shared cost maps, comparable service times and candidate selection", BENCHMARK,
         [verbosity, "test_route_eta", "test_route_search_policy"]),
    ]
    if args.learning:
        checks.append(("ConvLSTM contracts, tiny CPU training and held-out-label isolation", BENCHMARK,
                       [verbosity, "test_learned"]))
    return run_checks(checks)


if __name__ == "__main__":
    raise SystemExit(main())
