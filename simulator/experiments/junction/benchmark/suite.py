#!/usr/bin/env python3
"""Run matched junction variants, then evaluate difficulty and action forecasts."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from metrics import measure


def summarize(output):
    rows = []
    details = []
    for case in sorted(output.iterdir()):
        if not case.is_dir() or not (case / "config.json").exists():
            continue
        for run_dir in sorted(case.glob("run-*")):
            metrics = measure(run_dir)
            details.append({"run_dir": str(run_dir), **metrics})
            row = {key: value for key, value in metrics.items() if not isinstance(value, (list, dict))}
            row["run"] = run_dir.name
            rows.append(row)
            config = json.loads((case / "config.json").read_text())
            if config.get("record_plans"):
                subprocess.run([sys.executable, str(ROOT / "experiments/prediction/predict.py"), str(run_dir)], check=True)
    if not rows:
        raise ValueError(f"No recordings found under {output}")
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"runs": details, "metric_source_sha256": hashlib.sha256((HERE / "metrics.py").read_bytes()).hexdigest(),
              "interpretation": "Compare completion first, then time ratio and stationary approach time. These are measurements of this planner/configuration, not an intrinsic complexity proof. Repeat shortlisted cases before fixing a baseline."}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", help="Case directory names; default: all generated cases")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analyze-only", type=Path)
    args = parser.parse_args()
    if args.analyze_only:
        summarize(args.analyze_only.resolve())
        return
    if args.runs < 1:
        parser.error("--runs must be positive")
    configs = sorted((HERE / "cases").glob("*/config.json"))
    if args.cases:
        configs = [p for p in configs if p.parent.name in args.cases]
        missing = set(args.cases) - {p.parent.name for p in configs}
        if missing:
            parser.error(f"Unknown cases: {sorted(missing)}")
    if not configs:
        parser.error("No cases found. Run experiments/junction/benchmark/generate.py first.")
    output = args.output_dir.resolve() if args.output_dir else HERE.parent / "results" / ("benchmark-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    for index, config in enumerate(configs):
        command = [sys.executable, str(HERE.parent / "run.py"), "--config", str(config), "--runs", str(args.runs), "--output-dir", str(output / config.parent.name)]
        if args.build and index == 0:
            command.append("--build")
        subprocess.run(command, check=True)
    summarize(output)
    print(f"Saved benchmark: {output}")


if __name__ == "__main__":
    main()
