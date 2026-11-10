#!/usr/bin/env python3
"""Run a fixed T-junction fixture: six robots cross once, then park.

  python3 experiments/junction/run.py --build  # build modified server/PBS and run
  python3 experiments/junction/run.py          # run twice and save measurements
  python3 experiments/junction/run.py --view    # localhost:3001, Start Simulation
  python3 experiments/junction/run.py --stop-view
  python3 experiments/junction/run.py --config experiments/junction/stress/config.json --view
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/baseline"))
from run import capture, docker_prefix, flags, write_json
from measure import measure_run
from scenario import load_scenario
from experiments.zones.run import build

PREVIEW = "warehouse-robots-junction"


def optional_git(*args):
    """Source archives may be run before they are initialized as a Git repo."""
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True,
                                capture_output=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def pbs_paths():
    return sorted([*(ROOT / "planner/PBS/src").rglob("*.cpp"),
                   *(ROOT / "planner/PBS/inc").rglob("*.h")])


def check_pbs_sources(docker, image_id):
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in pbs_paths()}
    try:
        compiled = json.loads(capture(docker + ["run", "--rm", "--platform", "linux/amd64", image_id,
                                               "cat", "/usr/project/pbs-source-sha256.json"]))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise SystemExit("PBS source metadata is missing or invalid. Rebuild with --build.") from error
    if hashes != compiled:
        raise SystemExit("PBS sources differ from this image. Rebuild with --build.")
    return compiled


def mounts(inputs, scenario_name, map_name):
    return [
        "-v", f"{inputs / map_name}:/inputs/{map_name}:ro",
        "-v", f"{inputs / scenario_name}:/inputs/{scenario_name}:ro",
        "-v", f"{ROOT / 'run_lifelong.py'}:/usr/project/run_lifelong.py:ro",
        "-v", f"{HERE / 'scenario.py'}:/usr/project/experiments/junction/scenario.py:ro",
    ]


def recording_env(config):
    return ["-e", "LSMART_RECORD_PLANS=1"] if config.get("record_plans", False) else []


def execute(docker, image_id, output, config, scenario, index):
    run_dir = output / f"run-{index:02d}"
    run_dir.mkdir()
    params = dict(config["parameters"])
    params.update(container=True, visualizer="web", save_stats=True,
                  scenario_file=f"/inputs/{config['scenario']}",
                  stats_name="/results/stats.json", argos_config_filepath="/results/simulation.argos",
                  output_log="/results/simulator.log")
    name = "warehouse-junction-" + uuid.uuid4().hex[:10]
    command = docker + ["run", "--rm", "--init", "--name", name, "--platform", "linux/amd64",
                        "--ulimit", "nofile=65536:65536",
                        *recording_env(config),
                        *mounts(output, config["scenario"], scenario["map_file"]), "-v", f"{run_dir}:/results", image_id, "python3",
                        "run_lifelong.py", f"/inputs/{scenario['map_file']}", *flags(params)]
    write_json(run_dir / "command.json", command)
    print(f"Running {index}: {run_dir}", flush=True)
    started = time.monotonic()
    try:
        with (run_dir / "launcher.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=config.get("wall_timeout_s", 300))
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        subprocess.run(docker + ["stop", "--time", "5", name], capture_output=True)
        raise
    wall_seconds = time.monotonic() - started
    metrics = measure_run(run_dir, config["parameters"], scenario, stress=config.get("validation_mode") == "stress")
    metrics["wall_runtime_s"] = round(wall_seconds, 3)
    if result.returncode:
        metrics["valid"] = False
        metrics["errors"].append(f"Container exited with code {result.returncode}")
    write_json(run_dir / "junction_metrics.json", metrics)
    print(json.dumps({"valid": metrics["valid"], "errors": metrics["errors"],
                      "warnings": metrics["warnings"],
                      "tasks": metrics["baseline"]["total_finished_tasks"],
                      "all_parked_by_s": metrics["all_parked_by_s"]}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docker-host", default=os.environ.get("DOCKER_HOST"))
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--port", type=int, help="Browser port; defaults to config preview_port or 3001")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="New result directory (must not already exist)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--view", action="store_true")
    mode.add_argument("--stop-view", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    inputs = config_path.parent
    runs = args.runs if args.runs is not None else config.get("default_runs", 2)
    port = args.port if args.port is not None else config.get("preview_port", 3001)
    preview = PREVIEW if port == 3001 else f"{PREVIEW}-{port}"
    if runs < 1:
        parser.error("--runs must be positive")
    if not 1 <= port <= 65535:
        parser.error("--port must be between 1 and 65535")
    docker = docker_prefix(args.docker_host)
    scenario = json.loads((inputs / config["scenario"]).read_text())
    layout = json.loads((inputs / scenario["map_file"]).read_text())["layout"]
    load_scenario(inputs / config["scenario"], layout, config["parameters"]["num_agents"])
    if args.stop_view:
        subprocess.run(docker + ["stop", preview], check=True)
        return 0
    if args.build:
        build(docker, config["image"])
    image = json.loads(capture(docker + ["image", "inspect", config["image"]]))[0]
    image_id = image["Id"]
    native_paths = [*sorted((ROOT / "server/src").rglob("*.cpp")),
                    *sorted((ROOT / "server/inc").rglob("*.h"))]
    native_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in native_paths}
    compiled_hashes = json.loads(capture(docker + [
        "run", "--rm", "--platform", "linux/amd64", image_id,
        "cat", "/usr/project/server-source-sha256.json",
    ]))
    compiled_hashes = {k: v for k, v in compiled_hashes.items() if k.startswith("server/")}
    if compiled_hashes != native_hashes:
        raise SystemExit("Server sources differ from the compiled image. Rebuild with --build.")
    compiled_pbs = check_pbs_sources(docker, image_id)
    if args.view:
        params = dict(config["parameters"], map_filepath=f"/inputs/{scenario['map_file']}")
        subprocess.run(docker + ["run", "-d", "--rm", "--init", "--name", preview,
                                 "--ulimit", "nofile=65536:65536",
                                 *recording_env(config),
                                 "--platform", "linux/amd64", "-p", f"127.0.0.1:{port}:3000",
                                 *mounts(inputs, config["scenario"], scenario["map_file"]), "-e", f"LSMART_SCENARIO_FILE=/inputs/{config['scenario']}",
                                 image_id, "lsmart-viz", *flags(params)], check=True)
        print(f"{scenario['name']}: http://localhost:{port} — select Start Simulation.")
        return 0
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir.resolve() if args.output_dir else HERE / "results" / f"{stamp}-{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True)
    for name in (config["scenario"], scenario["map_file"]):
        shutil.copyfile(inputs / name, output / name)
    write_json(output / "config.json", config)
    source_paths = [ROOT / "run_lifelong.py", *native_paths, *pbs_paths(), HERE / "scenario.py",
                    HERE / "run.py", HERE / "measure.py",
                    ROOT / "experiments/baseline/analyze.py"]
    write_json(output / "provenance.json", {
        "image_id": image_id, "image_labels": image["Config"].get("Labels", {}),
        "base_revision": optional_git("rev-parse", "HEAD"),
        "checkout_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "compiled_server_source_sha256": compiled_hashes,
        "compiled_pbs_source_sha256": compiled_pbs,
        "compiled_server_matches_checkout": True,
        "git_diff": optional_git("diff"),
        "platform": "linux/amd64 container",
    })
    metrics = [execute(docker, image_id, output, config, scenario, i) for i in range(1, runs + 1)]
    summaries = [{"completed_tasks": m["baseline"]["total_finished_tasks"],
                  "parked_agents": sum(r["goal_arrival_s"] is not None and r["final_goal_error_m"] <= 0.15 for r in m["robots"]),
                  "congested": m["baseline"]["congested"]} for m in metrics]
    stress = config.get("validation_mode") == "stress"
    summary = {"valid": all(m["valid"] for m in metrics), "scenario": scenario["name"],
               "warnings": [f"run-{i:02d}: {warning}" for i, m in enumerate(metrics, 1) for warning in m["warnings"]],
               "validation_mode": "stress_recording" if stress else "complete_crossing",
               "runs": len(metrics), "comparison_performed": len(metrics) > 1,
               "outcomes_match": all(s == summaries[0] for s in summaries),
               "comparison_scope": "Task counts, final parking and congestion; asynchronous timing traces may differ.",
               "outcomes": summaries, "all_parked_by_s": [m["all_parked_by_s"] for m in metrics],
               "moving_agents": [m["baseline"]["moving_agents"] for m in metrics],
               "robots_entered_junction": [m["robots_entered_junction"] for m in metrics],
               "peak_junction_occupancy": [m["peak_junction_occupancy"] for m in metrics],
               "wall_runtime_s": [m["wall_runtime_s"] for m in metrics],
               "results_directory": str(output)}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] and (stress or summary["outcomes_match"]) else 1


if __name__ == "__main__":
    sys.exit(main())
