#!/usr/bin/env python3
"""Run the pinned 10-robot baseline and retain validated results.

  python3 experiments/baseline/run.py             # two runs, compare outcomes
  python3 experiments/baseline/run.py --view      # localhost:3000 preview
  python3 experiments/baseline/run.py --stop-view # stop preview container

Only the Python standard library and Docker CLI are required on the host.
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
import uuid
from zoneinfo import ZoneInfo

from analyze import analyze


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIEW = "warehouse-robots-preview"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def capture(command):
    return subprocess.check_output(command, text=True).strip()


def optional_git(*args):
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker_prefix(host=None):
    """Use an explicit daemon or Docker's normal environment/context selection."""
    return ["docker"] + (["--host", host] if host else [])


def flags(parameters):
    return [f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in parameters.items()]


def inspect_runtime(docker, config):
    image = json.loads(capture(docker + ["image", "inspect", config["image"]]))[0]
    paths = ["run_lifelong.py", "ArgosConfig/__init__.py", "ArgosConfig/ToArgos.py",
             config["parameters"]["map_filepath"]]
    inspector = (
        "import hashlib,json,pathlib,sys; "
        "print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() "
        "for p in sys.argv[1:]}))"
    )
    hashes = json.loads(capture(docker + ["run", "--rm", "--platform", "linux/amd64",
                                        config["image"], "python3", "-c", inspector, *paths]))
    return {
        "image_id": image["Id"], "repo_digests": image["RepoDigests"],
        "labels": image["Config"].get("Labels", {}),
        "local_revision": optional_git("rev-parse", "HEAD"),
        "local_status": optional_git("status", "--short"),
        "runtime_file_sha256": hashes,
        "local_file_sha256": {p: sha256(ROOT / p) for p in paths},
        "runtime_files_match_checkout": all(hashes[p] == sha256(ROOT / p) for p in paths),
        "native_binary_source_revision": "Not supplied by the published image; image digest pins the binaries.",
        "host_architecture": os.uname().machine,
        "container_platform": "linux/amd64",
    }


def run_once(docker, config, output, index):
    run_dir = output / f"run-{index:02d}"
    run_dir.mkdir()
    parameters = dict(config["parameters"])
    parameters.pop("map_filepath")
    parameters.update(container=True, visualizer="web", save_stats=True,
                      stats_name="/results/stats.json",
                      argos_config_filepath="/results/simulation.argos",
                      output_log="/results/simulator.log")
    # web emits tick records on stdout; no web server or display is needed here.
    name = "warehouse-robots-baseline-" + uuid.uuid4().hex[:10]
    command = docker + [
        "run", "--rm", "--init", "--name", name, "--platform", "linux/amd64",
        "-v", f"{run_dir}:/results", "-v", f"{output / 'map.json'}:/inputs/map.json:ro",
        config["image"], "python3", "run_lifelong.py", "/inputs/map.json", *flags(parameters),
    ]
    write_json(run_dir / "command.json", command)
    print(f"Running {index}: {run_dir}", flush=True)
    timeout = max(180, 20 * parameters["sim_duration"] / parameters["ticks_per_second"])
    try:
        with (run_dir / "launcher.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        subprocess.run(docker + ["stop", "--time", "5", name], capture_output=True)
        raise
    validation = analyze(run_dir, config["parameters"])
    if result.returncode:
        validation["errors"].append(f"Docker exited with code {result.returncode}")
        validation["valid"] = False
        write_json(run_dir / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return validation


def main():
    default_host = os.environ.get("DOCKER_HOST")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--docker-host", default=default_host)
    parser.add_argument("--runs", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--view", action="store_true")
    mode.add_argument("--stop-view", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    config = json.loads(args.config.read_text())
    docker = docker_prefix(args.docker_host)
    try:
        capture(docker + ["info", "--format", "{{.ServerVersion}}"])
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit("Docker is unavailable. Start your Docker runtime or select a working Docker context.") from error
    if args.stop_view:
        subprocess.run(docker + ["stop", PREVIEW], check=True)
        return 0
    if args.view:
        params = dict(config["parameters"])
        params["map_filepath"] = "/inputs/map.json"
        command = docker + [
            "run", "-d", "--rm", "--init", "--name", PREVIEW, "--platform", "linux/amd64",
            "-p", "127.0.0.1:3000:3000", "-v",
            f"{ROOT / config['parameters']['map_filepath']}:/inputs/map.json:ro",
            config["image"], "lsmart-viz", *flags(params),
        ]
        subprocess.run(command, check=True)
        print("Preview: http://localhost:3000 — select Start Simulation.")
        print("Stop: python3 experiments/baseline/run.py --stop-view")
        return 0
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    output = HERE / "results" / f"{stamp}-{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    shutil.copyfile(ROOT / config["parameters"]["map_filepath"], output / "map.json")
    write_json(output / "provenance.json", inspect_runtime(docker, config))
    results = [run_once(docker, config, output, index) for index in range(1, args.runs + 1)]
    outcome_keys = ["success", "congested", "total_finished_tasks", "total_finished_backup_tasks", "throughput"]
    stats = [json.loads((output / f"run-{i:02d}" / "stats.json").read_text())
             if (output / f"run-{i:02d}" / "stats.json").exists() else {}
             for i in range(1, args.runs + 1)]
    outcomes = [{key: stat.get(key) for key in outcome_keys} for stat in stats]
    initial_frames = []
    for i in range(1, args.runs + 1):
        trajectory_file = output / f"run-{i:02d}" / "trajectories.jsonl"
        first = ""
        if trajectory_file.exists():
            with trajectory_file.open() as stream:
                first = stream.readline()
        initial_frames.append(json.loads(first) if first.strip() else None)
    summary = {
        "valid": all(result["valid"] for result in results),
        "runs": args.runs,
        "comparison_performed": args.runs > 1,
        "outcomes_match": all(outcome == outcomes[0] for outcome in outcomes),
        "initial_frames_match": all(frame == initial_frames[0] for frame in initial_frames),
        "outcomes": outcomes,
        "comparison_scope": "Outcome fields and initial states; asynchronous timing traces may differ.",
        "results_directory": str(output),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] and summary["outcomes_match"] and summary["initial_frames_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
