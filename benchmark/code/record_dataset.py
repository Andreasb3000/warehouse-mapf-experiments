#!/usr/bin/env python3
"""Record disjoint workloads on the fixed nine-zone warehouse map.

This script preserves native simulator code and pins the existing Docker image.
Only assignment seeds vary; both safety rules receive identical starts/tasks.
The manifest is atomically updated after each structurally valid recording.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent / "simulator"
sys.path.insert(0, str(ROOT))
from experiments.baseline.analyze import _objects
from experiments.junction.scenario import load_scenario
from experiments.zones.generate import build_case, render_svg
from experiments.zones.run import docker_prefix, environment, flags, mounts, native_paths
from runtime import host_info, validate_native_image

TRIGGERS = ("body_in_outer", "outer_overlap")
WORKLOADS = [(seed, "train" if seed <= 7006 else "validation" if seed <= 7008 else "test")
             for seed in range(7001, 7012)]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_workload(seed, duration_s=180):
    map_document, scenario, config, tasks = build_case(3)
    generator = random.Random(seed)
    destinations = [(zone["id"], cell) for zone in scenario["zones"]
                    for cell in zone["bays"]["dropoffs"]]
    # Uniform permutation conditioned on no local-zone deliveries. The bounded
    # rejection loop is deterministic for the workload seed and recorded below.
    for attempt in range(100000):
        generator.shuffle(destinations)
        if all(destination[0] != agent["zone_id"]
               for agent, destination in zip(scenario["agents"], destinations)):
            break
    else:
        raise RuntimeError("Failed to construct a cross-zone destination permutation")
    zone_index = {z["id"]: i for i, z in enumerate(scenario["zones"])}
    allocation = [[0] * 9 for _ in range(9)]
    assignment = []
    for agent, task, (destination_zone, destination_cell) in zip(
            scenario["agents"], tasks["tasks"], destinations):
        agent["destination_zone_id"] = destination_zone
        agent["goals"][1] = list(destination_cell)
        task["destination_zone_id"] = destination_zone
        task["stages"][1]["zone_id"] = destination_zone
        task["stages"][1]["location"] = list(destination_cell)
        allocation[zone_index[agent["zone_id"]]][zone_index[destination_zone]] += 1
        assignment.append({"robot_id": agent["id"], "source_zone": agent["zone_id"],
                           "destination_zone": destination_zone, "start": agent["start"],
                           "pickup": agent["goals"][0], "dropoff": agent["goals"][1]})
    scenario["network"]["allocation_matrix"] = allocation
    tasks["allocation_matrix"] = allocation
    scenario["notes"].append(f"Report workload {seed}: randomized unique cross-zone delivery bays; fixed starts and local pickups.")
    config["parameters"].update(seed=seed, sim_duration=duration_s * 10)
    config["record_plans"] = True
    return map_document, scenario, config, tasks, {
        "workload_seed": seed, "permutation_attempts": attempt + 1,
        "distribution": "uniform permutation of 27 distinct dropoff bays conditioned on cross-zone deliveries",
        "assignments": assignment, "allocation_matrix": allocation}


def check_recording(run_directory, duration_s, trigger):
    events = list(_objects((run_directory / "simulator.log").read_text()))
    frames = [e for e in events if e.get("type") == "tick"]
    snapshots = [e for e in events if e.get("type") == "prediction_snapshot"]
    policies = [e for e in events if e.get("type") == "safety_config"]
    if len(frames) != duration_s * 10 or len(snapshots) != duration_s:
        raise ValueError(f"Incomplete recording: {len(frames)} frames / {len(snapshots)} snapshots")
    if len(policies) != 1 or len(policies[0]["robots"]) != 27 or any(
            r["robot_trigger"] != trigger or r["inner_stop_enabled"] or not r["enabled"]
            for r in policies[0]["robots"]):
        raise ValueError("Native safety settings disagree with workload settings")
    ids = set(range(27))
    if any({p["id"] for p in frame["agents"]} != ids for frame in frames):
        raise ValueError("Robot IDs differ across recorded frames")
    for snapshot in snapshots:
        clock = snapshot["clock"]
        if (clock % 10 or snapshot["snapshot"]["clock"] != clock
                or snapshot["pose_frame"]["clock"] != clock
                or {c["id"] for c in snapshot["controllers"]} != ids
                or {r["id"] for r in snapshot["snapshot"]["robots"]} != ids
                or {r["id"] for r in snapshot["pose_frame"]["agents"]} != ids):
            raise ValueError("Incomplete or incoherent prediction snapshot")
    return {"valid": True, "frames": len(frames), "snapshots": len(snapshots),
            "robot_ids": sorted(ids), "safety_policy_verified": True,
            "validation_scope": "recording completeness and origin state alignment; no forecast scoring or task outcome selection"}


def record_case(docker, image_id, directory, seed, split, trigger, duration_s, provenance):
    run_id = f"w{seed}_{trigger}"
    run_directory = directory / run_id
    map_document, scenario, config, tasks, assignments = make_workload(seed, duration_s)
    scenario["safety"].update(robot_trigger=trigger, inner_stop_enabled=False)
    config["image"] = image_id
    expected = {"map.json": map_document, "scenario.json": scenario,
                "config.json": config, "tasks.json": tasks, "assignment.json": assignments}
    if run_directory.exists():
        for name, value in expected.items():
            if json.loads((run_directory / name).read_text()) != value:
                raise ValueError(f"Cannot resume different existing inputs: {run_directory / name}")
        integrity = json.loads((run_directory / "recording_integrity.json").read_text())
        if not integrity.get("valid"):
            raise ValueError(f"Cannot reuse invalid recording: {run_directory}")
        print(f"Reusing {run_id}", flush=True)
    else:
        run_directory.mkdir(parents=True)
        for name, value in expected.items():
            write_json(run_directory / name, value)
        (run_directory / "layout.svg").write_text(render_svg(map_document, scenario))
        load_scenario(run_directory / "scenario.json", map_document["layout"], 27)
        parameters = dict(config["parameters"], container=True, visualizer="web", save_stats=True,
                          scenario_file="/inputs/scenario.json", stats_name="/results/stats.json",
                          argos_config_filepath="/results/simulation.argos", output_log="/results/simulator.log")
        container = "predictor-report-" + uuid.uuid4().hex[:10]
        command = docker + ["run", "--rm", "--init", "--name", container, "--platform", "linux/amd64",
                            "--ulimit", "nofile=65536:65536", *mounts(run_directory), *environment(config),
                            "-v", f"{run_directory}:/results", image_id, "python3", "run_lifelong.py",
                            "/inputs/map.json", *flags(parameters)]
        write_json(run_directory / "command.json", command)
        write_json(run_directory / "provenance.json", dict(provenance, run_id=run_id,
                   workload_seed=seed, split=split, trigger=trigger,
                   input_sha256={name: sha256(run_directory / name) for name in expected}))
        print(f"Recording {run_id} ({split})", flush=True)
        started = time.perf_counter()
        try:
            with (run_directory / "launcher.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                               timeout=config["wall_timeout_s"])
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            subprocess.run(docker + ["stop", "--time", "2", container], capture_output=True)
            raise
        runtime = time.perf_counter() - started
        integrity = dict(check_recording(run_directory, duration_s, trigger), wall_runtime_s=runtime)
        write_json(run_directory / "recording_integrity.json", integrity)
        print(f"Ready {run_id}: {runtime:.2f}s wall, {integrity['frames']} frames", flush=True)
    return {"run_id": run_id, "workload_seed": seed, "split": split, "trigger": trigger,
            "run_directory": str(run_directory), "integrity": integrity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "data")
    parser.add_argument("--image", default="warehouse-congestion:local")
    parser.add_argument("--docker-host", default=None, help="Override the active Docker context/DOCKER_HOST")
    parser.add_argument("--duration-s", type=int, help="Simulated seconds: 180 normally, 40 with --quick")
    parser.add_argument("--quick", action="store_true", help="Functional dataset: one workload per split, 40 s by default")
    parser.add_argument("--seeds", type=int, nargs='+', choices=[seed for seed, _ in WORKLOADS])
    parser.add_argument("--triggers", nargs='+', choices=TRIGGERS, default=list(TRIGGERS))
    args = parser.parse_args()
    args.duration_s = args.duration_s if args.duration_s is not None else (40 if args.quick else 180)
    seeds = args.seeds or ([7001, 7007, 7009] if args.quick else [seed for seed, _ in WORKLOADS])
    workloads = [(seed, split) for seed, split in WORKLOADS if seed in seeds]
    if args.duration_s <= 0:
        parser.error("duration must be positive")
    output = args.output.resolve()
    recordings = output / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    docker = docker_prefix(args.docker_host)
    image_id = subprocess.check_output(docker + ["image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    validate_native_image(docker, image_id)
    compiled = json.loads(subprocess.check_output(docker + ["run", "--rm", "--platform", "linux/amd64", image_id,
                          "cat", "/usr/project/server-source-sha256.json"], text=True))
    checkout = {str(p.relative_to(ROOT)): sha256(p) for p in native_paths()}
    if compiled != checkout:
        raise RuntimeError("Native image differs from checkout; rebuild separately before recording")
    provenance = {"image_id": image_id, "compiled_native_source_sha256": compiled,
                  "compiled_sources_match_checkout": True,
                  "recording_script_sha256": sha256(__file__),
                  "mounted_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                       (ROOT / "run_lifelong.py", ROOT / "experiments/junction/scenario.py")},
                  "generator_source_sha256": sha256(ROOT / "experiments/zones/generate.py"),
                  "host": host_info(),
                  "native_execution": "Linux/amd64 Docker image; host-dependent emulation may apply"}
    manifest = {"schema_version": 1, "status": "recording", "map": "fixed nine-zone 27-robot map",
                "duration_s": args.duration_s, "ticks_per_second": 10, "snapshot_interval_s": 1,
                "split_unit": "whole randomized destination-assignment workload; both safety policies stay together",
                "workload_splits": {split: [seed for seed, s in workloads if s == split]
                                    for split in ("train", "validation", "test")},
                "quick_functional_run": args.quick,
                "expected_runs": len(workloads) * len(args.triggers), "provenance": provenance, "runs": []}
    write_json(output / "manifest.json", manifest)
    try:
        for seed, split in workloads:
            for trigger in args.triggers:
                manifest["runs"].append(record_case(docker, image_id, recordings, seed, split, trigger,
                                                    args.duration_s, provenance))
                write_json(output / "manifest.json", manifest)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_json(output / "manifest.json", manifest)
        raise
    write_json(output / "manifest.json", manifest)
    print(f"Complete: {len(manifest['runs'])} recordings in {output}")


if __name__ == "__main__":
    main()
