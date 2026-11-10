#!/usr/bin/env python3
"""Check reactive execution against a clear corridor in the real ARGoS runtime.

This deliberately widens the outer rectangle to touch a shelf beside the route.
It checks slowdown and stopping behavior, not a collision-safety guarantee.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/baseline"))
from analyze import _objects
from run import docker_prefix, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="warehouse-congestion:local")
    parser.add_argument("--docker-host", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/safety-runtime-check")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    docker = docker_prefix(args.docker_host)
    image = subprocess.check_output(docker + ["image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    layout = ["@@@@@@@@@@@@", "@..........@", "@..........@", "@..........@",
              "@..@@@@@@..@", "@..........@", "@@@@@@@@@@@@"]
    map_data = {"name": "safety_corridor", "layout": layout, "n_row": 7, "n_col": 12,
                "n_endpoint": 0, "n_agent_loc": 0, "maxtime": 1000,
                "milp_runtime": None, "optimize_wait": False, "weight": False}
    results = {}
    for mode in ("disabled", "slow", "stop"):
        directory = output / mode
        directory.mkdir(exist_ok=True)
        settings = {"enabled": mode != "disabled", "speed_scale": 0.2,
                    "slow_half_width_m": 0.7}
        # A deliberately excessive side stop region should keep this robot
        # stationary; this separately proves stop suppression of TURN and MOVE.
        if mode == "stop":
            settings.update(stop_half_width_m=0.7, stop_front_m=0.7,
                            stop_back_m=0.7, slow_back_m=0.8)
        scenario = {"name": "safety_corridor", "map_file": "map.json",
                    "initial_orientation": "north", "safety": settings,
                    "agents": [{"id": 0, "start": [3, 3], "goals": [[3, 8]]}]}
        (directory / "map.json").write_text(json.dumps(map_data))
        (directory / "scenario.json").write_text(json.dumps(scenario))
        params = dict(num_agents=1, seed=42, n_threads=1, sim_duration=400,
                      ticks_per_second=10, planner="PBS", task_assigner_type="distinct_one_goal",
                      backup_solver="PIBT", single_agent_solver="SIPP", planner_invoke_policy="default",
                      sim_window_tick=20, planning_window=20, cutoffTime=1, velocity=100,
                      stop_at_congestion=False, rotation=False, screen=0, container=True,
                      visualizer="web", save_stats=True, scenario_file="/results/scenario.json",
                      stats_name="/results/stats.json", output_log="/results/simulator.log",
                      argos_config_filepath="/results/simulation.argos")
        name = f"safety-test-{mode}-{uuid.uuid4().hex[:8]}"
        command = docker + ["run", "--rm", "--init", "--name", name, "--platform", "linux/amd64",
                           "--ulimit", "nofile=65536:65536", "-e", "LSMART_SCENARIO_FILE=/results/scenario.json",
                           "-v", f"{directory}:/results", "-v", f"{ROOT / 'run_lifelong.py'}:/usr/project/run_lifelong.py:ro",
                           "-v", f"{ROOT / 'experiments/junction/scenario.py'}:/usr/project/experiments/junction/scenario.py:ro",
                           image, "python3", "run_lifelong.py", "/results/map.json", *flags(params)]
        (directory / "command.json").write_text(json.dumps(command, indent=2))
        try:
            with (directory / "launcher.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            subprocess.run(docker + ["stop", "--time", "2", name], capture_output=True)
            raise
        events = list(_objects((directory / "simulator.log").read_text()))
        frames = [e for e in events if e.get("type") == "tick"]
        assert [e["clock"] for e in frames] == list(range(400)), f"Missing frames: {mode}"
        poses = [e["agents"][0] for e in frames]
        arrivals = [e["clock"] / 10 for e, p in zip(frames, poses)
                    if math.hypot(p["x"] + 3, p["y"] + 8) <= .15]
        summaries = [e for e in events if e.get("type") == "safety_summary"]
        results[mode] = {"arrival_s": arrivals[0] if arrivals else None,
                         "maximum_tick_displacement_m": max(math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                                                             for a, b in zip(poses, poses[1:])),
                         "safety_summary": summaries[0] if summaries else None}
        if mode == "disabled":
            assert arrivals and not summaries
        elif mode == "slow":
            assert arrivals and len(summaries) == 1
            stats = summaries[0]["robots"][0]
            assert stats["slowed_move_ticks"] > 100
            assert 0 < stats["max_slow_wheel_command_cm_s"] <= 20.00001
            assert results[mode]["maximum_tick_displacement_m"] <= .031
        else:
            assert not arrivals and len(summaries) == 1
            assert summaries[0]["robots"][0]["stopped_command_ticks"] > 100
            assert results[mode]["maximum_tick_displacement_m"] == 0
    assert results["slow"]["arrival_s"] > 2 * results["disabled"]["arrival_s"]
    report = {"valid": True, "image_id": image, "cases": results}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
