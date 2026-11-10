#!/usr/bin/env python3
"""Exercise both outer-field trigger rules in the real native simulator.

The following case begins with overlapping outer fields and no body inside
another field. The isolated routes check that an unobstructed straight and a
right-angle turn do not acquire false slowdown from adjacent aisle walls.
These are execution checks, not continuous collision-safety guarantees.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline.analyze import _objects, analyze
from experiments.zones.run import docker_prefix, flags, native_paths

TRIGGERS = ("body_in_outer", "outer_overlap")
SAFETY = {
    "enabled": True, "inner_stop_enabled": False,
    "slow_front_m": .7, "slow_back_m": .7, "slow_half_width_m": .3,
    "stop_front_m": .3, "stop_back_m": .15, "stop_half_width_m": .18,
    "speed_scale": .2, "robot_radius_m": .085,
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixtures():
    corridor = ["@@@@@@@"] + ["@.....@"] * 10 + ["@@@@@@@"]
    corner = [["@"] * 12 for _ in range(12)]
    for row in range(3, 10):
        for col in range(3, 6):
            corner[row][col] = "."
    for row in range(3, 6):
        for col in range(3, 10):
            corner[row][col] = "."
    return {
        "isolated_straight": {
            "layout": corridor,
            "agents": [{"id": 0, "start": [8, 3], "goals": [[2, 3]]}],
        },
        "isolated_corner": {
            "layout": ["".join(row) for row in corner],
            "agents": [{"id": 0, "start": [8, 4], "goals": [[4, 4], [4, 8]]}],
        },
        "following": {
            "layout": corridor,
            "agents": [{"id": 0, "start": [8, 3], "goals": [[2, 3]]},
                       {"id": 1, "start": [9, 3], "goals": [[3, 3]]}],
        },
    }


def require(condition, message):
    # Explicit exceptions keep the checks active even under python -O.
    if not condition:
        raise AssertionError(message)


def check_recording(directory, fixture_name, trigger, scenario, parameters):
    validation = analyze(directory, parameters, stress=True)
    require(validation["valid"], f"Invalid native recording: {validation['errors']}")
    events = list(_objects((directory / "simulator.log").read_text()))
    frames = [event for event in events if event.get("type") == "tick"]
    configurations = [event for event in events if event.get("type") == "safety_config"]
    summaries = [event for event in events if event.get("type") == "safety_summary"]
    require(len(configurations) == len(summaries) == 1, "Expected one safety configuration and summary")
    expected_ids = {agent["id"] for agent in scenario["agents"]}
    policies = {record["agent_id"]: record for record in configurations[0]["robots"]}
    stats = {record["agent_id"]: record for record in summaries[0]["robots"]}
    require(set(policies) == set(stats) == expected_ids, "Safety logs omitted a robot")
    for agent_id in expected_ids:
        policy = policies[agent_id]
        require(policy["enabled"] and not policy["inner_stop_enabled"], "Outer-only safety was not active")
        require(policy["robot_trigger"] == trigger, "Native trigger differs from requested trigger")
        for key, value in SAFETY.items():
            require(policy[key] == value, f"Native policy changed {key} for robot {agent_id}")
        record = stats[agent_id]
        require(record["stopped_command_ticks"] == 0 and record["stop_region_ticks"] == 0,
                "An outer-only case stopped a command")
        require(record["max_slow_wheel_command_cm_s"] <= 20.00001, "Slow wheel command exceeded 20 cm/s")
        if record["slowed_move_ticks"]:
            require(record["max_slow_wheel_command_cm_s"] > 0, "Slowdown counter has no nonzero wheel command")

    initial = {event["agent_id"]: event for event in events
               if event.get("type") == "safety_state" and event["clock"] == 1}
    require(set(initial) == expected_ids, "Missing first native safety state at clock 1")
    goals = [event for event in events if event.get("type") == "goal_reached"]
    acknowledgements = {agent_id: [event for event in goals if event["agent"] == agent_id]
                        for agent_id in expected_ids}
    for agent in scenario["agents"]:
        acknowledged = acknowledgements[agent["id"]]
        require(len(acknowledged) == len(agent["goals"]), "Robot did not complete its finite route")
        require([[event["row"], event["col"]] for event in acknowledged] == agent["goals"],
                "Acknowledged goals differ from the specified route")

    if fixture_name.startswith("isolated"):
        require(all(event["mode"] == "clear" for event in initial.values()), "Isolated route started slowed")
        for record in stats.values():
            require(record["slow_region_ticks"] == record["slowed_move_ticks"] == 0,
                    "An isolated route unexpectedly triggered slowdown")
            require(record["obstacle_outer_trigger_ticks"] == 0, "A clear route triggered against a wall")
        if fixture_name == "isolated_corner":
            require(max(abs(frame["agents"][0]["rz"]) for frame in frames) > .7,
                    "Corner fixture did not turn")
    else:
        require(fixture_name == "following", "Unknown fixture")
        first_poses = {pose["id"]: pose for pose in frames[1]["agents"]}
        separation = math.hypot(first_poses[0]["x"] - first_poses[1]["x"],
                                first_poses[0]["y"] - first_poses[1]["y"])
        require(abs(separation - 1) < .001, "Initial following separation changed")
        require(SAFETY["slow_front_m"] + SAFETY["robot_radius_m"] < separation <
                SAFETY["slow_front_m"] + SAFETY["slow_back_m"],
                "Fixture no longer isolates polygon overlap from body entry")
        expected_mode = "slow" if trigger == "outer_overlap" else "clear"
        require(all(event["mode"] == expected_mode for event in initial.values()),
                f"Initial response should be {expected_mode} under {trigger}")
        if trigger == "outer_overlap":
            require(all(event["trigger_reason"] == "outer_polygon_overlap" for event in initial.values()),
                    "Initial slowdown did not identify polygon overlap")
            for record in stats.values():
                require(record["polygon_only_trigger_ticks"] > 0 and record["slowed_move_ticks"] > 0,
                        "Polygon-only overlap did not actually slow both robots")
        else:
            require(all(record["polygon_only_trigger_ticks"] == 0 for record in stats.values()),
                    "Body-entry mode counted polygon-only triggers")

    sampled_minimum = None
    if len(expected_ids) == 2:
        sampled_minimum = min(math.hypot(frame["agents"][0]["x"] - frame["agents"][1]["x"],
                                         frame["agents"][0]["y"] - frame["agents"][1]["y"])
                              for frame in frames)
        require(sampled_minimum >= .17, "Recorded robot bodies overlap")
    return {
        "valid": True, "fixture": fixture_name, "robot_trigger": trigger,
        "inner_stop_enabled": False, "frames": len(frames),
        "first_native_states": list(initial.values()),
        "last_goal_acknowledged_s": max(event["clock"] for event in goals) / parameters["ticks_per_second"],
        "minimum_sampled_robot_separation_m": sampled_minimum,
        "safety_summary": summaries[0], "recording_validation": validation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="warehouse-congestion:local")
    parser.add_argument("--docker-host", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/trigger-runtime-check")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    docker = docker_prefix(args.docker_host)
    image_id = subprocess.check_output(docker + ["image", "inspect", args.image, "--format", "{{.Id}}"],
                                       text=True).strip()
    compiled = json.loads(subprocess.check_output(docker + ["run", "--rm", "--platform", "linux/amd64", image_id,
                                                            "cat", "/usr/project/server-source-sha256.json"], text=True))
    checkout = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in native_paths()}
    require(compiled == checkout, "Native image differs from checkout; rebuild before validating")
    report = {"valid": False, "image_id": image_id, "compiled_native_source_sha256": compiled,
              "compiled_sources_match_checkout": True,
              "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": {},
              "notes": ["Pose checks use rounded samples, not a continuous collision-safety guarantee.",
                        "At initial 1 m separation, 1.4 m combined field length overlaps but 0.785 m body-entry reach does not.",
                        "Both trigger rules use identical geometry, tasks, planner configuration, and seed for each fixture."]}
    write_json(output / "validation.json", report)
    try:
        for fixture_name, fixture in fixtures().items():
            for trigger in TRIGGERS:
                case_name = f"{fixture_name}-{trigger}"
                directory = output / case_name
                directory.mkdir()
                layout = fixture["layout"]
                map_data = {"name": fixture_name, "layout": layout, "n_row": len(layout), "n_col": len(layout[0]),
                            "n_endpoint": 0, "n_agent_loc": 0, "maxtime": 1000,
                            "milp_runtime": None, "optimize_wait": False, "weight": False}
                scenario = {"name": fixture_name, "map_file": "map.json", "initial_orientation": "north",
                            "safety": dict(SAFETY, robot_trigger=trigger), "agents": deepcopy(fixture["agents"])}
                parameters = dict(num_agents=len(scenario["agents"]), seed=42, n_threads=1,
                                  sim_duration=600, ticks_per_second=10, planner="PBS",
                                  task_assigner_type="distinct_one_goal", backup_solver="PIBT",
                                  single_agent_solver="SIPP", planner_invoke_policy="default", sim_window_tick=20,
                                  planning_window=20, cutoffTime=1, velocity=100, stop_at_congestion=False,
                                  rotation=False, screen=0, container=True, visualizer="web", save_stats=True,
                                  scenario_file="/results/scenario.json", stats_name="/results/stats.json",
                                  output_log="/results/simulator.log", argos_config_filepath="/results/simulation.argos")
                write_json(directory / "map.json", map_data)
                write_json(directory / "scenario.json", scenario)
                write_json(directory / "config.json", {"scenario": "scenario.json", "parameters": parameters})
                name = "trigger-test-" + uuid.uuid4().hex[:10]
                command = docker + ["run", "--rm", "--init", "--name", name, "--platform", "linux/amd64",
                                   "--ulimit", "nofile=65536:65536", "-e", "LSMART_SCENARIO_FILE=/results/scenario.json",
                                   "-v", f"{directory}:/results", "-v", f"{ROOT / 'run_lifelong.py'}:/usr/project/run_lifelong.py:ro",
                                   "-v", f"{ROOT / 'experiments/junction/scenario.py'}:/usr/project/experiments/junction/scenario.py:ro",
                                   image_id, "python3", "run_lifelong.py", "/results/map.json", *flags(parameters)]
                write_json(directory / "command.json", command)
                print(f"Running {case_name}", flush=True)
                try:
                    with (directory / "launcher.log").open("w") as stream:
                        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    subprocess.run(docker + ["stop", "--time", "2", name], capture_output=True)
                    raise
                result = check_recording(directory, fixture_name, trigger, scenario, parameters)
                result["input_sha256"] = {filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                                          for filename in ("map.json", "scenario.json", "config.json")}
                report["cases"][case_name] = result
                write_json(directory / "trigger_validation.json", result)
                write_json(output / "validation.json", report)
                print(json.dumps({"case": case_name, "valid": True,
                                  "last_goal_acknowledged_s": result["last_goal_acknowledged_s"]}), flush=True)
        report["valid"] = True
    except Exception as exc:
        report["error"] = str(exc)
        write_json(output / "validation.json", report)
        raise
    write_json(output / "validation.json", report)
    print(json.dumps({"valid": report["valid"], "cases": len(report["cases"]),
                      "report": str(output / "validation.json")}, indent=2))


if __name__ == "__main__":
    main()
