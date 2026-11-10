#!/usr/bin/env python3
"""Replay a recorded action snapshot approximately, then score held-out poses.

Example:
  python3 experiments/prediction/predict.py experiments/junction/results/RUN/run-01
"""

import argparse
from bisect import bisect_right
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.baseline.analyze import _objects


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_difference(target, current):
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def _already_passed(position, action):
    """Do not send a partly executed robot backward along a dispatched move."""
    start, goal = action["start"], action["goal"]
    dr, dc = goal[0] - start[0], goal[1] - start[1]
    length = math.hypot(dr, dc)
    if length <= 1e-9:
        return False
    along = ((position[0] - start[0]) * dr + (position[1] - start[1]) * dc) / length
    across = abs((position[0] - start[0]) * dc - (position[1] - start[1]) * dr) / length
    return along >= length and across <= 0.15


def predict_snapshot(snapshot, pose_frame, horizons=(1, 2, 5, 10), *,
                     speed_m_s=1.0, turn_rate_deg_s=90.0, station_dwell_s=2.0,
                     duration_scales=None):
    """Predict using one snapshot and one current pose frame, never future data.

    Dependencies and per-robot action order form a scheduling graph. Unknown or
    cyclic dependencies remain blocked. Dispatched nodes have already passed
    dependency release; committed but undispatched nodes still wait. The
    accepted uncommitted suffix is held fixed for this conditional forecast.
    Optional per-robot duration scales multiply move, turn, and station durations;
    dependency releases follow these changed durations. Missing robot scales are 1.
    """
    horizons = sorted(set(float(horizon) for horizon in horizons))
    if not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("Forecast horizons must be positive finite seconds")
    if (not all(math.isfinite(value) for value in
                (speed_m_s, turn_rate_deg_s, station_dwell_s)) or
            speed_m_s <= 0 or turn_rate_deg_s <= 0 or station_dwell_s < 0):
        raise ValueError("Motion rates must be positive and station dwell nonnegative")
    if pose_frame["clock"] > snapshot["clock"]:
        raise ValueError("A forecast cannot use a pose from after its origin")
    duration_scales = {} if duration_scales is None else dict(duration_scales)
    if any(not math.isfinite(value) or value <= 0 for value in duration_scales.values()):
        raise ValueError("Duration scales must be positive finite numbers")
    actual = {pose["id"]: pose for pose in pose_frame["agents"]}
    initial = {}
    nodes, robot_nodes = {}, defaultdict(list)
    finished = {robot["id"]: robot["finished_node_id"] for robot in snapshot["robots"]}
    enqueued_count = committed_count = 0

    for robot in snapshot["robots"]:
        robot_id = robot["id"]
        if robot_id not in actual:
            raise ValueError(f"Origin pose missing robot {robot_id}")
        pose = actual[robot_id]
        position = (-float(pose["x"]), -float(pose["y"]))
        yaw = float(pose.get("rz", 0.0))
        initial[robot_id] = position
        previous = None
        for action in sorted(robot["actions"], key=lambda item: item["node_id"]):
            node_id = action["node_id"]
            if node_id <= robot["finished_node_id"]:
                continue
            key = robot_id, node_id
            if key in nodes:
                raise ValueError(f"Duplicate action node {key}")
            dispatched = node_id <= robot["enqueued_node_id"]
            committed = node_id <= robot["committed_node_id"]
            enqueued_count += dispatched
            committed_count += committed
            target = tuple(float(value) for value in action["goal"])
            end_yaw = yaw
            kind = action["type"]
            if kind == "M":
                if dispatched and _already_passed(position, action):
                    target = position
                duration = _distance(position, target) / speed_m_s
            elif kind == "T":
                target = position
                end_yaw = -float(action["orientation"]) * math.pi / 2
                duration = abs(_angle_difference(end_yaw, yaw)) / math.radians(turn_rate_deg_s)
            elif kind == "S":
                # The controller stops at the goal; S.start can inherit the
                # preceding half-move and is not a new movement instruction.
                target = position
                duration = station_dwell_s
            else:
                raise ValueError(f"Unsupported action type {kind!r}")
            duration *= duration_scales.get(robot_id, 1.0)
            dependencies = set()
            if previous is not None:
                dependencies.add(previous)
            if not dispatched:
                for dependency in action.get("dependencies", []):
                    dependency = tuple(dependency)
                    if dependency[1] > finished.get(dependency[0], -1):
                        dependencies.add(dependency)
            nodes[key] = {"start": position, "goal": target, "duration": duration,
                          "type": kind, "dependencies": dependencies,
                          "start_s": math.inf, "end_s": math.inf}
            robot_nodes[robot_id].append(key)
            previous, position, yaw = key, target, end_yaw

    remaining = {key: len(node["dependencies"]) for key, node in nodes.items()}
    dependents = defaultdict(list)
    earliest = defaultdict(float)
    for key, node in nodes.items():
        for dependency in node["dependencies"]:
            dependents[dependency].append(key)
    ready = deque(key for key, count in remaining.items() if count == 0)
    while ready:
        key = ready.popleft()
        node = nodes[key]
        node["start_s"] = earliest[key]
        node["end_s"] = node["start_s"] + node["duration"]
        for dependent in dependents[key]:
            earliest[dependent] = max(earliest[dependent], node["end_s"])
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)

    predictions = {}
    for horizon in horizons:
        positions = {}
        for robot_id, origin in initial.items():
            position = origin
            for key in robot_nodes[robot_id]:
                node = nodes[key]
                if horizon < node["start_s"]:
                    break
                if horizon >= node["end_s"]:
                    position = node["goal"]
                    continue
                if node["type"] == "M":
                    fraction = (horizon - node["start_s"]) / node["duration"]
                    position = tuple(a + fraction * (b - a)
                                     for a, b in zip(node["start"], node["goal"]))
                break
            positions[robot_id] = position
        predictions[horizon] = positions
    return predictions, {
        "unfinished_action_count": len(nodes),
        "enqueued_action_count": enqueued_count,
        "committed_action_count": committed_count,
        "unresolved_action_count": sum(not math.isfinite(node["end_s"]) for node in nodes.values()),
    }


def _read_recording(run_dir):
    frames, snapshots = {}, {}
    trajectory_path = run_dir / "trajectories.jsonl"
    if trajectory_path.exists():
        for line in trajectory_path.read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                frames[event["clock"]] = event
    sources = []
    for name in ("simulator.log", "launcher.log"):
        path = run_dir / name
        if not path.exists():
            continue
        sources.append(name)
        with path.open(errors="replace") as stream:
            for line in stream:
                for event in _objects(line):
                    if event.get("type") == "tick":
                        frames.setdefault(event["clock"], event)
                    elif event.get("type") == "plan_snapshot":
                        key = event["clock"], event["plan_version"]
                        snapshots.setdefault(key, event)
    return frames, [snapshots[key] for key in sorted(snapshots)], sources


def analyze(run_dir, horizons=(1, 2, 5, 10), *, turn_rate_deg_s=90.0,
            station_dwell_s=None, moving_threshold_m=0.25):
    """Write forecasts.jsonl and prediction_metrics.json for one recorded run."""
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path = run_dir.parent / "config.json"
    config = json.loads(config_path.read_text())
    parameters = config.get("parameters", config)
    scenario = json.loads((config_path.parent / config["scenario"]).read_text())
    ticks_per_second = float(parameters["ticks_per_second"])
    if not math.isfinite(ticks_per_second) or ticks_per_second <= 0:
        raise ValueError("ticks_per_second must be positive")
    if not math.isfinite(moving_threshold_m) or moving_threshold_m < 0:
        raise ValueError("moving_threshold_m must be finite and nonnegative")
    if station_dwell_s is None:
        station_dwell_s = 20 / ticks_per_second
    speed_m_s = float(parameters["velocity"]) / 100
    horizons = sorted(set(float(h) for h in horizons))
    if not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("Forecast horizons must be positive finite seconds")
    horizon_ticks = {h: round(h * ticks_per_second) for h in horizons}
    if any(not math.isclose(h * ticks_per_second, horizon_ticks[h], abs_tol=1e-7) for h in horizons):
        raise ValueError("Each horizon must be an integer number of simulation ticks")
    frames, snapshots, log_sources = _read_recording(run_dir)
    if not snapshots:
        raise ValueError("No plan_snapshot events found; record a new run with plan snapshots enabled")
    if not frames:
        raise ValueError("No recorded robot poses found")
    clocks = sorted(frames)
    zone = scenario["conflict_zone"]

    def occupancy(positions):
        return sum(zone["rows"][0] - .5 <= row <= zone["rows"][1] + .5 and
                   zone["cols"][0] - .5 <= col <= zone["cols"][1] + .5
                   for row, col in positions.values())

    totals = {h: {"forecast_origins": 0, "robot_endpoints": 0,
                  "moving_robot_endpoints": 0,
                  "forecast_origins_with_later_plan": 0,
                  "frozen_error": 0.0, "hold_error": 0.0,
                  "frozen_moving_error": 0.0, "hold_moving_error": 0.0,
                  "frozen_occupancy_error": 0.0, "hold_occupancy_error": 0.0}
              for h in horizons}
    used = skipped_pose = skipped_future = 0
    max_pose_age_s = 0.0
    planner_sources = defaultdict(int)
    with (run_dir / "forecasts.jsonl").open("w") as output:
        for snapshot_index, snapshot in enumerate(snapshots):
            clock = snapshot["clock"]
            pose_index = bisect_right(clocks, clock) - 1
            if pose_index < 0:
                skipped_pose += 1
                continue
            pose_frame = frames[clocks[pose_index]]
            predictions, diagnostics = predict_snapshot(
                snapshot, pose_frame, horizons, speed_m_s=speed_m_s,
                turn_rate_deg_s=turn_rate_deg_s, station_dwell_s=station_dwell_s)
            # Forecasts are generated before any target pose is read.
            origin = {pose["id"]: (-pose["x"], -pose["y"]) for pose in pose_frame["agents"]}
            pose_age_s = (clock - pose_frame["clock"]) / ticks_per_second
            max_pose_age_s = max(max_pose_age_s, pose_age_s)
            used += 1
            planner_sources[snapshot.get("planner_source", "unknown")] += 1
            for horizon, positions in predictions.items():
                target_clock = clock + horizon_ticks[horizon]
                output.write(json.dumps({
                    "origin_clock": clock, "origin_s": clock / ticks_per_second,
                    "plan_version": snapshot["plan_version"],
                    "planner_source": snapshot.get("planner_source", "unknown"),
                    "horizon_s": horizon, "target_clock": target_clock,
                    "pose_clock": pose_frame["clock"], "pose_age_s": pose_age_s,
                    "diagnostics": diagnostics,
                    "robots": [{"id": robot_id, "row": position[0], "col": position[1],
                                "hold_row": origin[robot_id][0], "hold_col": origin[robot_id][1]}
                               for robot_id, position in sorted(positions.items())],
                }, separators=(",", ":")) + "\n")
                if target_clock not in frames:
                    skipped_future += 1
                    continue
                target = {pose["id"]: (-pose["x"], -pose["y"])
                          for pose in frames[target_clock]["agents"]}
                if set(positions) != set(target):
                    raise ValueError(f"Robot identities differ at target clock {target_clock}")
                result = totals[horizon]
                result["forecast_origins"] += 1
                # Evaluation-only context: this is deliberately computed
                # after the frozen forecast has been written. Later plans
                # are never supplied to predict_snapshot.
                if (snapshot_index + 1 < len(snapshots) and
                        snapshots[snapshot_index + 1]["clock"] <= target_clock):
                    result["forecast_origins_with_later_plan"] += 1
                for robot_id, position in positions.items():
                    frozen_error = _distance(position, target[robot_id])
                    hold_error = _distance(origin[robot_id], target[robot_id])
                    result["robot_endpoints"] += 1
                    result["frozen_error"] += frozen_error
                    result["hold_error"] += hold_error
                    if hold_error > moving_threshold_m:
                        result["moving_robot_endpoints"] += 1
                        result["frozen_moving_error"] += frozen_error
                        result["hold_moving_error"] += hold_error
                actual_occupancy = occupancy(target)
                result["frozen_occupancy_error"] += abs(occupancy(positions) - actual_occupancy)
                result["hold_occupancy_error"] += abs(occupancy(origin) - actual_occupancy)

    if not any(total["robot_endpoints"] for total in totals.values()):
        raise ValueError("No forecast endpoints could be scored; record more ticks or choose shorter horizons")
    metrics = []
    for horizon, total in totals.items():
        def mean(key, denominator):
            return total[key] / total[denominator] if total[denominator] else None
        row = {"horizon_s": horizon, **{key: total[key] for key in
               ("forecast_origins", "robot_endpoints", "moving_robot_endpoints",
                "forecast_origins_with_later_plan")}}
        row["fraction_origins_with_later_plan"] = mean(
            "forecast_origins_with_later_plan", "forecast_origins")
        for method in ("frozen", "hold"):
            row["frozen_action" if method == "frozen" else "hold_position"] = {
                "endpoint_mean_error_m": mean(method + "_error", "robot_endpoints"),
                "moving_endpoint_mean_error_m": mean(method + "_moving_error", "moving_robot_endpoints"),
                "junction_occupancy_mae_robots": mean(method + "_occupancy_error", "forecast_origins"),
            }
        metrics.append(row)
    result = {
        "scenario": scenario["name"], "run_directory": str(run_dir),
        "predictor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "forecast_model": "frozen accepted action graph with approximate motion",
        "snapshots_used": used, "snapshots_skipped_without_origin_pose": skipped_pose,
        "forecast_targets_without_recorded_truth": skipped_future,
        "planner_sources": dict(planner_sources), "log_sources": log_sources,
        "recording": {
            "pose_frames": len(clocks), "first_clock": clocks[0], "last_clock": clocks[-1],
            "missing_ticks_between_first_and_last": sum(max(0, b - a - 1) for a, b in zip(clocks, clocks[1:])),
            "configured_simulation_ticks": parameters.get("sim_duration"),
        },
        "max_origin_pose_age_s": max_pose_age_s,
        "parameters": {"speed_m_s": speed_m_s, "turn_rate_deg_s": turn_rate_deg_s,
                       "station_dwell_s": station_dwell_s, "moving_threshold_m": moving_threshold_m},
        "metrics": metrics,
        "notes": [
            "Only the origin snapshot and latest pose at or before its clock enter each forecast.",
            "Moving cohort means actual endpoint displacement exceeds the stated threshold; future poses are used only for scoring.",
            "Frozen actions include the accepted uncommitted suffix; future replanning is not predicted.",
            "Later-plan fractions are evaluation-only counts of scored forecasts whose horizon includes a later accepted plan.",
            "Known plans ending early hold their final position; unknown dependencies remain blocked.",
            "Motion omits acceleration, PID behavior, contact physics, and communication delays, including controller action polling and its bounded dispatch queue.",
            "An already-active station action receives a full assumed dwell because remaining controller timers are not exported.",
            "These are simulator prediction errors, not estimates of real-warehouse accuracy.",
        ],
    }
    (run_dir / "prediction_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--horizons", nargs="+", type=float, default=[1, 2, 5, 10])
    parser.add_argument("--turn-rate-deg-s", type=float, default=90.0)
    parser.add_argument("--station-dwell-s", type=float)
    args = parser.parse_args()
    try:
        result = analyze(args.run_dir, args.horizons, turn_rate_deg_s=args.turn_rate_deg_s,
                         station_dwell_s=args.station_dwell_s)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
