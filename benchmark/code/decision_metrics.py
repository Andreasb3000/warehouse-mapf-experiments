#!/usr/bin/env python3
"""Finite-workload utility and paired, workload-cluster acceptance statistics.

This module measures outcomes, not forecast accuracy. It never treats the many
robots, ticks, or overlapping prediction origins as independent experiments.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parent / "simulator"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.baseline.analyze import _objects
from experiments.zones.analyze import validate_safety_configuration
from experiments.zones.queue_metrics import THRESHOLDS, build_approaches, measure_approach


DEFAULT_PROTOCOL = {
    "schema_version": 1,
    "frozen_before_test_outcomes": True,
    "primary_regime": "cross_zone",
    "control_regime": "within_zone",
    "methods": ["zero", "persistence", "kinematic", "local_ridge", "joint"],
    "required_baselines": ["zero", "persistence"],
    "safety_triggers": ["body_in_outer", "outer_overlap"],
    "horizon_s": 240.0,
    "primary_workload_seeds": list(range(8001, 8021)),
    "control_workload_seeds": list(range(8101, 8106)),
    "practical_gain_fraction": 0.05,
    "confidence_level": 0.95,
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20261031,
    "minimum_independent_workloads": 10,
    "maximum_extra_unfinished_per_condition": 0,
    "p95_wait_relative_tolerance": 0.10,
    "p95_wait_absolute_tolerance_s": 2.0,
    "severe_queue_min_robots": 3,
    "severe_queue_relative_tolerance": 0.10,
    "severe_queue_absolute_tolerance_s": 2.0,
    "control_mean_flow_degradation_fraction": 0.02,
    "decision_budget_s": 1.0,
    "primary_definition": "Mean min(completion-release, horizon-release) plus actual synchronous query time accumulated before completion or the horizon, across every released task. Unfinished work remains in the denominator and is separately guarded. Native uncharged flow time is reported alongside it.",
    "query_charging_definition": "Each query event contributes query_s to every task released at or before origin_s whose completion clock is at or after origin_s. The barrier precedes controller goal acknowledgements at the same native tick, so equality is charged. Unfinished tasks include all queries before the native horizon. This is a conservative stop-the-world wall-time model, not a validated asynchronous execution model.",
    "interval_definition": "Percentile bootstrap of paired workload seeds. Both safety policies for a seed are resampled together. Intervals are exploratory pointwise, not multiplicity-corrected.",
    "waiting_definition": "Per robot sum of 0.1 s samples stationary for the past 2 s, excluding turns, service bays, unreleased and completed tasks. This is stationary waiting, not causal traffic delay.",
    "severe_queue_definition": "A connected stationary approach queue containing at least 3 robots, using existing geometry/heading thresholds; report time on each approach, not only fleet totals.",
    "control_definition": "Local-destination tasks are a designed low-interaction control; realized congestion is reported and not assumed absent.",
    "scope": "Screening for this map, fleet, finite task batch, routing rule and communication assumptions; not proof for new maps or thousands of robots.",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_completions(events, tasks, ticks_per_second, horizon_s):
    """Match original task stages, skipping routing waypoints and duplicates.

    Do not infer delivery from an accepted plan or from proximity to a goal.
    Only native goal acknowledgements establish stage completion.
    """
    by_robot = {task["robot_id"]: task for task in tasks}
    if len(by_robot) != len(tasks):
        raise ValueError("Only one finite pickup/delivery task per robot is supported")
    progress = {robot_id: 0 for robot_id in by_robot}
    recorded = {robot_id: {} for robot_id in by_robot}
    for event in sorted(events, key=lambda item: item.get("clock", 0)):
        if event.get("type") != "goal_reached" or event.get("agent") not in by_robot:
            continue
        robot_id = event["agent"]
        task = by_robot[robot_id]
        index = progress[robot_id]
        if index >= len(task["stages"]):
            continue
        stage = task["stages"][index]
        location = stage["location"]
        if math.hypot(event["row"] - location[0], event["col"] - location[1]) > 0.02:
            continue
        time_s = event["clock"] / ticks_per_second
        release_s = float(task.get("release_at_s", task.get("release_s", 0)))
        if time_s + 1e-8 < release_s:
            raise ValueError("Native goal acknowledged before task release")
        if time_s <= horizon_s + 1e-8:
            recorded[robot_id][stage["operation"]] = time_s
            progress[robot_id] += 1
    rows = []
    for task in tasks:
        release_s = float(task.get("release_at_s", task.get("release_s", 0)))
        if not 0 <= release_s < horizon_s:
            raise ValueError("Every benchmark task must be released before the horizon")
        completed_s = recorded[task["robot_id"]].get("dropoff")
        rows.append({"task_id": task["id"], "robot_id": task["robot_id"],
                     "release_s": release_s,
                     "pickup_completed_s": recorded[task["robot_id"]].get("pickup"),
                     "delivery_completed_s": completed_s,
                     "restricted_flow_s": (horizon_s if completed_s is None else completed_s) - release_s})
    return rows


def charge_query_time(completion_rows, query_events, horizon_s):
    """Charge measured synchronous pauses; unavailable durations are an error."""
    origins = []
    for event in query_events:
        origin, duration = float(event["origin_s"]), float(event["query_s"])
        if not math.isfinite(origin) or not math.isfinite(duration) or not 0 <= origin < horizon_s or duration < 0:
            raise ValueError("Query origins/durations must be finite and inside the native horizon")
        origins.append(origin)
    if len(origins) != len(set(origins)):
        raise ValueError("Only one total query-duration event is allowed at each barrier")
    for row in completion_rows:
        stop_s = horizon_s if row["delivery_completed_s"] is None else row["delivery_completed_s"]
        # PreStep routing barrier precedes controller goal acknowledgements at
        # the same native clock. Every validated query is strictly before H.
        pause = sum(float(event["query_s"]) for event in query_events if row["release_s"] <= event["origin_s"] <= stop_s)
        row["synchronous_query_pause_s"] = pause
        row["charged_restricted_flow_s"] = row["restricted_flow_s"] + pause
    return completion_rows


def trajectory_metrics(frames, scenario, tasks, completion_rows, dt, protocol):
    """Stationary waits, actual travel, and approach-specific queue exposure."""
    ids = sorted(task["robot_id"] for task in tasks)
    rows = {row["robot_id"]: row for row in completion_rows}
    histories = {robot_id: deque(maxlen=round(THRESHOLDS["stationary_window_s"] / dt) + 1)
                 for robot_id in ids}
    service = {tuple(stage["location"]) for task in tasks for stage in task["stages"]}
    radii = {agent["id"]: agent.get("safety", {}).get("robot_radius_m",
              scenario.get("safety", {}).get("robot_radius_m", 0.085)) for agent in scenario["agents"]}
    approaches = build_approaches(scenario)
    queue = {item["id"]: {"queued_robot_seconds": 0.0, "severe_queue_seconds": 0.0,
                         "spillback_warning_seconds": 0.0, "max_queue_robots": 0}
             for item in approaches}
    wait = {robot_id: 0.0 for robot_id in ids}
    distance = {robot_id: 0.0 for robot_id in ids}
    longest = {robot_id: 0.0 for robot_id in ids}
    streak = {robot_id: 0.0 for robot_id in ids}
    previous = {}
    for frame in sorted(frames, key=lambda item: item["clock"]):
        time_s = frame["clock"] * dt
        eligible = set()
        poses = {pose["id"]: pose for pose in frame["agents"]}
        if set(poses) != set(ids):
            raise ValueError("Frame robot set differs from assigned finite tasks")
        for robot_id in ids:
            pose, row = poses[robot_id], rows[robot_id]
            # Goal acknowledgements label the server controller step. Physics
            # has already advanced one tick when that controller runs; pose
            # frames therefore observe the completed task at server_clock + 1.
            completion_pose_s = None if row["delivery_completed_s"] is None else row["delivery_completed_s"] + dt
            active = time_s >= row["release_s"] and (completion_pose_s is None or time_s < completion_pose_s)
            active_interval = time_s >= row["release_s"] and (completion_pose_s is None or time_s <= completion_pose_s + 1e-8)
            if robot_id in previous and active_interval:
                before = previous[robot_id]
                distance[robot_id] += math.hypot(pose["x"] - before["x"], pose["y"] - before["y"])
            previous[robot_id] = pose
            history = histories[robot_id]
            history.append(pose)
            at_service = any(math.hypot(-pose["x"] - r, -pose["y"] - c) <= THRESHOLDS["service_bay_exclusion_radius_m"]
                             for r, c in service)
            stationary = False
            if active and not at_service and len(history) == history.maxlen:
                anchor = history[0]
                stationary = all(math.hypot(p["x"] - anchor["x"], p["y"] - anchor["y"]) <= THRESHOLDS["stationary_displacement_m"] and
                                 abs(math.atan2(math.sin(p["rz"] - anchor["rz"]), math.cos(p["rz"] - anchor["rz"]))) <= THRESHOLDS["stationary_heading_change_rad"]
                                 for p in history)
            if stationary:
                eligible.add(robot_id)
                wait[robot_id] += dt
                streak[robot_id] += dt
                longest[robot_id] = max(longest[robot_id], streak[robot_id])
            else:
                streak[robot_id] = 0.0
        for approach in approaches:
            sample = measure_approach(approach, frame["agents"], eligible, radii)
            q = queue[approach["id"]]
            q["queued_robot_seconds"] += dt * sample["queued_robots"]
            q["severe_queue_seconds"] += dt * (sample["queued_robots"] >= protocol["severe_queue_min_robots"])
            q["spillback_warning_seconds"] += dt * sample["spillback_warning"]
            q["max_queue_robots"] = max(q["max_queue_robots"], sample["queued_robots"])
    return {"stationary_wait_robot_seconds": sum(wait.values()),
            "p95_stationary_wait_s": float(np.quantile(list(wait.values()), 0.95)),
            "maximum_continuous_stationary_wait_s": max(longest.values()),
            "travel_distance_m": sum(distance.values()),
            "robot_stationary_wait_s": {str(k): v for k, v in wait.items()},
            "robot_travel_distance_m": {str(k): v for k, v in distance.items()},
            "approaches": queue,
            "severe_queue_approach_seconds": sum(q["severe_queue_seconds"] for q in queue.values()),
            "maximum_queue_robots": max((q["max_queue_robots"] for q in queue.values()), default=0)}


def summarize_run(run_dir, metadata=None, protocol=None):
    """Read an actual native recording; no validity or completed-task filtering."""
    protocol = {**DEFAULT_PROTOCOL, **(protocol or {})}
    metadata = metadata or {}
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    scenario = json.loads((run_dir / config["scenario"]).read_text())
    tasks_path = run_dir / scenario["tasks_file"]
    tasks = json.loads(tasks_path.read_text())["tasks"]
    events = list(_objects((run_dir / "simulator.log").read_text(errors="replace")))
    safety_configuration, safety_errors = validate_safety_configuration(events, scenario)
    if safety_errors:
        raise ValueError("Native safety configuration mismatch: " + "; ".join(safety_errors))
    if metadata.get("trigger") and scenario.get("safety", {}).get("robot_trigger", "body_in_outer") != metadata["trigger"]:
        raise ValueError("Manifest trigger differs from actual scenario")
    frames = {e["clock"]: e for e in events if e.get("type") == "tick"}
    if (run_dir / "trajectories.jsonl").exists():
        frames.update({frame["clock"]: frame for line in (run_dir / "trajectories.jsonl").read_text().splitlines()
                       if line.strip() for frame in [json.loads(line)]})
    ticks = config["parameters"]["ticks_per_second"]
    horizon = config["parameters"]["sim_duration"] / ticks
    if abs(horizon - protocol["horizon_s"]) > 1e-8:
        raise ValueError("Recorded horizon differs from frozen protocol")
    expected_clocks = set(range(config["parameters"]["sim_duration"]))
    if set(frames) != expected_clocks:
        raise ValueError("Incomplete or unexpected pose recording; do not drop a failed run silently")
    rows = task_completions(events, tasks, ticks, horizon)
    query_path = run_dir / "routing_decisions.jsonl"
    if not query_path.is_file():
        raise ValueError("Missing routing_decisions.jsonl: synchronous query costs cannot be assumed zero")
    query_events = [json.loads(line) for line in query_path.read_text().splitlines() if line.strip()]
    barrier_events = {event["clock"] / ticks: event for event in events if event.get("type") == "routing_barrier_complete"}
    if barrier_events:
        if set(barrier_events) != {event["origin_s"] for event in query_events}:
            raise ValueError("Query timing log does not match all native routing barriers")
        if any(abs(float(event["query_s"]) - float(barrier_events[event["origin_s"]]["wall_s"])) > 1e-5 for event in query_events):
            raise ValueError("Charged query time differs from native full-barrier measurement")
    charge_query_time(rows, query_events, horizon)
    completed = [row["delivery_completed_s"] - row["release_s"] for row in rows if row["delivery_completed_s"] is not None]
    query_times = [float(event["query_s"]) for event in query_events]
    plans = [event for event in events if event.get("type") == "routing_plan"]
    result = {**metadata, "run_directory": str(run_dir.resolve()), "horizon_s": horizon,
              "tasks_total": len(rows), "deliveries_completed": len(completed),
              "deliveries_unfinished": len(rows) - len(completed),
              "native_restricted_mean_flow_s": float(np.mean([row["restricted_flow_s"] for row in rows])),
              "restricted_mean_flow_s": float(np.mean([row["charged_restricted_flow_s"] for row in rows])),
              "charged_horizon_s": horizon + sum(query_times),
              "query_count": len(query_times), "total_query_pause_s": sum(query_times),
              "forecast_queries": sum(event.get("reason") == "forecast" for event in query_events),
              "queries_with_nonzero_penalties": sum(event.get("nonzero_penalty_cells", 0) > 0 for event in query_events),
              "native_routing_plans": len(plans),
              "native_plan_failures": sum(not event.get("success", False) for event in plans),
              "successful_plans_with_nonzero_penalties": sum(event.get("success", False) and event.get("nonzero_penalty_cells", 0) > 0 for event in plans),
              "intervention_count_scope": "Nonzero-cost and successful-plan counts establish routing-cost application, not that geometric routes changed.",
              "query_latency_median_s": float(np.median(query_times)) if query_times else None,
              "query_latency_p95_s": float(np.quantile(query_times, .95)) if query_times else None,
              "query_budget_exceedances": sum(value > protocol["decision_budget_s"] for value in query_times),
              "query_deadline_scope": "Budget exceedance counts compare measured synchronous cost with a fixed budget; they do not measure geometric diversion deadlines.",
              "clock_semantics": "Task completion and barrier charging use native server control-step clocks at 0.1 s resolution. Pose-based distance and waiting use physical visualizer clocks; native completion is observed in poses one tick after its server acknowledgement label.",
              "mean_completed_flow_s": float(np.mean(completed)) if completed else None,
              "all_deliveries_completed_s": max(row["delivery_completed_s"] for row in rows) if len(completed) == len(rows) else None,
              "task_outcomes": rows,
              "source_hashes": {name: sha256(run_dir / name) for name in ("config.json", "scenario.json", "tasks.json", "simulator.log", "routing_decisions.jsonl")}}
    result.update(trajectory_metrics(list(frames.values()), scenario, tasks, rows, 1 / ticks, protocol))
    return result


def paired_cluster_gain(baseline_by_seed, candidate_by_seed, protocol=None):
    """Ratio of cluster means, with both safety policies clustered together."""
    protocol = {**DEFAULT_PROTOCOL, **(protocol or {})}
    if set(baseline_by_seed) != set(candidate_by_seed) or not baseline_by_seed:
        raise ValueError("Paired workload seed sets must be nonempty and identical")
    seeds = sorted(baseline_by_seed)
    b = np.asarray([baseline_by_seed[seed] for seed in seeds], dtype=float)
    c = np.asarray([candidate_by_seed[seed] for seed in seeds], dtype=float)
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(c)) or np.any(b <= 0):
        raise ValueError("Baseline flow times must be finite and positive")
    generator = np.random.default_rng(protocol["bootstrap_seed"])
    indices = generator.integers(0, len(seeds), size=(protocol["bootstrap_resamples"], len(seeds)))
    draws = 1 - c[indices].mean(axis=1) / b[indices].mean(axis=1)
    alpha = (1 - protocol["confidence_level"]) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    gain = 1 - float(c.mean() / b.mean())
    return {"independent_workloads": len(seeds), "baseline_mean_flow_s": float(b.mean()),
            "candidate_mean_flow_s": float(c.mean()), "relative_improvement": gain,
            "ci_low": float(low), "ci_high": float(high),
            "enough_workloads": len(seeds) >= protocol["minimum_independent_workloads"],
            "practical_gain_pass": gain >= protocol["practical_gain_fraction"] - 1e-12,
            "positive_interval_pass": float(low) > 0.0,
            "paired_seed_means": [{"workload_seed": seed, "baseline_s": float(x), "candidate_s": float(y)}
                                   for seed, x, y in zip(seeds, b, c)]}


def compare_candidate(runs, candidate, baseline, protocol=None):
    """Strictly paired comparison; missing controls/cases yield incomplete status."""
    protocol = {**DEFAULT_PROTOCOL, **(protocol or {})}
    key = lambda row: (row["workload_seed"], row["trigger"], row["demand_regime"])
    def selected(method):
        rows = [row for row in runs if row["method"] == method]
        result = {key(row): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate method/workload/policy/regime condition")
        return result
    b, c = selected(baseline), selected(candidate)
    missing = sorted(set(b) ^ set(c))
    if missing or not b:
        return {"candidate": candidate, "baseline": baseline, "status": "incomplete",
                "reason": "Missing paired conditions", "missing_conditions": missing}
    primary = {k: v for k, v in b.items() if k[2] == protocol["primary_regime"]}
    seeds = sorted(protocol["primary_workload_seeds"])
    expected = {(seed, trigger, protocol["primary_regime"]) for seed in seeds for trigger in protocol["safety_triggers"]}
    expected |= {(seed, trigger, protocol["control_regime"]) for seed in protocol["control_workload_seeds"]
                 for trigger in protocol["safety_triggers"]}
    if not seeds or not expected.issubset(b):
        return {"candidate": candidate, "baseline": baseline, "status": "incomplete",
                "reason": "All frozen primary and control workload seeds require both safety policies",
                "missing_conditions": sorted(expected - set(b))}
    baseline_seed = {seed: np.mean([v["restricted_mean_flow_s"] for k, v in primary.items() if k[0] == seed]) for seed in seeds}
    candidate_seed = {seed: np.mean([c[k]["restricted_mean_flow_s"] for k in primary if k[0] == seed]) for seed in seeds}
    gain = paired_cluster_gain(baseline_seed, candidate_seed, protocol)
    violations = []
    for condition in sorted(expected):
        left, right = b[condition], c[condition]
        if left["tasks_total"] != right["tasks_total"] or left["horizon_s"] != right["horizon_s"]:
            raise ValueError("Paired conditions have unequal workload size or observation horizon")
        if left.get("source_hashes", {}).get("tasks.json") != right.get("source_hashes", {}).get("tasks.json"):
            raise ValueError("Paired conditions do not have identical task assignment files")
        if right["deliveries_unfinished"] - left["deliveries_unfinished"] > protocol["maximum_extra_unfinished_per_condition"]:
            violations.append({"condition": condition, "guardrail": "extra_unfinished",
                               "baseline": left["deliveries_unfinished"], "candidate": right["deliveries_unfinished"]})
        wait_limit = left["p95_stationary_wait_s"] + max(protocol["p95_wait_absolute_tolerance_s"],
                                                       protocol["p95_wait_relative_tolerance"] * left["p95_stationary_wait_s"])
        if right["p95_stationary_wait_s"] > wait_limit + 1e-8:
            violations.append({"condition": condition, "guardrail": "p95_stationary_wait",
                               "baseline": left["p95_stationary_wait_s"], "candidate": right["p95_stationary_wait_s"], "limit": wait_limit})
        if set(left["approaches"]) != set(right["approaches"]):
            raise ValueError("Paired approach sets differ")
        for approach in left["approaches"]:
            old = left["approaches"][approach]["severe_queue_seconds"]
            new = right["approaches"][approach]["severe_queue_seconds"]
            limit = old + max(protocol["severe_queue_absolute_tolerance_s"], protocol["severe_queue_relative_tolerance"] * old)
            if new > limit + 1e-8:
                violations.append({"condition": condition, "guardrail": "severe_queue", "approach": approach,
                                   "baseline": old, "candidate": new, "limit": limit})
    controls = {}
    for trigger in protocol["safety_triggers"]:
        keys = [k for k in expected if k[1] == trigger and k[2] == protocol["control_regime"]]
        before = np.mean([b[k]["restricted_mean_flow_s"] for k in keys])
        after = np.mean([c[k]["restricted_mean_flow_s"] for k in keys])
        degradation = float(after / before - 1)
        controls[trigger] = {"baseline_mean_flow_s": float(before), "candidate_mean_flow_s": float(after),
                             "relative_degradation": degradation,
                             "pass": degradation <= protocol["control_mean_flow_degradation_fraction"] + 1e-12}
    passed = gain["enough_workloads"] and gain["practical_gain_pass"] and gain["positive_interval_pass"] and not violations and all(v["pass"] for v in controls.values())
    policy_gain = {}
    for trigger in protocol["safety_triggers"]:
        left = {seed: b[(seed, trigger, protocol["primary_regime"])]["restricted_mean_flow_s"] for seed in seeds}
        right = {seed: c[(seed, trigger, protocol["primary_regime"])]["restricted_mean_flow_s"] for seed in seeds}
        policy_gain[trigger] = paired_cluster_gain(left, right, protocol)
    return {"candidate": candidate, "baseline": baseline, "status": "pass" if passed else "not_passed",
            "primary": gain, "guardrail_violations": violations, "control": controls,
            "primary_by_safety_trigger": policy_gain,
            "interpretation": "A failed screen is not proof the predictor cannot help; it fails this prespecified routing-rule benchmark."}


def evaluate_runs(runs, protocol=None):
    protocol = {**DEFAULT_PROTOCOL, **(protocol or {})}
    comparisons = [compare_candidate(runs, method, baseline, protocol)
                   for method in protocol["methods"] if method not in protocol["required_baselines"]
                   for baseline in protocol["required_baselines"]]
    candidates = {method: all(row["status"] == "pass" for row in comparisons if row["candidate"] == method)
                  for method in protocol["methods"] if method not in protocol["required_baselines"]}
    return {"protocol": protocol, "runs": runs, "comparisons": comparisons,
            "utility_screen_pass": candidates,
            "readiness_claim": "Utility-screen passage alone does not establish deadline feasibility, communication feasibility, safety guarantees or out-of-map generalization."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--protocol", type=Path, default=REPORT / "configs/decision_protocol.json")
    parser.add_argument("--output", type=Path, default=REPORT / "results/decision_evaluation.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Evaluate a completed manifest; do not silently analyze an interrupted selection of runs")
    if manifest.get("pilot", False):
        raise ValueError("Pilot recordings validate the implementation; they are not acceptance-test observations")
    if manifest.get("protocol") != protocol:
        raise ValueError("Manifest's frozen protocol differs from the requested evaluation protocol")
    if manifest.get("expected_runs") != len(manifest["runs"]):
        raise ValueError("Manifest run count differs from the frozen expected count")
    runs = [summarize_run(row["run_directory"], row, protocol) for row in manifest["runs"]]
    result = evaluate_runs(runs, protocol)
    result["protocol_sha256"] = sha256(args.protocol)
    result["manifest_sha256"] = sha256(args.manifest)
    result["scoring_code_sha256"] = sha256(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"runs": len(runs), "utility_screen_pass": result["utility_screen_pass"]}, indent=2))
