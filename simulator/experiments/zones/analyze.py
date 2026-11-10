#!/usr/bin/env python3
"""Measure transport tasks, zone traffic and actually applied safety caps."""

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline.analyze import analyze as validate_recording, _objects
from experiments.zones.queue_metrics import THRESHOLDS, build_approaches, measure_approach, summarize_queues


def resolved_safety(scenario, agent):
    """Missing switches preserve recordings made before selectable triggers."""
    return {"robot_trigger": "body_in_outer", "inner_stop_enabled": True,
            **scenario.get("safety", {}), **agent.get("safety", {})}


def validate_safety_configuration(events, scenario):
    """Compare native effective settings against the requested policy, per robot."""
    records = [event for event in events if event.get("type") == "safety_config"]
    enabled = any(resolved_safety(scenario, agent).get("enabled", False)
                  for agent in scenario["agents"])
    if len(records) != 1:
        return None, (["Safety must emit exactly one effective configuration"] if enabled else [])
    robots = records[0].get("robots", [])
    expected = {agent["id"]: resolved_safety(scenario, agent) for agent in scenario["agents"]}
    errors = []
    if len(robots) != len(expected) or {robot["agent_id"] for robot in robots} != set(expected):
        errors.append("Native safety configuration does not contain the expected robots")
    for robot in robots:
        agent_id = robot["agent_id"]
        if agent_id not in expected:
            continue
        actual = {"robot_trigger": "body_in_outer", "inner_stop_enabled": True, **robot}
        for key, value in expected[agent_id].items():
            if key not in actual or actual[key] != value:
                errors.append(f"Robot {agent_id} native safety {key} does not match requested value")
    return records[0], errors


def percentile(values, probability):
    """Linear empirical percentile, including zero-valued recorded samples."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def in_bounds(row, col, bounds):
    return (bounds["rows"][0] - .5 <= row < bounds["rows"][1] + .5 and
            bounds["cols"][0] - .5 <= col < bounds["cols"][1] + .5)


def payload_events(events, tasks):
    """Station completion acknowledgements, not planner commitment counters."""
    expected = {task["robot_id"]: task for task in tasks}
    stages = {robot_id: 0 for robot_id in expected}
    result, errors = [], []
    for event in events:
        if event.get("type") != "goal_reached":
            continue
        robot_id = event["agent"]
        task = expected.get(robot_id)
        if task is None:
            errors.append(f"Unknown robot in goal acknowledgement: {robot_id}")
            continue
        index = stages[robot_id]
        if index >= len(task["stages"]):
            errors.append(f"Robot {robot_id} acknowledged an extra task")
            continue
        stage = task["stages"][index]
        row, col = stage["location"]
        if math.hypot(event["row"] - row, event["col"] - col) > .02:
            errors.append(f"Robot {robot_id} acknowledged goals in the wrong order")
            continue
        result.append({"clock": event["clock"], "agent_id": robot_id,
                       "task_id": task["id"], "payload_id": task["payload_id"],
                       "operation": stage["operation"], "zone_id": stage["zone_id"],
                       "loaded": stage["operation"] == "pickup"})
        stages[robot_id] += 1
    return sorted(result, key=lambda item: (item["clock"], item["agent_id"])), errors


def analyze(run_dir):
    run_dir = Path(run_dir).resolve()
    input_dir = run_dir if (run_dir / "config.json").is_file() else run_dir.parent
    config = json.loads((input_dir / "config.json").read_text())
    params = config["parameters"]
    scenario = json.loads((input_dir / config["scenario"]).read_text())
    tasks = json.loads((input_dir / scenario["tasks_file"]).read_text())
    validation = validate_recording(run_dir, params, stress=True)
    errors = list(validation["errors"])
    if errors:
        result = {"valid": False, "errors": errors, "recording": validation}
        (run_dir / "zone_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    frames = [json.loads(line) for line in (run_dir / "trajectories.jsonl").read_text().splitlines()]
    events = list(_objects((run_dir / "simulator.log").read_text(errors="replace")))
    safety_config, safety_errors = validate_safety_configuration(events, scenario)
    errors.extend(safety_errors)
    transitions, task_errors = payload_events(events, tasks["tasks"])
    errors.extend(task_errors)
    with (run_dir / "payload_events.jsonl").open("w") as stream:
        for event in transitions:
            stream.write(json.dumps(event) + "\n")
    rows = []
    for task in tasks["tasks"]:
        robot_id = task["robot_id"]
        pickup = next((e["clock"] for e in transitions if e["agent_id"] == robot_id and e["operation"] == "pickup"), None)
        dropoff = next((e["clock"] for e in transitions if e["agent_id"] == robot_id and e["operation"] == "dropoff"), None)
        final = next(p for p in frames[-1]["agents"] if p["id"] == robot_id)
        goal = task["stages"][-1]["location"]
        first = next(p for p in frames[0]["agents"] if p["id"] == robot_id)
        start = scenario["agents"][robot_id]["start"]
        if math.hypot(-first["x"] - start[0], -first["y"] - start[1]) > .02:
            errors.append(f"Robot {robot_id} did not start at its assigned bay")
        rows.append({"agent_id": robot_id, "task_id": task["id"], "payload_id": task["payload_id"],
                     "source_zone": task["source_zone_id"], "destination_zone": task["destination_zone_id"],
                     "pickup_completed_s": None if pickup is None else pickup / params["ticks_per_second"],
                     "dropoff_completed_s": None if dropoff is None else dropoff / params["ticks_per_second"],
                     "final_payload_state": "delivered" if dropoff is not None else "loaded" if pickup is not None else "awaiting_pickup",
                     "final_dropoff_error_m": math.hypot(-final["x"] - goal[0], -final["y"] - goal[1])})
    with (run_dir / "transport_tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dt = 1 / params["ticks_per_second"]
    histories = {a["id"]: deque(maxlen=round(THRESHOLDS["stationary_window_s"] / dt) + 1) for a in scenario["agents"]}
    service_cells = {tuple(s["location"]) for task in tasks["tasks"] for s in task["stages"]}
    delivered_at = {row["agent_id"]: row["dropoff_completed_s"] for row in rows}
    safety_states = {a["id"]: "clear" for a in scenario["agents"]}
    changes = sorted([e for e in events if e.get("type") == "safety_state"], key=lambda e: e["clock"])
    change_index = 0
    zone_rows = []
    approaches = build_approaches(scenario)
    queue_samples = []
    robot_radii = {agent["id"]: resolved_safety(scenario, agent).get("robot_radius_m", .085)
                   for agent in scenario["agents"]}
    minimum_separation = math.inf
    for frame in frames:
        time_s = frame["clock"] * dt
        while change_index < len(changes) and changes[change_index]["clock"] <= frame["clock"]:
            event = changes[change_index]
            safety_states[event["agent_id"]] = event["mode"]
            change_index += 1
        counts = {z["id"]: {"clock": frame["clock"], "time_s": time_s, "zone_id": z["id"],
                            "robots": 0, "junction_robots": 0, "stationary_active_robots": 0,
                            "slow_region_robots": 0, "stop_region_robots": 0} for z in scenario["zones"]}
        stationary_active_ids = set()
        for pose in frame["agents"]:
            robot_id = pose["id"]
            row, col = -pose["x"], -pose["y"]
            history = histories[robot_id]; history.append(pose)
            zone = next((z for z in scenario["zones"] if in_bounds(row, col, z["bounds"])), None)
            if zone is None:
                errors.append(f"Robot {robot_id} lies outside every zone at tick {frame['clock']}")
                continue
            count = counts[zone["id"]]; count["robots"] += 1
            count["junction_robots"] += in_bounds(row, col, zone["conflict_zone"])
            count["slow_region_robots"] += safety_states[robot_id] == "slow"
            count["stop_region_robots"] += safety_states[robot_id] == "stop"
            delivered = delivered_at[robot_id] is not None and time_s >= delivered_at[robot_id]
            at_service = any(math.hypot(row-r, col-c) <= THRESHOLDS["service_bay_exclusion_radius_m"] for r, c in service_cells)
            if delivered or at_service or len(history) < history.maxlen:
                continue
            anchor = history[0]
            stationary = all(math.hypot(p["x"]-anchor["x"], p["y"]-anchor["y"]) <= THRESHOLDS["stationary_displacement_m"] and
                             abs(math.atan2(math.sin(p["rz"]-anchor["rz"]), math.cos(p["rz"]-anchor["rz"]))) <= THRESHOLDS["stationary_heading_change_rad"]
                             for p in history)
            count["stationary_active_robots"] += stationary
            if stationary:
                stationary_active_ids.add(robot_id)
        for approach in approaches:
            queue_samples.append({"clock": frame["clock"], "time_s": time_s,
                                  **measure_approach(approach, frame["agents"], stationary_active_ids, robot_radii)})
        zone_rows.extend(counts.values())
        for index, pose in enumerate(frame["agents"]):
            for other in frame["agents"][index+1:]:
                minimum_separation = min(minimum_separation, math.hypot(pose["x"]-other["x"], pose["y"]-other["y"]))
    with (run_dir / "zone_traffic.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(zone_rows[0])); writer.writeheader(); writer.writerows(zone_rows)
    queue_summary = summarize_queues(queue_samples, approaches, dt)
    (run_dir / "queue_approaches.json").write_text(json.dumps({"approaches": approaches, "thresholds": THRESHOLDS}, indent=2) + "\n")
    with (run_dir / "approach_queues.csv").open("w", newline="") as stream:
        fields = ["clock", "time_s", "approach_id", "queued_robot_ids", "queued_robots", "queue_tail_extent_m",
                  "queue_body_span_m", "tail_to_upstream_exit_m", "spillback_warning"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({**sample, "queued_robot_ids": json.dumps(sample["queued_robot_ids"])} for sample in queue_samples)
    safety_summaries = [e for e in events if e.get("type") == "safety_summary"]
    safety_enabled = any(resolved_safety(scenario, agent).get("enabled", False)
                         for agent in scenario["agents"])
    if safety_enabled and len(safety_summaries) != 1:
        errors.append("Enabled safety policy must emit exactly one terminal summary")
    safety_summary = safety_summaries[-1] if safety_summaries else None
    slowed_ticks = stopped_ticks = 0
    if safety_summary:
        expected_ids = set(range(params["num_agents"]))
        if ({r["agent_id"] for r in safety_summary["robots"]} != expected_ids or
                len(safety_summary["robots"]) != len(expected_ids)):
            errors.append("Safety summary does not contain the expected robots")
        for record in safety_summary["robots"]:
            if type(record["agent_id"]) is not int or record["agent_id"] not in expected_ids:
                continue
            agent = scenario["agents"][record["agent_id"]]
            settings = resolved_safety(scenario, agent)
            limit = params["velocity"] * settings["speed_scale"]
            if record["max_slow_wheel_command_cm_s"] > limit + 1e-4:
                errors.append(f"Robot {record['agent_id']} exceeded its slow wheel command cap")
            slowed_ticks += record["slowed_move_ticks"]
            stopped_ticks += record["stopped_command_ticks"]
            if not settings["inner_stop_enabled"] and record["stopped_command_ticks"]:
                errors.append(f"Robot {record['agent_id']} stopped commands with inner stopping disabled")
            if settings["robot_trigger"] == "body_in_outer" and record.get("polygon_only_trigger_ticks", 0):
                errors.append(f"Robot {record['agent_id']} counted polygon-only triggers in body mode")
    trigger_fields = ("polygon_only_trigger_ticks", "robot_outer_trigger_ticks", "obstacle_outer_trigger_ticks")
    trigger_totals = {field: None for field in trigger_fields}
    if safety_summary:
        for field in trigger_fields:
            records = safety_summary["robots"]
            if all(field in record for record in records):
                for record in records:
                    value = record[field]
                    if type(value) is not int or not 0 <= value <= safety_summary["clock"]:
                        errors.append(f"Invalid {field} for robot {record['agent_id']}")
                trigger_totals[field] = dt * sum(record[field] for record in records)
    per_zone = {}
    for zone in scenario["zones"]:
        samples = [r for r in zone_rows if r["zone_id"] == zone["id"]]
        per_zone[zone["id"]] = {"kind": zone["kind"], "peak_robots": max(r["robots"] for r in samples),
                               "peak_junction_robots": max(r["junction_robots"] for r in samples),
                               "peak_stationary_active_robots": max(r["stationary_active_robots"] for r in samples),
                               "p95_stationary_active_robots": percentile([r["stationary_active_robots"] for r in samples], .95),
                               "stationary_active_robot_seconds": dt * sum(r["stationary_active_robots"] for r in samples)}
    queue_by_clock = {}
    for row in zone_rows:
        queue_by_clock[row["clock"]] = queue_by_clock.get(row["clock"], 0) + row["stationary_active_robots"]
    completion_times = [row["dropoff_completed_s"] for row in rows if row["dropoff_completed_s"] is not None]
    modes = sorted({resolved_safety(scenario, agent)["robot_trigger"] for agent in scenario["agents"]})
    inner_values = {resolved_safety(scenario, agent)["inner_stop_enabled"] for agent in scenario["agents"]}
    result = {"valid": not errors, "errors": errors, "warnings": validation["warnings"],
              "recording": validation["summary"], "scenario": scenario["name"], "zones": per_zone,
              "seed": params.get("seed"), "tasks_total": len(rows),
              "deliveries_unfinished": len(rows) - len(completion_times),
              "last_delivery_completed_s": max(completion_times) if completion_times else None,
              "all_deliveries_completed_s": max(completion_times) if len(completion_times) == len(rows) and rows else None,
              "mean_completed_delivery_time_s": sum(completion_times) / len(completion_times) if completion_times else None,
              "p95_completed_delivery_time_s": percentile(completion_times, .95),
              "stationary_active_robot_seconds": dt * sum(queue_by_clock.values()),
              "peak_stationary_active_robots": max(queue_by_clock.values()),
              "p95_stationary_active_robots": percentile(list(queue_by_clock.values()), .95),
              "queue_proxy": queue_summary,
              **{key: queue_summary[key] if queue_summary["available"] else None for key in
                 ("queued_robot_seconds", "max_queue_robots", "max_queue_tail_extent_m",
                  "spillback_warning_approach_seconds", "max_simultaneous_spillback_warning_approaches")},
              "pickups_completed": sum(r["pickup_completed_s"] is not None for r in rows),
              "deliveries_completed": sum(r["dropoff_completed_s"] is not None for r in rows),
              "robots_loaded_at_end": sum(r["final_payload_state"] == "loaded" for r in rows),
              "robots_parked_at_dropoff": sum(r["final_dropoff_error_m"] <= .15 for r in rows),
              "safety_enabled": safety_enabled, "slowed_move_robot_seconds": dt * slowed_ticks,
              "stopped_command_robot_seconds": dt * stopped_ticks,
              "robot_trigger": modes[0] if len(modes) == 1 else "mixed",
              "inner_stop_enabled": next(iter(inner_values)) if len(inner_values) == 1 else "mixed",
              "safety_config": safety_config,
              "polygon_only_trigger_robot_seconds": trigger_totals["polygon_only_trigger_ticks"],
              "robot_outer_trigger_robot_seconds": trigger_totals["robot_outer_trigger_ticks"],
              "obstacle_outer_trigger_robot_seconds": trigger_totals["obstacle_outer_trigger_ticks"],
              "sampled_min_robot_separation_m": minimum_separation if math.isfinite(minimum_separation) else None,
              "safety_summary": safety_summary,
              "notes": ["Payload state changes only on station completion acknowledgements. Payload mass/identity are metadata, not physical load dynamics.",
                        "Stationary active counts exclude completed deliveries, turning and service bays; they are a queue proxy, not a causal traffic-blocking classification.",
                        "Safety region occupancy includes idle robots; slowed_move and stopped_command counters measure commands actually affected.",
                        "Polygon-only trigger time counts enabled observer-robot ticks with another outer polygon overlapping and no robot body or obstacle inside the observer outer polygon. It may include parked robots; it is not slowed-command time.",
                        "Completed delivery times start at simulation initialization. They are finite-workload completion times, not delays relative to isolated travel; unfinished tasks are reported separately.",
                        "Peak and p95 stationary-active counts cover all zones. Approach queue proxies additionally require downstream headings and a connected group close to a junction entry; recorded thresholds and approach rectangles define them explicitly.",
                        "Spillback warnings mark a queue tail approaching the upstream junction boundary; they do not prove causal upstream obstruction. Isolated-travel delay is not measured.",
                        "Safety uses exact local geometry from simulator ground truth, without sensor noise/occlusion or radio simulation.",
                        "Speed-cap and sampled-separation checks are experimental validation, not a continuous collision-safety guarantee."]}
    (run_dir / "zone_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(); print(json.dumps(analyze(args.run_dir), indent=2))
