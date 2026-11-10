"""Measure the finite crossing from sampled poses, without inferring RF or safety guarantees."""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from analyze import analyze


def measure_run(run_dir, config, scenario, *, stress=False):
    run_dir = Path(run_dir)
    validation = analyze(run_dir, config, stress=stress)
    errors = list(validation["errors"])
    frames = [json.loads(line) for line in (run_dir / "trajectories.jsonl").read_text().splitlines()]
    zone = scenario["conflict_zone"]
    r0, r1 = zone["rows"]
    c0, c1 = zone["cols"]
    dt = 1 / config["ticks_per_second"]

    def in_zone(row, col):
        return r0 - 0.5 <= row <= r1 + 0.5 and c0 - 0.5 <= col <= c1 + 0.5

    def in_approach(approach, row, col):
        if approach == "west":
            return r0 - 0.5 <= row <= r1 + 0.5 and col < c0 - 0.5
        if approach == "east":
            return r0 - 0.5 <= row <= r1 + 0.5 and col > c1 + 0.5
        if approach == "north":
            return c0 - 0.5 <= col <= c1 + 0.5 and row < r0 - 0.5
        if approach == "south":
            return c0 - 0.5 <= col <= c1 + 0.5 and row > r1 + 0.5
        return False

    records = []
    for agent in scenario["agents"]:
        records.append({
            "agent_id": agent["id"], "approach": agent["approach"],
            "first_entry_s": None, "first_exit_s": None, "goal_arrival_s": None,
            "parked_from_s": None,
            "stopped_on_approach_s": 0.0, "final_goal_error_m": None,
        })
    previous = {}
    min_pair_distance = math.inf
    occupancy = []
    measure_separation = len(scenario["agents"]) <= 100
    for frame in frames:
        time_s = round(frame["clock"] * dt, 8)
        occupied = 0
        poses = {pose["id"]: pose for pose in frame["agents"]}
        for agent in scenario["agents"]:
            pose = poses.get(agent["id"])
            if pose is None:
                continue
            row, col = -pose["x"], -pose["y"]
            record = records[agent["id"]]
            inside = in_zone(row, col)
            occupied += inside
            if inside and record["first_entry_s"] is None:
                record["first_entry_s"] = time_s
            elif not inside and record["first_entry_s"] is not None and record["first_exit_s"] is None:
                record["first_exit_s"] = time_s
            goal_row, goal_col = agent["goals"][-1]
            goal_error = math.hypot(row - goal_row, col - goal_col)
            if goal_error <= 0.15 and record["goal_arrival_s"] is None:
                record["goal_arrival_s"] = time_s
            if goal_error > 0.15:
                record["parked_from_s"] = None
            elif record["parked_from_s"] is None:
                record["parked_from_s"] = time_s
            record["final_goal_error_m"] = goal_error
            if record["first_entry_s"] is None and in_approach(agent["approach"], row, col) and agent["id"] in previous:
                prev_row, prev_col = previous[agent["id"]]
                if math.hypot(row - prev_row, col - prev_col) / dt < 0.05:
                    record["stopped_on_approach_s"] += dt
            previous[agent["id"]] = row, col
            if frame["clock"] == 0 and math.hypot(row - agent["start"][0], col - agent["start"][1]) > 0.02:
                errors.append(f"Robot {agent['id']} did not start at its scripted position")
        agents = frame["agents"]
        if measure_separation:
            for index, a in enumerate(agents):
                for b in agents[index + 1:]:
                    min_pair_distance = min(min_pair_distance, math.hypot(a["x"] - b["x"], a["y"] - b["y"]))
        occupancy.append((time_s, occupied))
    expected_tasks = sum(len(agent["goals"]) for agent in scenario["agents"])
    if not stress and validation["summary"]["total_finished_tasks"] != expected_tasks:
        errors.append(f"Expected exactly {expected_tasks} scripted completions")
    if validation["summary"]["total_finished_backup_tasks"] != 0:
        errors.append("Fixture used backup destinations")
    for record in records:
        record["stopped_on_approach_s"] = round(record["stopped_on_approach_s"], 3)
        if not stress and (record["first_entry_s"] is None or record["first_exit_s"] is None):
            errors.append(f"Robot {record['agent_id']} did not traverse the junction")
        if not stress and (record["goal_arrival_s"] is None or record["final_goal_error_m"] > 0.15):
            errors.append(f"Robot {record['agent_id']} did not finish parked at its goal")
    with (run_dir / "crossings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (run_dir / "junction_occupancy.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["simulation_time_s", "robots_in_conflict_zone"])
        writer.writerows(occupancy)
    result = {
        "valid": not errors, "errors": errors, "warnings": validation["warnings"], "baseline": validation["summary"],
        "analysis_source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "baseline/analyze.py")},
        "validation_mode": "stress_recording" if stress else "complete_crossing",
        "robots_entered_junction": sum(r["first_entry_s"] is not None for r in records),
        "robots_exited_junction": sum(r["first_exit_s"] is not None for r in records),
        "peak_junction_occupancy": max((n for _, n in occupancy), default=0),
        "robots": records,
        "sampled_min_robot_separation_m": min_pair_distance if math.isfinite(min_pair_distance) else None,
        "all_parked_by_s": max(r["parked_from_s"] for r in records) if records and all(r["parked_from_s"] is not None for r in records) else None,
        "measurement_notes": f"Poses are sampled every {dt:g} seconds and rounded upstream. Arrival/parking uses a 0.15 m position tolerance; parked_from_s remains within that tolerance through the last recorded frame. Stopped time includes rotations and initialization. Separation is sampled, not a continuous collision certificate; pairwise separation is omitted above 100 robots. Stress validation checks recording integrity, not successful traffic flow.",
    }
    (run_dir / "junction_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
