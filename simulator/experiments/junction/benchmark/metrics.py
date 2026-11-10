#!/usr/bin/env python3
"""Difficulty measurements from executed poses, with unfinished trips retained."""

import argparse
from collections import deque
import csv
import json
import math
from pathlib import Path


def shortest_distance(layout, start, goal):
    start, goal = tuple(start), tuple(goal)
    frontier = deque([(start, 0)])
    seen = {start}
    while frontier:
        (row, col), distance = frontier.popleft()
        if (row, col) == goal:
            return distance
        for cell in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            r, c = cell
            if 0 <= r < len(layout) and 0 <= c < len(layout[0]) and layout[r][c] not in "@T" and cell not in seen:
                seen.add(cell)
                frontier.append((cell, distance + 1))
    raise ValueError(f"Unreachable goal: {start} -> {goal}")


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def approach_distance(row, col, approach, zone):
    r0, r1 = zone["rows"]
    c0, c1 = zone["cols"]
    if approach == "west" and r0 - .5 <= row <= r1 + .5 and col < c0 - .5:
        return c0 - .5 - col
    if approach == "east" and r0 - .5 <= row <= r1 + .5 and col > c1 + .5:
        return col - c1 - .5
    if approach == "north" and c0 - .5 <= col <= c1 + .5 and row < r0 - .5:
        return r0 - .5 - row
    if approach == "south" and c0 - .5 <= col <= c1 + .5 and row > r1 + .5:
        return row - r1 - .5
    return None


def measure(run_dir):
    run_dir = Path(run_dir)
    root = run_dir.parent
    config = json.loads((root / "config.json").read_text())
    scenario = json.loads((root / config["scenario"]).read_text())
    layout = json.loads((root / scenario["map_file"]).read_text())["layout"]
    execution = json.loads((run_dir / "junction_metrics.json").read_text())
    if not execution["valid"]:
        raise ValueError(f"Invalid recording: {execution['errors']}")
    frames = [json.loads(line) for line in (run_dir / "trajectories.jsonl").read_text().splitlines()]
    params = config["parameters"]
    dt = 1 / params["ticks_per_second"]
    # Last observed sample, not the unobserved right edge of the last time bin.
    horizon = frames[-1]["clock"] * dt
    velocity = params["velocity"] / 100
    records = []
    for agent, observed in zip(scenario["agents"], execution["robots"]):
        if len(agent["goals"]) != 1:
            raise ValueError("Benchmark metrics currently require one destination per robot")
        distance = shortest_distance(layout, agent["start"], agent["goals"][0])
        lower_bound = distance / velocity
        if lower_bound <= 0:
            raise ValueError("Benchmark trips must have positive length")
        parked = observed["parked_from_s"]
        elapsed = horizon if parked is None else parked
        records.append({
            "agent_id": agent["id"], "approach": agent["approach"],
            "shortest_distance_m": distance, "travel_time_lower_bound_s": lower_bound,
            "parked_at_s": parked, "unfinished": parked is None,
            "observed_elapsed_s": elapsed,
            "observed_time_ratio": elapsed / lower_bound,
            "observed_excess_over_lower_bound_s": max(0, elapsed - lower_bound),
            "stationary_approach_s": 0.0,
        })
    # Require two seconds without translation OR appreciable rotation; exclude
    # the first two seconds and count only before the first junction entry.
    # This is a measured queue proxy, not a claim that every stop was forced.
    window = max(1, round(2 / dt))
    histories = {agent["id"]: deque(maxlen=window + 1) for agent in scenario["agents"]}
    queue_trace = []
    max_queue = 0
    max_back = 0.0
    for frame in frames:
        time_s = frame["clock"] * dt
        by_id = {pose["id"]: pose for pose in frame["agents"]}
        counts = {a["approach"]: 0 for a in scenario["agents"]}
        for agent, observed, record in zip(scenario["agents"], execution["robots"], records):
            pose = by_id[agent["id"]]
            history = histories[agent["id"]]
            history.append(pose)
            if len(history) < window + 1 or time_s < 2:
                continue
            if observed["first_entry_s"] is not None and time_s >= observed["first_entry_s"]:
                continue
            distance = approach_distance(-pose["x"], -pose["y"], agent["approach"], scenario["conflict_zone"])
            if distance is None:
                continue
            anchor = history[0]
            stationary = all(math.hypot(p["x"] - anchor["x"], p["y"] - anchor["y"]) <= .1
                             and abs(math.atan2(math.sin(p["rz"] - anchor["rz"]), math.cos(p["rz"] - anchor["rz"]))) <= .1
                             for p in history)
            if stationary:
                record["stationary_approach_s"] += dt
                counts[agent["approach"]] += 1
                max_back = max(max_back, distance)
        max_queue = max(max_queue, sum(counts.values()))
        queue_trace.append({"time_s": time_s, "total": sum(counts.values()), **counts})
    total = len(records)
    finished = [r for r in records if not r["unfinished"]]
    approaches = {}
    for name in sorted({r["approach"] for r in records}):
        subset = [r for r in records if r["approach"] == name]
        approaches[name] = {
            "agents": len(subset), "parked": sum(not r["unfinished"] for r in subset),
            "stationary_robot_seconds": sum(r["stationary_approach_s"] for r in subset),
            "p90_observed_time_ratio": quantile([r["observed_time_ratio"] for r in subset], .9),
        }
    stats = execution["baseline"]
    calls = stats["n_mapf_calls"]
    fallback = stats["n_rule_based_calls"]
    result = {
        "scenario": scenario["name"], "agents": total, "observed_horizon_s": horizon,
        "parked_agents": len(finished), "completion_fraction": len(finished) / total,
        "clearance_time_s": execution["all_parked_by_s"],
        "completed_by_60s": sum(r["parked_at_s"] is not None and r["parked_at_s"] <= 60 for r in records),
        "physical_completions_per_minute": len(finished) * 60 / (params["sim_duration"] * dt),
        "median_observed_time_ratio": quantile([r["observed_time_ratio"] for r in records], .5),
        "p90_observed_time_ratio": quantile([r["observed_time_ratio"] for r in records], .9),
        "unfinished_ratios_are_lower_bounds": bool(total - len(finished)),
        "stationary_approach_robot_seconds": sum(r["stationary_approach_s"] for r in records),
        "peak_stationary_approach_robots": max_queue, "furthest_stationary_approach_distance_m": max_back,
        "peak_junction_occupancy": execution["peak_junction_occupancy"],
        "planner_calls": calls, "fallback_calls": fallback,
        "fallback_fraction": fallback / calls if calls else None,
        "wall_runtime_s": execution.get("wall_runtime_s"), "approaches": approaches,
        "notes": [
            "Actual sampled poses determine completion, not the early native task counter.",
            "Time ratio uses obstacle-aware grid distance / maximum speed. It includes turning, acceleration, startup and waiting; it is not isolated-run congestion delay.",
            "Unfinished trips remain in time-ratio statistics at their observed age. Their final ratios can only be larger; do not read these as complete-trip percentiles.",
            "Stationary approach time requires <=0.1 m translation and <=0.1 rad heading change throughout the preceding 2 s, before first junction entry. It excludes that initial confirmation window and may include voluntary waiting.",
            "Fallback fraction and wall time diagnose this planner/configuration; they are separate from traffic difficulty.",
        ],
    }
    (run_dir / "difficulty_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    for filename, rows in (("difficulty_robots.csv", records), ("approach_queues.csv", queue_trace)):
        with (run_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(measure(args.run_dir), indent=2))
