#!/usr/bin/env python3
"""Validate a baseline run and export recorded ARGoS poses (meters/radians)."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def _objects(log):
    """Find JSON objects even when another process prefixes a log message."""
    decoder = json.JSONDecoder()
    for line in log.splitlines():
        offset = 0
        while True:
            start = line.find("{", offset)
            if start < 0:
                break
            try:
                value, end = decoder.raw_decode(line, start)
            except ValueError:
                offset = start + 1
                continue
            offset = end
            if isinstance(value, dict):
                yield value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def analyze(run_dir, config, *, stress=False):
    """Write trajectories.jsonl, trajectories.csv, and validation.json."""
    run_dir = Path(run_dir)
    errors = []
    warnings = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    n_agents = int(config["num_agents"])
    n_ticks = int(config["sim_duration"])
    ticks_per_second = int(config["ticks_per_second"])
    require(n_agents > 0 and n_ticks > 0 and ticks_per_second > 0, "Invalid simulation dimensions.")
    duration = n_ticks / max(1, ticks_per_second)
    try:
        log = (run_dir / "simulator.log").read_text(errors="replace")
    except OSError as exc:
        log = ""
        errors.append(f"Cannot read simulator.log: {exc}")
    launcher_path = run_dir / "launcher.log"
    launcher_log = launcher_path.read_text(errors="replace") if launcher_path.exists() else ""
    try:
        stats = json.loads((run_dir / "stats.json").read_text())
        if not isinstance(stats, dict):
            raise ValueError("expected a JSON object")
    except (OSError, ValueError) as exc:
        stats = {}
        errors.append(f"Cannot read stats.json: {exc}")

    frames = [event for event in _objects(log) if event.get("type") == "tick"]
    clocks = [frame.get("clock") for frame in frames]
    require(clocks == list(range(n_ticks)), f"Expected consecutive trajectory clocks 0..{n_ticks - 1}; found {len(frames)} frames.")
    expected_agents = set(range(n_agents))
    observed_agents = set()
    first_positions = {}
    moved_agents = set()
    with (run_dir / "trajectories.jsonl").open("w") as jsonl, (run_dir / "trajectories.csv").open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["simulation_time_s", "tick", "agent_id", "x_m", "y_m", "heading_rad"])
        bad_frames = 0
        for frame in frames:
            jsonl.write(json.dumps(frame, separators=(",", ":")) + "\n")
            agents = frame.get("agents")
            if not isinstance(agents, list):
                bad_frames += 1
                continue
            ids = [agent.get("id") for agent in agents if isinstance(agent, dict)]
            if len(ids) != n_agents or any(type(i) is not int for i in ids) or set(ids) != expected_agents:
                bad_frames += 1
                continue
            for agent in agents:
                agent_id = agent["id"]
                observed_agents.add(agent_id)
                if not all(_number(agent.get(key)) for key in ("x", "y", "z", "rx", "ry", "rz")) or not _number(frame.get("clock")):
                    bad_frames += 1
                    continue
                position = (agent["x"], agent["y"])
                first_positions.setdefault(agent_id, position)
                if position != first_positions[agent_id]:
                    moved_agents.add(agent_id)
                writer.writerow([frame["clock"] / max(1, ticks_per_second), frame["clock"], agent_id, *position, agent["rz"]])
        require(bad_frames == 0, f"Invalid agent identities or poses in {bad_frames} trajectory entries.")
    require(observed_agents == expected_agents, "Trajectory does not contain all expected agents.")
    if not stress:
        require(moved_agents == expected_agents, "At least one robot never moved in the recorded trajectory.")

    require(bool(re.search(rf"Simulation count:\s*{n_ticks}\b", log)), "Log does not confirm the requested final simulation tick.")
    require("Server closed successfully." in log, "Log does not confirm clean server shutdown.")
    require("Timeout expired" not in log + launcher_log, "Launcher reported a timeout.")
    require(stats.get("success") is True, "Simulator did not report success.")
    if stress:
        require(type(stats.get("congested")) is bool, "Simulator omitted its congestion status.")
    else:
        require(stats.get("congested") is False, "Simulator reported congestion or omitted its status.")
    total = stats.get("total_finished_tasks")
    backup = stats.get("total_finished_backup_tasks")
    throughput = stats.get("throughput")
    require(type(total) is int and (total >= 0 if stress else total > 0), "Invalid or missing completed task count." if stress else "No completed tasks reported.")
    require(type(backup) is int and type(total) is int and 0 <= backup <= total, "Invalid backup task count.")
    if _number(total) and _number(backup) and _number(throughput) and duration > 0:
        require(math.isclose(throughput, (total - backup) / duration, rel_tol=1e-8, abs_tol=1e-10), "Throughput disagrees with task count and duration.")
    else:
        errors.append("Missing or nonfinite throughput inputs.")
    invoke_ticks = stats.get("planner_invoke_ticks", [])
    valid_invokes = isinstance(invoke_ticks, list) and bool(invoke_ticks) and all(type(t) is int and 0 <= t < n_ticks for t in invoke_ticks)
    if valid_invokes:
        valid_invokes = invoke_ticks[0] == 0 and all(a < b for a, b in zip(invoke_ticks, invoke_ticks[1:]))
    require(valid_invokes, "Invalid planner invocation tick sequence.")
    require(isinstance(invoke_ticks, list) and stats.get("n_planner_invokes") == len(invoke_ticks), "Planner invocation count disagrees with its trace.")
    per_tick = stats.get("stats_per_tick", {})
    adg_stat_samples = {}
    for key in ("n_unfinished_nodes", "min_unfinished_nodes", "portion_idle_robots"):
        values = per_tick.get(key) if isinstance(per_tick, dict) else None
        adg_stat_samples[key] = len(values) if isinstance(values, list) else None
        valid = isinstance(values, list) and (0 < len(values) <= n_ticks if stress else len(values) == n_ticks)
        require(valid, f"Expected {'1..' if stress else ''}{n_ticks} entries for stats_per_tick.{key}.")
        if valid:
            require(all(_number(v) and v >= 0 and (key != "portion_idle_robots" or v <= 1) for v in values), f"Invalid values in stats_per_tick.{key}.")
    require(len(set(adg_stat_samples.values())) == 1, "ADG statistic arrays have inconsistent lengths.")
    if stress and any(count is not None and count < n_ticks for count in adg_stat_samples.values()):
        warnings.append("ADG statistics omit ticks before the first plan is installed (ADG::recordStatsPerTick). They have no timestamps and must not be directly aligned with pose frames; pose-frame coverage is validated separately.")

    result = {
        "valid": not errors,
        "validation_mode": "stress_recording" if stress else "baseline_completion",
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "frames": len(frames), "agents": len(observed_agents),
            "moving_agents": len(moved_agents), "simulation_seconds": duration,
            "total_finished_tasks": total, "total_finished_backup_tasks": backup,
            "throughput": throughput, "n_planner_invokes": stats.get("n_planner_invokes"),
            "cpu_runtime": stats.get("cpu_runtime"), "congested": stats.get("congested"),
            "n_mapf_calls": stats.get("n_mapf_calls"), "n_rule_based_calls": stats.get("n_rule_based_calls"),
            "adg_stat_samples": adg_stat_samples,
        },
        "trajectory_notes": "ARGoS poses rounded upstream to 0.01; meters and radians; map row=-x, col=-y. Frames are captured before each simulation tick.",
    }
    (run_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, help="Defaults to RUN_DIR/../config.json")
    args = parser.parse_args()
    config_path = args.config or args.run_dir.parent / "config.json"
    config = json.loads(config_path.read_text())
    result = analyze(args.run_dir, config.get("parameters", config), stress=config.get("validation_mode") == "stress")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)
