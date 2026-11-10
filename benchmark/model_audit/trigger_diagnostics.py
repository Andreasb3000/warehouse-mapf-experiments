"""Read-only physical diagnostics for the matched native safety experiments.

Safety-region exposure describes proximity. It is not a causal decomposition
of delay: the native client records one selected blocker, visits robots before
boxes at equal severity, and may report overlaps during turns or service.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

REPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPORT / "code"))
from fleet_execution import objects


def analyze_triggers(directory, *, separation_tolerance_m=0.015):
    """Return exposure, completion/parking and sampled body-clearance evidence.

    Two streaming log passes retain only sparse state-change/acknowledgement
    events, not the full trajectory. Safety changes have physical pose clocks.
    Native goal acknowledgements use the server clock; their physical completion
    boundary is acknowledgement_clock + 1. Samples cover [0, duration), so the
    final safety update at duration is not included in selected exposure.
    """
    directory = Path(directory)
    scenario = json.loads((directory / "scenario.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    tasks = json.loads((directory / "tasks.json").read_text())["tasks"]
    parameters = config["parameters"]
    frequency = float(parameters["ticks_per_second"])
    horizon = parameters["sim_duration"]
    if (not math.isfinite(frequency) or frequency <= 0 or
            type(horizon) is not int or horizon <= 0 or
            not math.isfinite(separation_tolerance_m) or separation_tolerance_m < 0):
        raise ValueError("Invalid recording clock or separation tolerance")
    dt = 1.0 / frequency
    ids = sorted(agent["id"] for agent in scenario["agents"])
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Expected distinct robot IDs")
    id_set = set(ids)
    task_by_id = {task["robot_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks) or set(task_by_id) != id_set:
        raise ValueError("Expected one original task per robot")
    goals = {i: tuple(task_by_id[i]["stages"][-1]["location"]) for i in ids}
    all_service = np.unique(np.array([stage["location"] for task in tasks
                                     for stage in task["stages"]], dtype=float), axis=0)
    effective = {agent["id"]: {**scenario.get("safety", {}), **agent.get("safety", {})}
                 for agent in scenario["agents"]}
    radii = np.array([effective[i].get("robot_radius_m", .085) for i in ids], float)
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Invalid configured body radii")

    changes, acknowledgements, summaries, configurations = [], [], [], []
    for event in objects(directory / "simulator.log"):
        kind = event["type"]
        if kind == "safety_state":
            clock, robot = event["clock"], event["agent_id"]
            if (type(clock) is not int or not 0 <= clock <= horizon or robot not in id_set
                    or event["mode"] not in ("clear", "slow", "stop")):
                raise ValueError("Invalid safety-state event")
            if event["mode"] != "clear":
                if event.get("blocker_kind") not in ("robot", "box"):
                    raise ValueError("Unknown selected blocker kind")
                if event["blocker_kind"] == "robot" and int(event["blocker_id"]) not in id_set:
                    raise ValueError("Unknown selected robot blocker")
            changes.append(event)
        elif kind == "goal_reached":
            acknowledgements.append(event)
        elif kind == "safety_summary":
            summaries.append(event)
        elif kind == "safety_config":
            configurations.append(event)
    if len(summaries) != 1 or len(configurations) != 1:
        raise ValueError("Expected one native safety configuration and summary")
    for collection in (summaries[0]["robots"], configurations[0]["robots"]):
        if len(collection) != len(ids) or {r["agent_id"] for r in collection} != id_set:
            raise ValueError("Native safety robot set differs from scenario")
    for robot in configurations[0]["robots"]:
        for key, value in effective[robot["agent_id"]].items():
            if key not in robot or robot[key] != value:
                raise ValueError(f"Native safety configuration mismatch: {key}")

    progress = Counter()
    completion_clock = {}
    for event in sorted(acknowledgements, key=lambda e: e["clock"]):
        robot, clock = event["agent"], event["clock"]
        if robot not in id_set or type(clock) is not int or not 0 <= clock < horizon:
            raise ValueError("Invalid original-task acknowledgement")
        task = task_by_id[robot]
        index = progress[robot]
        if index >= len(task["stages"]):
            raise ValueError("Extra native task acknowledgement")
        if math.dist((event["row"], event["col"]), task["stages"][index]["location"]) > .02:
            raise ValueError("Acknowledged task differs from original assigned stage")
        progress[robot] += 1
        if progress[robot] == len(task["stages"]):
            completion_clock[robot] = clock + 1

    counter_names = ("slow_region_ticks", "stop_region_ticks", "robot_outer_trigger_ticks",
                     "obstacle_outer_trigger_ticks", "polygon_only_trigger_ticks",
                     "slowed_move_ticks", "stopped_command_ticks")
    native_ticks = {key: 0 for key in counter_names}
    for robot in summaries[0]["robots"]:
        for key in counter_names:
            value = robot[key]
            if type(value) is not int or not 0 <= value <= horizon:
                raise ValueError(f"Invalid native counter: {key}")
            native_ticks[key] += value

    changes.sort(key=lambda e: (e["clock"], e["agent_id"]))
    change_keys = [(e["clock"], e["agent_id"]) for e in changes]
    if len(change_keys) != len(set(change_keys)):
        raise ValueError("Duplicate safety-state clock for one robot")
    current, change_index, frames = {}, 0, 0
    exposure, modes, reasons, pairs = Counter(), Counter(), Counter(), Counter()
    completed_away_ticks = Counter()
    completed_max_distance = {i: 0.0 for i in completion_clock}
    pair_a, pair_b = np.triu_indices(len(ids), 1)
    pair_radii = radii[pair_a] + radii[pair_b]
    minimum_separation = math.inf
    minimum_clearance = math.inf
    closest = None
    overlaps = 0
    overlap_pairs = set()
    for event in objects(directory / "simulator.log"):
        if event["type"] != "tick":
            continue
        clock = event["clock"]
        if clock != frames or clock >= horizon:
            raise ValueError("Missing, duplicated or out-of-order native pose frame")
        agents = {agent["id"]: agent for agent in event["agents"]}
        if len(event["agents"]) != len(ids) or set(agents) != id_set:
            raise ValueError("Pose robot set differs from assigned fleet")
        positions = np.array([[-agents[i]["x"], -agents[i]["y"]] for i in ids], float)
        if not np.isfinite(positions).all():
            raise ValueError("Nonfinite native pose")
        locations = dict(zip(ids, positions))
        at_service = dict(zip(ids, (np.linalg.norm(positions[:, None] - all_service[None], axis=2)
                                   <= .25).any(axis=1)))
        while change_index < len(changes) and changes[change_index]["clock"] <= clock:
            state = changes[change_index]
            current[state["agent_id"]] = state
            change_index += 1
        for robot in ids:
            if completion_clock.get(robot, horizon) <= clock:
                distance = float(np.linalg.norm(locations[robot] - goals[robot]))
                completed_max_distance[robot] = max(completed_max_distance[robot], distance)
                if distance > .25:
                    completed_away_ticks[robot] += 1
                continue
            exposure["active_robot_ticks"] += 1
            state = current.get(robot)
            if state is None or state["mode"] == "clear":
                continue
            kind = state["blocker_kind"]
            exposure["any_selected_trigger_ticks"] += 1
            exposure[f"{kind}_selected_trigger_ticks"] += 1
            exposure["recipient_at_service_ticks" if at_service[robot]
                     else "recipient_away_from_service_ticks"] += 1
            modes[state["mode"]] += 1
            reasons[state.get("trigger_reason", "unspecified")] += 1
            if kind == "robot":
                blocker = int(state["blocker_id"])
                if completion_clock.get(blocker, horizon) <= clock:
                    exposure["finished_blocker_ticks"] += 1
                    distance = float(np.linalg.norm(locations[blocker] - goals[blocker]))
                    exposure["finished_blocker_at_delivery_bay_ticks" if distance <= .25
                             else "finished_blocker_away_from_delivery_bay_ticks"] += 1
                    pairs[(robot, blocker)] += 1
        if len(pair_a):
            distances = np.linalg.norm(positions[pair_a] - positions[pair_b], axis=1)
            minimum_separation = min(minimum_separation, float(distances.min()))
            clearances = distances - pair_radii
            index = int(clearances.argmin())
            if clearances[index] < minimum_clearance:
                minimum_clearance = float(clearances[index])
                closest = {"clock": clock, "time_s": clock * dt,
                           "robot_ids": [ids[pair_a[index]], ids[pair_b[index]]],
                           "center_distance_m": float(distances[index]),
                           "sum_configured_radii_m": float(pair_radii[index])}
            violations = np.flatnonzero(clearances < -separation_tolerance_m)
            overlaps += len(violations)
            overlap_pairs.update((ids[pair_a[j]], ids[pair_b[j]]) for j in violations)
        frames += 1
    if frames != horizon:
        raise ValueError("Incomplete native pose recording")

    expected_keys = ("active_robot_ticks", "any_selected_trigger_ticks", "robot_selected_trigger_ticks",
                     "box_selected_trigger_ticks", "recipient_at_service_ticks",
                     "recipient_away_from_service_ticks", "finished_blocker_ticks",
                     "finished_blocker_at_delivery_bay_ticks", "finished_blocker_away_from_delivery_bay_ticks")
    exposure_seconds = {key.removesuffix("_ticks") + "_s": exposure[key] * dt for key in expected_keys}
    return {
        "schema_version": 1,
        "run_directory": str(directory.resolve()),
        "native_pose_frames": frames,
        "clock_definition": "Pose/safety physical clock; delivery completion observed at server acknowledgement + 1 tick; selected exposure uses [0,duration).",
        "selected_exposure": {
            "units": "Robot-seconds before recipient completes its original delivery, including service/turns.",
            **exposure_seconds,
            "by_response_mode_s": {key: value * dt for key, value in sorted(modes.items())},
            "by_trigger_reason_s": {key: value * dt for key, value in sorted(reasons.items())},
            "finished_blocker_pairs": [{"recipient": i, "finished_blocker": j, "robot_seconds": value * dt}
                                       for (i, j), value in sorted(pairs.items())],
            "limitation": "One selected blocker only; robots are selected before boxes at equal severity, so other overlaps may coexist. Exposure does not establish an executed MOVE was slowed or quantify causal delay.",
        },
        "native_whole_recording": {
            "ticks": native_ticks,
            "robot_seconds": {key.removesuffix("_ticks") + "_s": value * dt
                              for key, value in native_ticks.items()},
            "scope": "All robots over the entire native run, including completed robots. Robot and obstacle region counters are nonexclusive; polygon-only excludes body and obstacle triggers. slowed_move_ticks counts actual capped MOVE updates, without blocker attribution.",
        },
        "completed_robot_motion": {
            "completed_robots": len(completion_clock),
            "robots_ever_more_than_0_25m_from_delivery_bay_after_completion": len(completed_away_ticks),
            "away_from_delivery_bay_robot_s": sum(completed_away_ticks.values()) * dt,
            "max_distance_from_delivery_bay_m": max(completed_max_distance.values(), default=None),
            "per_robot": [{"robot_id": i, "physical_completion_s": completion_clock[i] * dt,
                           "away_robot_s": completed_away_ticks[i] * dt,
                           "max_distance_from_delivery_bay_m": completed_max_distance[i]}
                          for i in sorted(completion_clock)],
            "scope": "Geometric check of the stated parking behavior; completion acknowledges delivery and does not itself prove subsequent parking.",
        },
        "sampled_separation": {
            "sample_dt_s": dt,
            "tolerance_m": separation_tolerance_m,
            "minimum_center_distance_m": None if not math.isfinite(minimum_separation) else minimum_separation,
            "minimum_configured_disc_clearance_m": None if not math.isfinite(minimum_clearance) else minimum_clearance,
            "closest_pair": closest,
            "pair_samples_below_sum_radii_minus_tolerance": int(overlaps),
            "distinct_pairs_below_tolerance": [list(pair) for pair in sorted(overlap_pairs)],
            "scope": "Rounded 0.1s pose samples and configured diagnostic body discs; 0.015m tolerance covers coordinate rounding. This is not a continuous collision guarantee or obstacle-contact check.",
        },
    }
