#!/usr/bin/env python3
"""Audit physical observation phase, complete barriers, and native commitments."""
from pathlib import Path
import argparse
import json
import math
import sys

REPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPORT.parent / "simulator"))
from experiments.baseline.analyze import _objects


def check(directory):
    directory = Path(directory)
    events = list(_objects((directory / "simulator.log").read_text()))
    frames = {e["clock"]: e for e in events if e.get("type") == "tick"}
    plans = {e["clock"]: e for e in events if e.get("type") == "plan_snapshot"}
    barriers = {e["clock"]: e for e in events if e.get("type") == "routing_barrier_complete"}
    requests = [json.loads(p.read_text()) for p in (directory / "bridge").glob("request_*.json")]
    requests.sort(key=lambda e: e["clock"])
    assert len(requests) == len(plans) == len(barriers) > 0
    retained_actions = 0
    pose_error = 0.0
    for request in requests:
        decision = request["clock"]
        observation = request["observation_clock"]
        assert observation == decision + 1
        assert request["snapshot"]["server_clock"] == decision
        assert request["snapshot"]["clock"] == observation
        assert request["pose_frame"]["clock"] == observation
        for controller in request["controllers"]:
            assert controller["clock"] == observation
            assert controller["control_step_count"] == decision
            assert controller["poll_phase"] == decision % controller["poll_interval_ticks"]
        assert barriers[decision]["wall_s"] >= 0
        truth = {r["id"]: r for r in frames[observation]["agents"]}
        for robot in request["pose_frame"]["agents"]:
            target = truth[robot["id"]]
            error = math.hypot(robot["x"] - target["x"], robot["y"] - target["y"])
            pose_error = max(pose_error, error)
            assert error <= math.sqrt(2) * 0.005 + 1e-7, (decision, robot["id"], error)
        before = {r["id"]: r for r in request["snapshot"]["robots"]}
        for robot in plans[decision]["robots"]:
            old = {a["node_id"]: a for a in before[robot["id"]]["actions"]}
            for action in robot["actions"]:
                if action["node_id"] > robot["committed_node_id"]:
                    continue
                previous = old[action["node_id"]]
                for key in ("type", "start", "goal", "orientation", "task_id"):
                    assert action[key] == previous[key], (decision, robot["id"], key)
                retained_actions += 1
    return {"valid": True, "decision_boundaries": len(requests),
            "retained_committed_actions_checked": retained_actions,
            "max_pose_difference_from_matching_visualizer_frame_m": pose_error,
            "observation_phase": "ARGoS physical pose clock equals server decision clock plus one tick",
            "scope": "Recorded phase, controller metadata, native plan acceptance, and preserved committed actions; not collision or predictor-accuracy certification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = check(args.run)
    (args.run / "bridge_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
