#!/usr/bin/env python3
"""Export a numeric fleet-state vector without replacing native timestamps.

The exported six features are observed kinematics, not a complete simulator
checkpoint. Accepted plans and controller memory remain ragged 1-second records
in each native simulator.log and are indexed in the accompanying JSON schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parent / "simulator"
sys.path.insert(0, str(REPO))
from experiments.prediction.joint_evaluate import read_run

FEATURE_NAMES = ["row_m", "col_m", "sin_yaw", "cos_yaw", "v_row_m_s", "v_col_m_s"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(entry):
    run_directory = Path(entry["run_directory"])
    config, scenario, _, snapshots, frames, _ = read_run(run_directory)
    clocks = np.array(sorted(frames), dtype=np.int64)
    if len(clocks) != config["parameters"]["sim_duration"] or not np.all(np.diff(clocks) == 1):
        raise ValueError("Export requires complete, consecutive native pose frames")
    ticks_per_second = config["parameters"]["ticks_per_second"]
    times = clocks / ticks_per_second
    robot_ids = np.array(sorted(agent["id"] for agent in scenario["agents"]), dtype=np.int64)
    poses = []
    for clock in clocks:
        agents = sorted(frames[int(clock)]["agents"], key=lambda agent: agent["id"])
        if [agent["id"] for agent in agents] != robot_ids.tolist():
            raise ValueError("Robot membership differs across pose frames")
        poses.append([[-agent["x"], -agent["y"], agent["rz"]] for agent in agents])
    poses = np.array(poses, dtype=np.float64)
    state = np.full((*poses.shape[:2], len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    state[:, :, :2] = poses[:, :, :2]
    state[:, :, 2] = np.sin(poses[:, :, 2])
    state[:, :, 3] = np.cos(poses[:, :, 2])
    state[1:, :, 4:] = np.diff(poses[:, :, :2], axis=0) / np.diff(times)[:, None, None]
    valid_mask = np.isfinite(state)
    if not valid_mask[:, :, :4].all() or not valid_mask[1:].all() or valid_mask[0, :, 4:].any():
        raise ValueError("Unexpected missing or nonfinite kinematic state")
    snapshot_clocks = np.array(sorted(event["clock"] for event in snapshots), dtype=np.int64)
    destination = REPORT / "data/cache" / entry["run_id"]
    destination.mkdir(exist_ok=True, parents=True)
    state_path = destination / "state_vectors.npz"
    np.savez_compressed(state_path, v_t=state, valid_mask=valid_mask,
                        clocks=clocks, time_s=times, robot_ids=robot_ids,
                        feature_names=np.array(FEATURE_NAMES), snapshot_clocks=snapshot_clocks)
    schema = {
        "schema_version": 1, "run_id": entry["run_id"], "state_shape": list(state.shape),
        "state_dimension_if_flattened": int(state.shape[1] * state.shape[2]),
        "feature_names": FEATURE_NAMES, "axis_order": ["native_pose_frame", "robot_id", "feature"],
        "coordinate_mapping": "row=-native_x; col=-native_y; yaw=native_rz in radians",
        "time_mapping": "time_s=native_clock/ticks_per_second; native clocks retained without relabeling",
        "ticks_per_second": ticks_per_second, "pose_sample_interval_s": 1 / ticks_per_second,
        "first_pose_clock": int(clocks[0]), "last_pose_clock": int(clocks[-1]),
        "velocity_definition": "backward finite difference of recorded position over0.1 seconds; first frame is NaN and invalid",
        "rounding_note": "Replay positions are rounded to0.01m; derived0.1s velocity therefore has quantization error. It is not native wheel speed.",
        "ragged_context": {"source": str(run_directory / "simulator.log"),
                           "event_type": "prediction_snapshot", "snapshot_clocks": snapshot_clocks.tolist(),
                           "cadence_s": 1, "fields": ["snapshot", "controllers", "pose_frame"],
                           "note": "The last coherent snapshot may lie one tick beyond the final recorded pose frame. Exact snapshot poses remain in the source log."},
        "source_sha256": {"simulator.log": digest(run_directory / "simulator.log"),
                          "scenario.json": digest(run_directory / "scenario.json"),
                          "config.json": digest(run_directory / "config.json"),
                          "export_state.py": digest(__file__)},
        "state_vectors_sha256": digest(state_path), "full_simulator_checkpoint": False,
    }
    (destination / "state_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Exported {entry['run_id']}: {state.shape}, clocks {clocks[0]}..{clocks[-1]}", flush=True)
    return {"run_id": entry["run_id"], "path": str(state_path), "schema": str(destination / "state_schema.json"),
            "shape": list(state.shape), "state_vectors_sha256": schema["state_vectors_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPORT / "data/manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "complete":
        raise ValueError("Finish native recording before exporting state vectors")
    exported = [export(entry) for entry in manifest["runs"]]
    (REPORT / "data/state_vectors_index.json").write_text(json.dumps({
        "runs": exported, "feature_names": FEATURE_NAMES,
        "state_dimension_per_robot": 6, "fleet_dimension": 162,
        "note": "A compact observed kinematic vector with separate ragged controller/plan context; not a sufficient Markov state for the native simulator."
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
