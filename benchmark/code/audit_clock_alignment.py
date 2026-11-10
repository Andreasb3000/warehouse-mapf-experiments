#!/usr/bin/env python3
"""Verify archived forecast poses against native physics, not RPC tick labels.

ARGoS advances physics before the pre-controller snapshot. Its physics clock is
one tick ahead of the server's current controller step. The visualizer records
that same physical pose at the ARGoS clock. Renaming only one timeline would
introduce an error into otherwise aligned historical recordings.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from features import REPORT, pose_array
from experiments.prediction.joint_evaluate import read_run


def main():
    manifest = json.loads((REPORT / "data/manifest.json").read_text())
    rows = []
    for entry in manifest["runs"]:
        _, _, _, snapshots, frames, _ = read_run(Path(entry["run_directory"]))
        max_coordinate_error = 0.0
        max_heading_error = 0.0
        offsets = set()
        compared = 0
        for event in snapshots:
            observation = event["clock"]
            server = event["snapshot"]["server_clock"]
            offsets.add(observation - server)
            if event["pose_frame"]["clock"] != observation or event["snapshot"]["clock"] != observation:
                raise ValueError("Snapshot's pose and action observation clocks differ")
            for controller in event["controllers"]:
                if controller["clock"] != observation:
                    raise ValueError("Controller observation clock differs from exact pose")
                if controller["poll_phase"] != controller["control_step_count"] % controller["poll_interval_ticks"]:
                    raise ValueError("Recorded polling phase disagrees with native controller count")
            if observation not in frames:
                continue
            exact, logged = pose_array(event["pose_frame"]), pose_array(frames[observation])
            coordinate_error = float(np.abs(exact[:, :2] - logged[:, :2]).max())
            heading_error = float(np.abs(np.arctan2(np.sin(exact[:, 2] - logged[:, 2]),
                                                  np.cos(exact[:, 2] - logged[:, 2]))).max())
            if coordinate_error > .00501 or heading_error > .00501:
                raise ValueError("Exact origin pose does not match same-clock visualizer pose within 0.01-unit rounding")
            max_coordinate_error = max(max_coordinate_error, coordinate_error)
            max_heading_error = max(max_heading_error, heading_error)
            compared += 1
        if offsets != {1}:
            raise ValueError("Unexpected archived physics/server clock relationship")
        rows.append({"run_id": entry["run_id"], "matching_snapshot_frames": compared,
                     "physics_minus_server_ticks": sorted(offsets),
                     "max_coordinate_rounding_error_m": max_coordinate_error,
                     "max_heading_rounding_error_rad": max_heading_error})
    result = {"status": "passed", "recordings": len(rows), "matched_origin_frames": sum(row["matching_snapshot_frames"] for row in rows),
              "conclusion": "Archived passive prediction snapshots and visualizer trajectories already share the physics observation clock. Their 1-tick offset from server control clocks does not require relabeling or retraining the offline dataset.",
              "live_bridge_requirement": "Live requests use server clock for handshake/goal bookkeeping, and a separate observation_clock equal to the unchanged nested pose/action/controller clocks for causal pose history. Controller polling phase must remain untouched.",
              "rows": rows,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = REPORT / "results/clock_alignment_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "recordings", "matched_origin_frames", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
