#!/usr/bin/env python3
"""Evaluate frozen joint forecasts using coherent native execution snapshots.

No future poses, plans, or controller states are passed to the predictor.
Future data is used only after all methods have produced their forecasts.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.baseline.analyze import _objects
from experiments.prediction.joint import rollout
from experiments.prediction.predict import predict_snapshot
from experiments.prediction.flow_metrics import (approaches_for, crossing_events,
                                                occupancy, queue_samples, zones_for)
from experiments.zones.viewer import _input_directory

METHODS = ("joint", "plan_only", "hold", "legacy")
FLOW_METHODS = METHODS[:3]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def read_run(run_dir):
    directory, config, scenario_path = _input_directory(run_dir)
    scenario = json.loads(scenario_path.read_text())
    map_document = json.loads((directory/scenario["map_file"]).read_text())
    events = list(_objects((run_dir/"simulator.log").read_text(errors="replace")))
    snapshots = [e for e in events if e.get("type") == "prediction_snapshot"]
    plans = [e["clock"] for e in events if e.get("type") == "plan_snapshot"]
    frames = {e["clock"]: e for e in events if e.get("type") == "tick"}
    if (run_dir/"trajectories.jsonl").is_file():
        frames.update({f["clock"]: f for line in (run_dir/"trajectories.jsonl").read_text().splitlines()
                       if line.strip() for f in [json.loads(line)]})
    if not snapshots:
        raise ValueError(f"{run_dir}: no coherent prediction_snapshot events; record with the new image")
    return config, scenario, map_document, snapshots, frames, plans


def to_agents(frame):
    return [{"id": a["id"], "row": -a["x"], "col": -a["y"], "yaw": a["rz"]}
            for a in frame["agents"]]


def empty_scores():
    return {"position_errors": [], "moving_errors": [], "occupancy_errors": [],
            "unexhausted_plan_errors": [], "exhausted_plan_errors": [],
            "queue_errors": [], "tail_errors": [], "truth_queue_positive": 0,
            "queue_tp": 0, "queue_fp": 0, "queue_fn": 0, "queue_tn": 0,
            "spillback_tp": 0, "spillback_fp": 0, "spillback_fn": 0, "spillback_tn": 0}


def average(values):
    return statistics.mean(values) if values else None


def finalize_scores(s):
    def ratio(a, b): return a/b if b else None
    return {"robot_endpoints": len(s["position_errors"]),
            "moving_robot_endpoints": len(s["moving_errors"]),
            "position_mae_m": average(s["position_errors"]),
            "position_p95_m": sorted(s["position_errors"])[max(0, math.ceil(.95*len(s["position_errors"]))-1)] if s["position_errors"] else None,
            "moving_position_mae_m": average(s["moving_errors"]),
            "unexhausted_plan_robot_endpoints": len(s["unexhausted_plan_errors"]),
            "unexhausted_plan_position_mae_m": average(s["unexhausted_plan_errors"]),
            "exhausted_plan_robot_endpoints": len(s["exhausted_plan_errors"]),
            "exhausted_plan_position_mae_m": average(s["exhausted_plan_errors"]),
            "junction_occupancy_mae_robots": average(s["occupancy_errors"]),
            "approach_queue_count_mae_robots": average(s["queue_errors"]),
            "approach_queue_tail_mae_m": average(s["tail_errors"]),
            "multi_robot_queue": {"tp": s["queue_tp"], "fp": s["queue_fp"],
                "fn": s["queue_fn"], "tn": s["queue_tn"],
                "precision": ratio(s["queue_tp"], s["queue_tp"]+s["queue_fp"]),
                "recall": ratio(s["queue_tp"], s["queue_tp"]+s["queue_fn"])},
            "spillback_warning": {"tp": s["spillback_tp"], "fp": s["spillback_fp"],
                "fn": s["spillback_fn"], "tn": s["spillback_tn"],
                "precision": ratio(s["spillback_tp"], s["spillback_tp"]+s["spillback_fp"]),
                "recall": ratio(s["spillback_tp"], s["spillback_tp"]+s["spillback_fn"])}}


def score_endpoint(scores, prediction, truth, origin, zones, predicted_queue=None, true_queue=None,
                   *, unexhausted_ids=None):
    actual_ids = [a["id"] for a in truth["agents"]]
    predicted_ids = [a["id"] for a in prediction["agents"]]
    initial_ids = [a["id"] for a in origin["agents"]]
    if any(len(ids) != len(set(ids)) for ids in (actual_ids, predicted_ids, initial_ids)):
        raise ValueError("Robot IDs must be unique in forecast, truth, and origin")
    if set(predicted_ids) != set(actual_ids) or set(initial_ids) != set(actual_ids):
        raise ValueError("Forecast/origin robot IDs differ from truth")
    if unexhausted_ids is not None and not set(unexhausted_ids).issubset(actual_ids):
        raise ValueError("Plan coverage IDs are not present in truth")
    queue_pairs = []
    if predicted_queue is not None and true_queue is not None:
        predicted_approaches = predicted_queue["approaches"]
        true_approaches = true_queue["approaches"]
        predicted_keys = [a["approach_id"] for a in predicted_approaches]
        true_keys = [a["approach_id"] for a in true_approaches]
        if (len(predicted_keys) != len(set(predicted_keys))
                or len(true_keys) != len(set(true_keys))):
            raise ValueError("Queue approach IDs must be unique")
        if set(predicted_keys) != set(true_keys):
            raise ValueError("Forecast queue approach IDs differ from truth")
        predicted_by_id = {a["approach_id"]: a for a in predicted_approaches}
        queue_pairs = [(predicted_by_id[a["approach_id"]], a) for a in true_approaches]
    actual = {a["id"]: a for a in truth["agents"]}
    initial = {a["id"]: a for a in origin["agents"]}
    for a in prediction["agents"]:
        b, start = actual[a["id"]], initial[a["id"]]
        error = math.hypot(a["row"]-b["row"], a["col"]-b["col"])
        scores["position_errors"].append(error)
        if unexhausted_ids is not None:
            key = "unexhausted_plan_errors" if a["id"] in unexhausted_ids else "exhausted_plan_errors"
            scores[key].append(error)
        if math.hypot(b["row"]-start["row"], b["col"]-start["col"]) > .25:
            scores["moving_errors"].append(error)
    pred_counts, actual_counts = occupancy(prediction, zones), occupancy(truth, zones)
    scores["occupancy_errors"].extend(abs(pred_counts[k]-actual_counts[k]) for k in actual_counts)
    for p, t in queue_pairs:
        scores["queue_errors"].append(abs(p["queued_robots"]-t["queued_robots"]))
        scores["tail_errors"].append(abs(p["queue_tail_extent_m"]-t["queue_tail_extent_m"]))
        for name, predicted, actual_event in (
                ("queue", p["queued_robots"] >= 2, t["queued_robots"] >= 2),
                ("spillback", p["spillback_warning"], t["spillback_warning"])):
            suffix = "tp" if predicted and actual_event else "fp" if predicted else "fn" if actual_event else "tn"
            scores[f"{name}_{suffix}"] += 1


def unexhausted_plan_ids(diagnostics, robot_ids, horizon_s):
    """Origin-only cohort shared by every method, determined by joint rollout.

    Include robots whose known accepted actions are not exhausted strictly before
    the horizon. Exclude both completed/parked robots and robots whose next task
    route is not yet known when they exhaust early. No future truth is inspected;
    this cohort does not guarantee that an accepted plan will remain unchanged.
    """
    exhausted = diagnostics["plan_exhausted_at_s"]
    if set(exhausted) != {str(ident) for ident in robot_ids}:
        raise ValueError("Plan exhaustion diagnostics must cover every robot ID")
    result = set()
    for ident in robot_ids:
        time_s = exhausted[str(ident)]
        if time_s is not None and (not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < 0):
            raise ValueError("Plan exhaustion times must be finite nonnegative values or null")
        if time_s is None or time_s >= horizon_s-1e-8:
            result.add(ident)
    return result


def score_crossings(predicted, truth):
    """Keep missed/censored exits visible; arrival MAE excludes origin occupants.

    Clearance times measure residual time from the forecast origin when a robot
    is already inside. Entry time zero in that case is not an observed arrival.
    """
    complete = {k: v for k, v in truth.items() if v["exit_s"] is not None}
    predicted_complete = {k: v for k, v in predicted.items() if v["exit_s"] is not None}
    common = set(complete) & set(predicted_complete)
    arrivals = {k for k in common if not (complete[k].get("already_inside_at_origin", False)
                                         or predicted_complete[k].get("already_inside_at_origin", False))}
    return {"actual_clearances": len(complete), "predicted_clearances": len(predicted_complete),
            "matched_clearances": len(common), "missed_clearances": len(set(complete)-set(predicted_complete)),
            "false_clearances": len(set(predicted_complete)-set(complete)),
            "actual_censored_visits": sum(v["exit_s"] is None for v in truth.values()),
            "predicted_censored_visits": sum(v["exit_s"] is None for v in predicted.values()),
            "matched_arrivals": len(arrivals),
            "initially_inside_matched_clearances": len(common-arrivals),
            "clearance_errors_s": [abs(predicted_complete[k]["exit_s"]-complete[k]["exit_s"]) for k in sorted(common)],
            "arrival_errors_s": [abs(predicted_complete[k]["entry_s"]-complete[k]["entry_s"]) for k in sorted(arrivals)]}


def evaluate_run(run_dir, output, *, horizons=(5, 10, 20), origin_stride_s=1, demo=False):
    run_dir = Path(run_dir).resolve()
    config, scenario, map_doc, snapshots, frames, plans = read_run(run_dir)
    params = config.get("parameters", config)
    tps = float(params["ticks_per_second"]); dt = 1/tps
    horizons = sorted(set(float(h) for h in horizons))
    if not horizons or any(not math.isfinite(h) or h <= 0 or abs(h*tps-round(h*tps)) > 1e-8 for h in horizons):
        raise ValueError("Horizons must be positive whole simulation ticks")
    if not math.isfinite(origin_stride_s) or origin_stride_s <= 0:
        raise ValueError("Origin stride must be positive")
    horizon = max(horizons); ticks = round(horizon*tps)
    speed = params["velocity"]/100
    zones = zones_for(scenario)
    if not zones:
        raise ValueError(
            "Congestion evaluation requires junction metadata: add scenario.zones "
            "with conflict_zone bounds and centers, or a single scenario.conflict_zone. "
            "An unannotated map cannot establish queue prediction performance.")
    approaches = approaches_for(scenario, map_doc["layout"])
    if not approaches:
        raise ValueError(
            "Congestion evaluation found no directed approaches. For multiple zones, "
            "add scenario.network.edges linking zone IDs; for a single junction, "
            "provide traversable cardinal approaches into its conflict_zone. "
            "The current adapter supports cardinal geometry on a 1 m grid; "
            "other layouts need an approach adapter before evaluation.")
    scores = {h: {m: empty_scores() for m in METHODS} for h in horizons}
    crossing = {m: defaultdict(list) for m in FLOW_METHODS}
    counter_totals = {m: defaultdict(int) for m in FLOW_METHODS}
    runtimes, samples, origin_records = [], [], []
    last_origin = -math.inf
    used, with_later_plan, skipped = 0, 0, 0
    truth_positive_origins = 0
    warning_leads = {m: [] for m in FLOW_METHODS}
    isolated_delay_errors, isolated_delay_pairs = [], 0
    path = output/"forecasts.jsonl"
    with path.open("w") as stream:
        for event in sorted(snapshots, key=lambda e: e["clock"]):
            clock = event["clock"]
            if clock/tps-last_origin < origin_stride_s-1e-8:
                continue
            if any(clock+k not in frames for k in range(ticks+1)):
                skipped += 1; continue
            snapshot, pose = event["snapshot"], event["pose_frame"]
            if snapshot["clock"] != clock or pose["clock"] != clock:
                raise ValueError("Incoherent prediction snapshot timestamps")
            kwargs = dict(horizon_s=horizon, dt=dt, speed_m_s=speed,
                          station_dwell_s=20/tps, controllers=event["controllers"])
            started = time.perf_counter()
            joint = rollout(snapshot, pose, scenario, map_doc, **kwargs)
            runtimes.append(time.perf_counter()-started)
            no_safety = rollout(snapshot, pose, scenario, map_doc, safety_enabled=False, **kwargs)
            isolated_snapshot = deepcopy(snapshot)
            for robot in isolated_snapshot["robots"]:
                for action in robot["actions"]:
                    action["dependencies"] = [d for d in action.get("dependencies", []) if d[0] == robot["id"]]
            isolated = rollout(isolated_snapshot, pose, scenario, map_doc, robot_interactions=False, **kwargs)
            legacy, legacy_diagnostics = predict_snapshot(snapshot, pose, horizons,
                                                         speed_m_s=speed, station_dwell_s=20/tps)
            predicted = {"joint": joint["frames"], "plan_only": no_safety["frames"]}
            predicted["hold"] = [{"offset_s": f["offset_s"], "agents": to_agents(pose)} for f in joint["frames"]]
            # All forecasts above are complete before reading any future truth.
            truth = [{"offset_s": k*dt, "agents": to_agents(frames[clock+k])} for k in range(ticks+1)]
            true_queues = queue_samples(truth, approaches, scenario, speed_m_s=speed)
            pred_queues = {m: queue_samples(f, approaches, scenario, speed_m_s=speed) for m, f in predicted.items()}
            origin_row = {"origin_clock": clock, "origin_s": clock/tps,
                          "plan_version": snapshot["plan_version"],
                          "later_plan_within_horizon": any(clock < c <= clock+ticks for c in plans),
                          "runtime_s": runtimes[-1], "diagnostics": joint["diagnostics"], "horizons": {}}
            for h in horizons:
                index = round(h*tps)
                old = {"offset_s": h, "agents": [{"id": ident, "row": p[0], "col": p[1], "yaw": 0}
                                                   for ident, p in legacy[h].items()]}
                covered_ids = unexhausted_plan_ids(joint["diagnostics"],
                                                   [a["id"] for a in pose["agents"]], h)
                horizon_row = {"actual": truth[index]["agents"], "forecasts": {},
                               "actual_queues": true_queues[index],
                               "unexhausted_plan_robot_ids": sorted(covered_ids)}
                for method in METHODS:
                    frame = old if method == "legacy" else predicted[method][index]
                    queues = None if method == "legacy" else pred_queues[method][index]
                    score_endpoint(scores[h][method], frame, truth[index], truth[0], zones, queues, true_queues[index],
                                   unexhausted_ids=covered_ids)
                    horizon_row["forecasts"][method] = frame["agents"]
                origin_row["horizons"][str(h)] = horizon_row
            truth_cross = crossing_events(truth, zones)
            origin_row["crossings"] = {}
            for method, method_frames in predicted.items():
                predicted_cross = crossing_events(method_frames, zones)
                measurement = score_crossings(predicted_cross, truth_cross)
                origin_row["crossings"][method] = measurement
                for k, value in measurement.items():
                    if isinstance(value, list): crossing[method][k].extend(value)
                    else: counter_totals[method][k] += value
            # Same model-isolated reference for predicted and actual clearance.
            # This measures prediction error, NOT an independently calibrated delay truth.
            iso_cross = crossing_events(isolated["frames"], zones)
            joint_cross = crossing_events(joint["frames"], zones)
            delays = []
            for key, target in truth_cross.items():
                reference, forecast = iso_cross.get(key, {}), joint_cross.get(key, {})
                if all(v.get("exit_s") is not None for v in (reference, forecast, target)):
                    observed_delay = target["exit_s"]-reference["exit_s"]
                    forecast_delay = forecast["exit_s"]-reference["exit_s"]
                    delays.append({"id": key[0], "zone": key[1],
                                   "model_isolated_clearance_s": reference["exit_s"],
                                   "actual_clearance_s": target["exit_s"],
                                   "predicted_clearance_s": forecast["exit_s"],
                                   "observed_excess_vs_model_s": observed_delay,
                                   "predicted_excess_vs_model_s": forecast_delay})
                    isolated_delay_errors.append(abs(observed_delay-forecast_delay)); isolated_delay_pairs += 1
            origin_row["model_reference_delays"] = delays
            # Positive event is a future onset, not a queue already present at2s.
            initial_queue = true_queues[round(2*tps)] if len(true_queues) > round(2*tps) else None
            true_onsets = [q["offset_s"] for q in true_queues if q and q["multi_robot_queue"]]
            if true_onsets and initial_queue and not initial_queue["multi_robot_queue"]:
                truth_positive_origins += 1
                onset = min(true_onsets)
                for method in FLOW_METHODS:
                    if any(q and q["multi_robot_queue"] for q in pred_queues[method]):
                        warning_leads[method].append(onset)
            stream.write(json.dumps(origin_row, separators=(",", ":"), allow_nan=False)+"\n")
            used += 1; last_origin = clock/tps
            with_later_plan += origin_row["later_plan_within_horizon"]
            origin_records.append({"origin_s": clock/tps, "runtime_s": runtimes[-1],
                                   "max_actual_queue": max((q["max_queue"] for q in true_queues if q), default=0)})
            if demo and clock/tps in (10, 30, 50, 70, 90, 110, 130, 150):
                samples.append({"label": f"{scenario['name']} · seed {params['seed']}",
                                "origin_s": clock/tps, "plan_version": snapshot["plan_version"],
                                "trigger": scenario.get("safety", {}).get("robot_trigger", "body_in_outer"),
                                "diagnostics": joint["diagnostics"],
                                "frames": [{"offset_s": truth[k]["offset_s"], "actual": truth[k]["agents"],
                                            **{m: predicted[m][k]["agents"] for m in FLOW_METHODS}}
                                           for k in range(0, ticks+1, max(1, round(.5*tps)))]})
    if not used:
        raise ValueError("No complete forecast windows available")
    result = {"scenario": scenario["name"], "run_directory": str(run_dir),
              "seed": params["seed"], "trigger": scenario.get("safety", {}).get("robot_trigger"),
              "forecast_origins": used, "skipped_incomplete_windows": skipped,
              "origins_with_later_plan": with_later_plan,
              "runtime_s": {"mean": average(runtimes), "max": max(runtimes),
                            "p95": sorted(runtimes)[max(0, math.ceil(.95*len(runtimes))-1)]},
              "metrics": {str(h): {m: finalize_scores(scores[h][m]) for m in METHODS} for h in horizons},
              "clearance": {m: {**counter_totals[m],
                                   "matched_clearance_mae_s": average(crossing[m]["clearance_errors_s"]),
                                   "matched_arrival_mae_s": average(crossing[m]["arrival_errors_s"])} for m in FLOW_METHODS},
              "model_reference_delay": {"matched_count": isolated_delay_pairs,
                                         "mae_s": average(isolated_delay_errors),
                                         "note": "Both excess times subtract the same approximate model-isolated reference. Its error is NOT measured; negative excess values are retained. MAE equals matched clearance-time MAE on this subset."},
              "queue_onset": {"positive_origin_windows": truth_positive_origins,
                               "methods": {m: {"warned_positive_windows": len(v), "mean_potential_lead_s": average(v)}
                                           for m, v in warning_leads.items()},
                               "note": "Overlapping origin windows are correlated. Potential lead counts any predicted queue within the same window; not calibrated probabilities or matched per-approach onset accuracy."},
              "max_actual_queue_proxy_robots": max(r["max_actual_queue"] for r in origin_records),
              "parameters": {"horizons_s": horizons, "dt_s": dt, "origin_stride_s": origin_stride_s,
                             "queue_history_s": 2, "queue_speed_fraction": .3,
                             "queue_min_robots_event": 2,
                             "plan_coverage_definition": "Same joint-rollout cohort for all methods: known accepted actions are not exhausted strictly before the endpoint. Excludes both finished robots and robots awaiting unseen future task routes. Uses no future truth; it does not establish that routes remain unchanged."},
              "origin_diagnostics": origin_records}
    write_json(output/"metrics.json", result)
    return result, samples, {"layout": map_doc["layout"], "zones": zones, "cell_size_m": 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizons", type=float, nargs="+", default=[5, 10, 20])
    p.add_argument("--origin-stride", type=float, default=1)
    p.add_argument("--jobs", type=int, default=1, help="Independent recordings to score concurrently")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    results, samples, layout = [], [], None
    completed = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        pending = {}
        for i, directory in enumerate(args.run_dirs):
            out = args.output/f"run-{i+1:02d}"; out.mkdir()
            future = executor.submit(evaluate_run, directory, out, horizons=args.horizons,
                                     origin_stride_s=args.origin_stride, demo=True)
            pending[future] = (i, directory)
        for future in as_completed(pending):
            i, directory = pending[future]
            completed[i] = future.result()
            result = completed[i][0]
            print(json.dumps({"run": str(directory), "origins": result["forecast_origins"],
                              "joint_10s": result["metrics"].get("10.0", {}).get("joint"),
                              "runtime_s": result["runtime_s"]}), flush=True)
    for i in sorted(completed):
        result, demo, current_layout = completed[i]
        results.append(result)
        if layout is None: layout = current_layout
        if current_layout == layout: samples.extend(demo)
    metrics = {"runs": results, "methods": {
        "joint": "Synchronous rollout with native snapshot and safety responses",
        "plan_only": "Same rollout and controller state with all safety responses disabled (ablation)",
        "legacy": "Previous fixed-duration accepted-action predictor",
        "hold": "All robots hold their origin poses"},
        "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in Path(__file__).parent.glob("*.py")},
        "notes": ["Forecasts use only coherent origin state. Future poses and plans enter scoring only.",
                  "Routes/dependencies are frozen within each forecast. Unseen future task assignments and replanning are not predicted.",
                  "Queue and spillback are operational geometry/speed proxies, not proven obstruction. Slowly moving groups are included.",
                  "Clearance MAE is conditional on matched exits; inspect missed and false exits alongside it.",
                  "No fitted parameters, artificial corruption, or forecast-driven routing were used in this comparison.",
                  "All robot samples/origin windows from one run are correlated; counts are not independent trials.",
                  "A deterministic rollout does not produce calibrated event probabilities or certify collision safety."]}
    write_json(args.output/"metrics.json", metrics)
    write_json(args.output/"demo_data.json", {"title": "Warehouse congestion forecasts",
               "scenario": layout, "metrics": {"summary": "Frozen forecasts from recorded execution snapshots · 0.1-second joint rollout · compare with actual execution"},
               "samples": samples})
    if samples:
        from experiments.prediction.joint_viewer import write_viewer
        write_viewer(args.output)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
