#!/usr/bin/env python3
"""Score clean and deliberately imperfect accepted-plan forecasts on one recording."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.prediction.predict import _distance, _read_recording, predict_snapshot


PROFILES = {
    "clean": {"duration_bias": 0.0, "observation_delay_s": 0.0, "delivery_dropout": 0.0},
    "mild": {"duration_bias": 0.15, "observation_delay_s": 0.5, "delivery_dropout": 0.05},
    "severe": {"duration_bias": 0.4, "observation_delay_s": 2.0, "delivery_dropout": 0.2},
}


def _uniform(seed, *keys):
    """Stable draws independent of iteration order, horizons, and Python RNG state."""
    data = json.dumps([seed, *keys], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") / 2**64


def _validate_profile(profile):
    if set(profile) != {"duration_bias", "observation_delay_s", "delivery_dropout"}:
        raise ValueError("Profiles require duration_bias, observation_delay_s, delivery_dropout")
    if any(not math.isfinite(value) for value in profile.values()):
        raise ValueError("Profile values must be finite")
    if not 0 <= profile["duration_bias"] < 1:
        raise ValueError("duration_bias must be in [0, 1)")
    if profile["observation_delay_s"] < 0 or not 0 <= profile["delivery_dropout"] <= 1:
        raise ValueError("Observation delay must be nonnegative and dropout in [0, 1]")


def forecast_at_origin(snapshots, frames, origin_clock, horizons, *, profile,
                       seed=42, ticks_per_second=10, speed_m_s=1,
                       turn_rate_deg_s=90, station_dwell_s=2):
    """Return a causal forecast and delivery mask, or None before inputs exist.

    Select the most recent plan at/before the delayed observation cutoff and a
    pose at/before that plan. Advance that old plan by its age plus the horizon.
    Future poses and plans are never used. Dropout hides output forecasts only;
    all robots remain in the internal scheduling graph.
    """
    _validate_profile(profile)
    if not math.isfinite(ticks_per_second) or ticks_per_second <= 0:
        raise ValueError("ticks_per_second must be positive finite")
    if not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("Horizons must be positive finite seconds")
    cutoff = origin_clock - profile["observation_delay_s"] * ticks_per_second
    eligible = [state for state in snapshots if state["clock"] <= cutoff]
    if not eligible:
        return None
    state = max(eligible, key=lambda item: (item["clock"], item["plan_version"]))
    clocks = sorted(clock for clock in frames if clock <= state["clock"])
    if not clocks:
        return None
    pose = frames[clocks[-1]]
    plan_age_s = (origin_clock - state["clock"]) / ticks_per_second
    scales = {robot["id"]: 1 + profile["duration_bias"] *
              (2 * _uniform(seed, "duration", robot["id"]) - 1)
              for robot in state["robots"]}
    generated, diagnostics = predict_snapshot(
        state, pose, [plan_age_s + h for h in horizons], speed_m_s=speed_m_s,
        turn_rate_deg_s=turn_rate_deg_s, station_dwell_s=station_dwell_s,
        duration_scales=scales)
    predictions = {h: generated[plan_age_s + h] for h in horizons}
    delivered = {robot_id: _uniform(seed, "delivery", origin_clock, robot_id) >=
                 profile["delivery_dropout"] for robot_id in scales}
    return {
        "predictions": predictions, "delivered": delivered,
        "hold": {p["id"]: (-p["x"], -p["y"]) for p in pose["agents"]},
        "observation_cutoff_clock": cutoff, "snapshot_clock": state["clock"],
        "pose_clock": pose["clock"], "plan_version": state["plan_version"],
        "planner_source": state.get("planner_source", "unknown"),
        "plan_age_s": plan_age_s,
        "pose_age_s": (origin_clock - pose["clock"]) / ticks_per_second,
        "duration_scales": scales, "diagnostics": diagnostics,
    }


def _zones(scenario):
    raw = scenario.get("zones", [])
    if isinstance(raw, dict):
        raw = [dict(value, id=name) for name, value in raw.items()]
    result = {}
    for index, zone in enumerate(raw):
        bounds = zone.get("bounds", zone)
        if "rows" not in bounds or "cols" not in bounds:
            raise ValueError("Zones must define rectangular rows/cols bounds")
        result[str(zone.get("id", zone.get("name", index)))] = bounds
    if not result and "conflict_zone" in scenario:
        result["junction"] = scenario["conflict_zone"]
    return result


def _count(positions, zone):
    # Half-open cell boundaries avoid double-counting adjacent rectangular zones.
    return sum(zone["rows"][0] - .5 <= row < zone["rows"][1] + .5 and
               zone["cols"][0] - .5 <= col < zone["cols"][1] + .5
               for row, col in positions.values())


def _mean(values):
    return sum(values) / len(values) if values else None


def _new_totals():
    return {"origins": 0, "possible": 0, "delivered": 0,
            "error": [], "hold_error": [], "moving_error": [], "moving_hold_error": [],
            "clean_error": [], "clean_moving_error": [],
            "zones": defaultdict(lambda: {"error": [], "hold_error": [], "interval_hit": [],
                                           "interval_width": []})}


def analyze(run_dir, *, output_dir=None, profiles=("clean", "mild", "severe"), seed=42,
            horizons=(1, 2, 5, 10), profile_settings=None, turn_rate_deg_s=90,
            station_dwell_s=None, moving_threshold_m=.25):
    """Write corrupted_forecasts.jsonl and corruption_metrics.json beside one run."""
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else run_dir
    settings = {**PROFILES, **(profile_settings or {})}
    profiles = list(dict.fromkeys(profiles))
    if not profiles or any(name not in settings for name in profiles):
        raise ValueError("Choose at least one defined corruption profile")
    for name in profiles:
        _validate_profile(settings[name])
    horizons = sorted(set(float(h) for h in horizons))
    if not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError("Horizons must be positive finite seconds")
    if not math.isfinite(moving_threshold_m) or moving_threshold_m < 0:
        raise ValueError("Moving threshold must be finite and nonnegative")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path = run_dir.parent / "config.json"
    config = json.loads(config_path.read_text())
    scenario = json.loads((config_path.parent / config["scenario"]).read_text())
    parameters = config.get("parameters", config)
    tps = float(parameters["ticks_per_second"])
    if not math.isfinite(tps) or tps <= 0:
        raise ValueError("ticks_per_second must be positive finite")
    horizon_ticks = {h: round(h * tps) for h in horizons}
    if any(not math.isclose(h * tps, horizon_ticks[h], abs_tol=1e-7) for h in horizons):
        raise ValueError("Each horizon must be an integer number of simulation ticks")
    model = {"ticks_per_second": tps, "speed_m_s": float(parameters["velocity"]) / 100,
             "turn_rate_deg_s": turn_rate_deg_s,
             "station_dwell_s": 20 / tps if station_dwell_s is None else station_dwell_s}
    frames, snapshots, sources = _read_recording(run_dir)
    if not frames or not snapshots:
        raise ValueError("Recording must contain tick poses and plan_snapshot events")
    clocks = sorted(frames)
    zones = _zones(scenario)
    totals = {name: {h: _new_totals() for h in horizons} for name in profiles}
    metadata = {name: {"generated_origins": 0, "skipped_without_past_inputs": 0,
                       "targets_without_truth": 0, "plan_age_s": [], "pose_age_s": [],
                       "compute_ms": [], "duration_scales": {}} for name in profiles}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "corrupted_forecasts.jsonl").open("w") as stream:
        for origin_clock in sorted({state["clock"] for state in snapshots}):
            # Clean reference is used only in evaluation after each independent
            # profile forecast is generated, never as an input to corruption.
            clean = forecast_at_origin(snapshots, frames, origin_clock, horizons,
                                       profile=PROFILES["clean"], seed=seed, **model)
            origin_results = {}
            for name in profiles:
                started = time.perf_counter()
                result = forecast_at_origin(snapshots, frames, origin_clock, horizons,
                                            profile=settings[name], seed=seed, **model)
                elapsed_ms = (time.perf_counter() - started) * 1000
                meta = metadata[name]
                if result is None:
                    meta["skipped_without_past_inputs"] += 1
                    continue
                meta["generated_origins"] += 1
                for key in ("plan_age_s", "pose_age_s"):
                    meta[key].append(result[key])
                meta["compute_ms"].append(elapsed_ms)
                meta["duration_scales"].update(result["duration_scales"])
                origin_results[name] = result
                for h in horizons:
                    delivered = {i: p for i, p in result["predictions"][h].items()
                                 if result["delivered"][i]}
                    missing = len(result["delivered"]) - len(delivered)
                    stream.write(json.dumps({
                        "profile": name, "origin_clock": origin_clock,
                        "horizon_s": h, "target_clock": origin_clock + horizon_ticks[h],
                        **{key: result[key] for key in (
                            "observation_cutoff_clock", "snapshot_clock", "pose_clock",
                            "plan_version", "planner_source", "plan_age_s", "pose_age_s",
                            "diagnostics")},
                        "robots": [{"id": i, "available": result["delivered"][i],
                                    "row": p[0] if result["delivered"][i] else None,
                                    "col": p[1] if result["delivered"][i] else None}
                                   for i, p in sorted(result["predictions"][h].items())],
                        "zones": {zone_id: {"known_count": _count(delivered, zone),
                                            "missing_robot_count": missing,
                                            "count_range": [_count(delivered, zone),
                                                            _count(delivered, zone) + missing]}
                                  for zone_id, zone in zones.items()},
                    }, separators=(",", ":")) + "\n")

            # Truth is consulted only after the origin's forecasts are written.
            origin_index = bisect_right(clocks, origin_clock) - 1
            true_origin = {p["id"]: (-p["x"], -p["y"])
                           for p in frames[clocks[origin_index]]["agents"]}
            for name, result in origin_results.items():
                for h in horizons:
                    target_clock = origin_clock + horizon_ticks[h]
                    if target_clock not in frames:
                        metadata[name]["targets_without_truth"] += 1
                        continue
                    truth = {p["id"]: (-p["x"], -p["y"])
                             for p in frames[target_clock]["agents"]}
                    positions = result["predictions"][h]
                    if set(positions) != set(truth) or set(truth) != set(true_origin):
                        raise ValueError(f"Robot identities differ at clock {target_clock}")
                    total = totals[name][h]
                    total["origins"] += 1
                    total["possible"] += len(positions)
                    delivered = {i: p for i, p in positions.items() if result["delivered"][i]}
                    total["delivered"] += len(delivered)
                    for i, position in delivered.items():
                        error = _distance(position, truth[i])
                        hold_error = _distance(result["hold"][i], truth[i])
                        clean_error = _distance(clean["predictions"][h][i], truth[i])
                        total["error"].append(error)
                        total["hold_error"].append(hold_error)
                        total["clean_error"].append(clean_error)
                        if _distance(true_origin[i], truth[i]) > moving_threshold_m:
                            total["moving_error"].append(error)
                            total["moving_hold_error"].append(hold_error)
                            total["clean_moving_error"].append(clean_error)
                    missing = len(positions) - len(delivered)
                    for zone_id, zone in zones.items():
                        z = total["zones"][zone_id]
                        known_count, actual_count = _count(delivered, zone), _count(truth, zone)
                        z["interval_hit"].append(known_count <= actual_count <= known_count + missing)
                        z["interval_width"].append(missing)
                        if not missing:
                            z["error"].append(abs(known_count - actual_count))
                            z["hold_error"].append(abs(_count(result["hold"], zone) - actual_count))
    if not any(total["possible"] for per_profile in totals.values() for total in per_profile.values()):
        raise ValueError("No forecast targets have recorded truth; use shorter horizons or a longer run")

    reports = {}
    for name in profiles:
        meta = metadata[name]
        rows = []
        for h in horizons:
            total = totals[name][h]
            rows.append({
                "horizon_s": h, "scored_origins": total["origins"],
                "eligible_robot_endpoints": total["possible"],
                "delivered_robot_endpoints": total["delivered"],
                "coverage": total["delivered"] / total["possible"] if total["possible"] else None,
                "moving_robot_endpoints": len(total["moving_error"]),
                "endpoint_mean_error_m": _mean(total["error"]),
                "moving_endpoint_mean_error_m": _mean(total["moving_error"]),
                "hold_endpoint_mean_error_m": _mean(total["hold_error"]),
                "hold_moving_endpoint_mean_error_m": _mean(total["moving_hold_error"]),
                "matched_clean_endpoint_mean_error_m": _mean(total["clean_error"]),
                "matched_clean_moving_endpoint_mean_error_m": _mean(total["clean_moving_error"]),
                "zones": {zone_id: {
                    "complete_coverage_origins": len(z["error"]),
                    "occupancy_mae_robots": _mean(z["error"]),
                    "hold_occupancy_mae_robots": _mean(z["hold_error"]),
                    "missing_data_range_contains_truth_fraction": _mean(z["interval_hit"]),
                    "mean_missing_data_range_width_robots": _mean(z["interval_width"]),
                } for zone_id, z in total["zones"].items()},
            })
        reports[name] = {
            "parameters": settings[name],
            **{key: meta[key] for key in ("generated_origins", "skipped_without_past_inputs",
                                         "targets_without_truth", "duration_scales")},
            "mean_plan_age_s": _mean(meta["plan_age_s"]),
            "max_plan_age_s": max(meta["plan_age_s"], default=None),
            "mean_pose_age_s": _mean(meta["pose_age_s"]),
            "max_pose_age_s": max(meta["pose_age_s"], default=None),
            "mean_forecast_compute_ms": _mean(meta["compute_ms"]), "metrics": rows,
        }
    metrics = {
        "scenario": scenario["name"], "run_directory": str(run_dir), "seed": seed,
        "forecast_model": "accepted-plan action rollout with controlled corruption",
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("predict.py", "corrupt.py")},
        "model_parameters": model, "moving_threshold_m": moving_threshold_m,
        "log_sources": sources, "profiles": reports,
        "notes": [
            "All profiles score the same recorded physical run; corruption changes forecasts, not execution.",
            "Duration bias is persistent per robot and scales move, turn, and station actions; dependencies inherit it.",
            "Delayed inputs use the latest eligible plan and a pose at/before that plan, then roll forward age plus horizon.",
            "Dropout hides forecast delivery only; it does not simulate missing internal observations or RF propagation.",
            "Hold and matched-clean errors use exactly the same delivered robot/origin/horizon cohort as each profile.",
            "Moving cohorts use actual displacement from issue time, not stale observation time; truth is evaluation-only.",
            "Occupancy MAE is scored only with complete delivery; count ranges express missing forecasts, not calibrated uncertainty.",
            "Mean compute time measures this offline Python call on this host, not end-to-end communication latency.",
            "Presets are experimental error controls, not measured or claimed Amazon DeepFleet error distributions.",
        ],
    }
    (output_dir / "corruption_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES))
    parser.add_argument("--profile-config", type=Path,
                        help="JSON mapping profile names to the three corruption parameters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", nargs="+", type=float, default=[1, 2, 5, 10])
    parser.add_argument("--turn-rate-deg-s", type=float, default=90)
    parser.add_argument("--station-dwell-s", type=float)
    args = parser.parse_args()
    try:
        settings = json.loads(args.profile_config.read_text()) if args.profile_config else None
        result = analyze(args.run_dir, output_dir=args.output_dir, profiles=args.profiles,
                         profile_settings=settings, seed=args.seed, horizons=args.horizons,
                         turn_rate_deg_s=args.turn_rate_deg_s, station_dwell_s=args.station_dwell_s)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
