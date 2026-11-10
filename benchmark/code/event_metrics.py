#!/usr/bin/env python3
"""Descriptive hotspot detection on the existing frozen forecast test set.

A zone is hot if its future mean slow occupancy reaches one robot. Currently
clear means fewer than one slow robot now; it is not an observed onset time,
and it does not establish whether a robot still has a useful diversion option.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPORT = Path(__file__).resolve().parents[1]
WINDOWS = ((0, 5), (5, 10), (10, 20))
THRESHOLD = 1.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zone_masks(scenario, road):
    """Use the executable scenario's inclusive cell-center zone bounds."""
    rr, cc = np.indices(road.shape)
    zones, masks = [], []
    for zone in scenario["zones"]:
        lower_r, upper_r = zone["bounds"]["rows"]
        lower_c, upper_c = zone["bounds"]["cols"]
        mask = road & (rr >= lower_r) & (rr <= upper_r) & (cc >= lower_c) & (cc <= upper_c)
        zones.append({"id": zone["id"], "kind": zone["kind"], "bounds": zone["bounds"],
                      "road_cells": int(mask.sum())})
        masks.append(mask)
    masks = np.stack(masks)
    if not np.array_equal(masks.sum(axis=0), road.astype(int)):
        raise ValueError("Zones must partition every road cell exactly once")
    return zones, masks


def sum_zones(field, masks):
    """Sum cell occupancy; any leading sample/window axes are preserved."""
    return np.stack([np.asarray(field, dtype=np.float64)[..., mask].sum(axis=-1) for mask in masks], axis=-1)


def confusion(predicted_hot, actual_hot, selected=None):
    if predicted_hot.shape != actual_hot.shape:
        raise ValueError("Prediction and truth event shapes differ")
    selected = np.ones(actual_hot.shape, dtype=bool) if selected is None else np.broadcast_to(selected, actual_hot.shape)
    p, y = predicted_hot[selected], actual_hot[selected]
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    ratio = lambda a, b: a / b if b else None
    return {"zone_origin_windows": len(y), "actual_hotspots": tp + fn, "predicted_hotspots": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "false_positive_rate": ratio(fp, fp + tn),
            "false_alarms_per_1000_zone_origin_windows": ratio(1000 * fp, len(y)),
            "actual_prevalence": ratio(tp + fn, len(y))}


def score_windows(predicted, actual, current_clear, selection=None):
    if selection is not None:
        predicted, actual, current_clear = predicted[selection], actual[selection], current_clear[selection]
    return [{"window_s": [start, stop], "threshold_slow_robot_seconds": THRESHOLD * (stop - start),
             "all_zones": confusion(predicted[:, k] >= THRESHOLD, actual[:, k] >= THRESHOLD),
             "currently_clear_zones": confusion(predicted[:, k] >= THRESHOLD, actual[:, k] >= THRESHOLD, current_clear)}
            for k, (start, stop) in enumerate(WINDOWS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=REPORT / "results/test_predictions.npz")
    parser.add_argument("--data", type=Path, default=REPORT / "data/dataset.npz")
    parser.add_argument("--index", type=Path, default=REPORT / "data/dataset_index.json")
    parser.add_argument("--output", type=Path, default=REPORT / "results/event_evaluation.json")
    args = parser.parse_args()
    saved = np.load(args.predictions)
    dataset = np.load(args.data)
    index = json.loads(args.index.read_text())["samples"]
    indices = saved["indices"]
    expected = np.asarray([i for i, row in enumerate(index) if row["split"] == "test"])
    if not np.array_equal(indices, expected):
        raise ValueError("Saved forecast indices differ from the frozen test split")
    if not np.array_equal(saved["truth"], dataset["target"][indices]):
        raise ValueError("Saved forecast truth differs from frozen dataset targets")
    if not np.array_equal(saved["road"], dataset["road"]):
        raise ValueError("Forecast and feature archives have different road masks")
    metadata = [index[i] for i in indices]
    scenario_path = Path(metadata[0]["run_directory"]) / "scenario.json"
    scenario = json.loads(scenario_path.read_text())
    zones, masks = zone_masks(scenario, saved["road"])
    for directory in {row["run_directory"] for row in metadata}:
        other = json.loads((Path(directory) / "scenario.json").read_text())
        if not np.array_equal(zone_masks(other, saved["road"])[1], masks):
            raise ValueError("Held-out zone geometry differs between recordings")
    current = sum_zones(dataset["history"][indices, -1, 1], masks)
    current_clear = current < THRESHOLD
    actual = sum_zones(saved["truth"][:, :, 1], masks)
    methods = [key for key in saved.files if key not in ("truth", "indices", "road", "roi")]
    result = {"definition": "A hotspot is at least one mean slow robot across a whole zone over a forecast window, using the same slow-occupancy target as raster scoring.",
              "threshold_mean_slow_robots": THRESHOLD, "windows_s": WINDOWS,
              "currently_clear_definition": "Current instantaneous slow occupancy summed over zone road cells is strictly less than one robot; this cohort is selected using only origin information.",
              "interpretation": "Descriptive frozen-test diagnostic, not an acceptance gate. Current-clear future hotspots are not measured onset lead times or geometric diversion deadlines. Adjacent origins/windows are correlated; this sample has only held-out workload seeds with the recorded safety conditions.",
              "undefined_metrics": "Precision is null when no hotspot is predicted; recall is null when no actual hotspot exists.",
              "storage_precision": "Zone sums use float64 accumulation of the saved raster values; original target and baseline arrays are float16. The fixed threshold is applied without test-based tuning.",
              "test_origins": len(indices), "independent_workload_seeds": sorted({row["workload_seed"] for row in metadata}),
              "zone_count": len(zones), "zones": zones,
              "currently_clear_zone_origins": int(current_clear.sum()),
              "total_zone_origins": int(current_clear.size),
              "methods": {}, "per_run": {},
              "source_hashes": {"test_predictions": digest(args.predictions), "dataset": digest(args.data),
                                "dataset_index": digest(args.index), "scenario": digest(scenario_path), "code": digest(__file__)}}
    for method in methods:
        prediction = saved[method]
        if prediction.shape != saved["truth"].shape or not np.isfinite(prediction).all() or np.any(prediction < 0):
            raise ValueError("Invalid forecast field: " + method)
        predicted = sum_zones(prediction[:, :, 1], masks)
        result["methods"][method] = score_windows(predicted, actual, current_clear)
        result["per_run"][method] = {
            run: score_windows(predicted, actual, current_clear, np.asarray([row["run_id"] == run for row in metadata]))
            for run in sorted({row["run_id"] for row in metadata})}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "test_origins": len(indices), "methods": methods,
                      "actual_hotspots_by_window": (actual >= THRESHOLD).sum(axis=(0, 2)).tolist(),
                      "current_clear_future_hotspots_by_window": ((actual >= THRESHOLD) & current_clear[:, None, :]).sum(axis=(0, 2)).tolist()}, indent=2))


if __name__ == "__main__":
    main()
