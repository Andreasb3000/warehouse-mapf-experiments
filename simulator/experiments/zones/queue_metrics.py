"""Operational approach queue and spillback-warning proxies, from actual poses.

The caller supplies stationary, active robot IDs after excluding completed tasks,
service bays and turning. Geometry uses the runnable fixture's 1 m cell pitch.
These measures do not identify the cause of waiting or prove upstream blockage.
"""

import math


THRESHOLDS = {"stationary_window_s": 2.0, "stationary_displacement_m": .1,
              "stationary_heading_change_rad": .1, "service_bay_exclusion_radius_m": .2,
              "heading_tolerance_deg": 45.0,
              "leading_center_to_entry_max_m": 2.0,
              "consecutive_longitudinal_center_gap_max_m": 2.0,
              "spillback_tail_to_upstream_exit_max_m": .5}


def build_approaches(scenario):
    """One rectangle per directed cardinal link, excluding junction interiors."""
    zones = {zone["id"]: zone for zone in scenario.get("zones", [])}
    approaches = []
    for first, second in scenario.get("network", {}).get("edges", []):
        for source_id, target_id in ((first, second), (second, first)):
            source, target = zones[source_id], zones[target_id]
            start, end = source["center"], target["center"]
            dr, dc = end[0] - start[0], end[1] - start[1]
            if bool(dr) == bool(dc):
                raise ValueError("Queue approach extraction requires nonzero cardinal links")
            axis, cross = (0, 1) if dr else (1, 0)
            direction = 1 if end[axis] > start[axis] else -1
            along_key, cross_key = ("rows", "cols") if axis == 0 else ("cols", "rows")
            # Conflict bounds describe occupied cell centers. The physical/logical
            # boundary is half a cell beyond the outermost cell center.
            source_edge = source["conflict_zone"][along_key][1 if direction > 0 else 0] + direction * .5
            target_edge = target["conflict_zone"][along_key][0 if direction > 0 else 1] - direction * .5
            length = direction * (target_edge - source_edge)
            low = max(source["conflict_zone"][cross_key][0], target["conflict_zone"][cross_key][0]) - .5
            high = min(source["conflict_zone"][cross_key][1], target["conflict_zone"][cross_key][1]) + .5
            if length <= 0 or high <= low:
                raise ValueError("Queue approach has no positive length or width")
            exit_point = list(start); exit_point[axis] = source_edge
            entry_point = list(end); entry_point[axis] = target_edge
            unit = [0, 0]; unit[axis] = direction
            approaches.append({"id": f"{source_id}->{target_id}", "source_zone": source_id,
                               "target_zone": target_id, "axis": axis, "cross_axis": cross,
                               "direction": direction, "unit_row_col": unit,
                               "upstream_exit_row_col": exit_point, "downstream_entry_row_col": entry_point,
                               "transverse_min_m": low, "transverse_max_m": high,
                               "length_m": length, "width_m": high-low})
    return approaches


def measure_approach(approach, poses, eligible_ids, radii, thresholds=None):
    """Measure the first connected longitudinal group waiting near the entry."""
    thresholds = {**THRESHOLDS, **(thresholds or {})}
    axis, cross = approach["axis"], approach["cross_axis"]
    unit_row, unit_col = approach["unit_row_col"]
    cosine = math.cos(math.radians(thresholds["heading_tolerance_deg"]))
    candidates = []
    for pose in poses:
        robot_id = pose["id"]
        if robot_id not in eligible_ids:
            continue
        row_col = [-pose["x"], -pose["y"]]
        if not approach["transverse_min_m"] <= row_col[cross] < approach["transverse_max_m"]:
            continue
        progress = approach["direction"] * (row_col[axis] - approach["upstream_exit_row_col"][axis])
        if not 0 <= progress < approach["length_m"]:
            continue
        # World x=-row and y=-col. Heading zero points toward decreasing row.
        alignment = -math.cos(pose["rz"]) * unit_row - math.sin(pose["rz"]) * unit_col
        if alignment + 1e-12 < cosine:
            continue
        candidates.append((approach["length_m"] - progress, robot_id))
    candidates.sort()
    group = []
    if candidates and candidates[0][0] <= thresholds["leading_center_to_entry_max_m"]:
        group.append(candidates[0])
        for candidate in candidates[1:]:
            if candidate[0] - group[-1][0] > thresholds["consecutive_longitudinal_center_gap_max_m"]:
                break
            group.append(candidate)
    # Include each measured robot body's rear edge. This is distance upstream
    # from junction entry, not a sum of diameters or a causal blocking length.
    tail_extent = max((distance + radii.get(robot_id, .085) for distance, robot_id in group), default=0.0)
    body_span = (tail_extent - min(distance-radii.get(robot_id, .085) for distance, robot_id in group)) if group else 0.0
    upstream_clearance = approach["length_m"] - tail_extent if group else None
    warning = bool(group) and upstream_clearance <= thresholds["spillback_tail_to_upstream_exit_max_m"]
    return {"approach_id": approach["id"], "queued_robot_ids": [robot_id for _, robot_id in group],
            "queued_robots": len(group), "queue_tail_extent_m": tail_extent,
            "queue_body_span_m": body_span, "tail_to_upstream_exit_m": upstream_clearance,
            "spillback_warning": warning}


def summarize_queues(samples, approaches, dt):
    by_clock = {}
    for sample in samples:
        current = by_clock.setdefault(sample["clock"], {"queued": 0, "warnings": 0})
        current["queued"] += sample["queued_robots"]
        current["warnings"] += sample["spillback_warning"]
    per_approach = {}
    for approach in approaches:
        selected = [sample for sample in samples if sample["approach_id"] == approach["id"]]
        per_approach[approach["id"]] = {
            "queued_robot_seconds": dt*sum(sample["queued_robots"] for sample in selected),
            "max_queue_robots": max((sample["queued_robots"] for sample in selected), default=0),
            "max_queue_tail_extent_m": max((sample["queue_tail_extent_m"] for sample in selected), default=0.0),
            "spillback_warning_seconds": dt*sum(sample["spillback_warning"] for sample in selected)}
    return {"available": bool(approaches), "thresholds": THRESHOLDS, "approaches": per_approach,
            "queued_robot_seconds": dt*sum(sample["queued_robots"] for sample in samples),
            "max_queue_robots": max((sample["queued_robots"] for sample in samples), default=0),
            "max_simultaneously_queued_robots": max((value["queued"] for value in by_clock.values()), default=0),
            "max_queue_tail_extent_m": max((sample["queue_tail_extent_m"] for sample in samples), default=0.0),
            "spillback_warning_approach_seconds": dt*sum(sample["spillback_warning"] for sample in samples),
            "max_simultaneous_spillback_warning_approaches": max((value["warnings"] for value in by_clock.values()), default=0),
            "definition": "Connected group of eligible stationary-active robot centers facing a junction, led within 2 m of its entry, with consecutive longitudinal gaps at most 2 m. The group aggregates the approach width; no separate lanes are inferred.",
            "tail_definition": "Distance from downstream junction entry to the rearmost body edge of the connected group, including any empty distance between entry and leader.",
            "spillback_definition": "Warning when this queue tail reaches within 0.5 m of the upstream conflict-zone exit. This is a geometric warning proxy, not proof of upstream traffic obstruction.",
            "geometry": "1 m grid pitch; conflict boundaries are half a cell beyond the listed center bounds. Approach longitudinal bounds include upstream exit and exclude downstream entry; transverse bounds are half-open."}
