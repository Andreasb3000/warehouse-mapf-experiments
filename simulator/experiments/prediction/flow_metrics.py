"""Shared operational measurements for predicted and recorded fleet motion.

Queues include slow-moving groups. These geometry/speed proxies do not identify
the cause of a delay or certify that an upstream junction is obstructed.
"""
import math

from experiments.zones.queue_metrics import build_approaches, measure_approach


def zones_for(scenario):
    zones = scenario.get("zones", [])
    if zones:
        return zones
    if "conflict_zone" in scenario:
        bounds = scenario["conflict_zone"]
        return [{"id": "junction", "conflict_zone": bounds,
                 "center": [sum(bounds[k])/2 for k in ("rows", "cols")]}]
    return []


def approaches_for(scenario, layout):
    """Use network links, or derive cardinal approaches to a single junction."""
    if scenario.get("network", {}).get("edges"):
        return build_approaches(scenario)
    zones = zones_for(scenario)
    if len(zones) != 1:
        return []
    zone = zones[0]
    bounds, center = zone["conflict_zone"], zone["center"]
    result = []
    for axis in (0, 1):
        cross = 1-axis
        along, across = (("rows", "cols") if axis == 0 else ("cols", "rows"))
        for direction in (-1, 1):
            edge = bounds[along][0 if direction > 0 else 1]-direction*.5
            point = list(center)
            point[axis] = int(edge-direction*.5)
            cells = []
            while (0 <= point[0] < len(layout) and 0 <= point[1] < len(layout[0])
                   and layout[int(point[0])][int(point[1])] not in "@T"):
                cells.append(point[axis])
                point[axis] -= direction
            if not cells:
                continue
            start, end = list(center), list(center)
            start[axis] = cells[-1]-direction*.5
            end[axis] = edge
            unit = [0, 0]; unit[axis] = direction
            result.append({"id": f"{axis}:{direction}->junction", "axis": axis,
                           "cross_axis": cross, "direction": direction,
                           "unit_row_col": unit, "upstream_exit_row_col": start,
                           "downstream_entry_row_col": end,
                           "length_m": abs(edge-start[axis]),
                           "transverse_min_m": bounds[across][0]-.5,
                           "transverse_max_m": bounds[across][1]+.5,
                           "width_m": bounds[across][1]-bounds[across][0]+1})
    return result


def inside(agent, zone):
    b = zone["conflict_zone"]
    return (b["rows"][0]-.5 <= agent["row"] < b["rows"][1]+.5
            and b["cols"][0]-.5 <= agent["col"] < b["cols"][1]+.5)


def occupancy(frame, zones):
    return {z["id"]: sum(inside(a, z) for a in frame["agents"]) for z in zones}


def queue_samples(frames, approaches, scenario, *, speed_m_s=1.0, window_s=2.0):
    """Two-second low-speed, approximately straight groups near junction entry.

    Use exactly the same detector for predictions and truth. Exclude all service
    locations (including final parking) rather than using future task labels.
    """
    goals = [g for a in scenario["agents"] for g in a.get("goals", [])]
    radii = {a["id"]: {**scenario.get("safety", {}), **a.get("safety", {})}.get(
        "robot_radius_m", .085) for a in scenario["agents"]}
    lookup = [{a["id"]: a for a in f["agents"]} for f in frames]
    result = []
    start = 0
    for end, frame in enumerate(frames):
        while start < end and frames[start+1]["offset_s"] <= frame["offset_s"]-window_s+1e-8:
            start += 1
        if frame["offset_s"]-frames[start]["offset_s"] < window_s-1e-8:
            result.append(None); continue
        eligible = set()
        duration = frame["offset_s"]-frames[start]["offset_s"]
        for agent in frame["agents"]:
            ident = agent["id"]
            if any(math.hypot(agent["row"]-g[0], agent["col"]-g[1]) <= .25 for g in goals):
                continue
            history = [lookup[k][ident] for k in range(start, end+1)]
            distance = sum(math.hypot(b["row"]-a["row"], b["col"]-a["col"])
                           for a, b in zip(history, history[1:]))
            angular_change = max(abs(math.atan2(math.sin(a["yaw"]-history[0]["yaw"]),
                                                math.cos(a["yaw"]-history[0]["yaw"]))) for a in history)
            if distance/duration <= .3*speed_m_s+1e-8 and angular_change <= .1:
                eligible.add(ident)
        poses = [{"id": a["id"], "x": -a["row"], "y": -a["col"], "rz": a["yaw"]}
                 for a in frame["agents"]]
        measurements = [measure_approach(a, poses, eligible, radii) for a in approaches]
        result.append({"offset_s": frame["offset_s"], "approaches": measurements,
                       "max_queue": max((m["queued_robots"] for m in measurements), default=0),
                       "queued_robots": sum(m["queued_robots"] for m in measurements),
                       "max_tail_m": max((m["queue_tail_extent_m"] for m in measurements), default=0),
                       "multi_robot_queue": any(m["queued_robots"] >= 2 for m in measurements),
                       "spillback_warning": any(m["spillback_warning"] for m in measurements)})
    return result


def crossing_events(frames, zones):
    """First observed entry/exit per robot and zone; ongoing visits enter at0.

    Entry/exit are discrete pose-boundary crossings, not native station events.
    Unfinished visits retain exit=None, rather than being dropped or extrapolated.
    """
    visits, active, done = {}, {}, set()
    for frame in frames:
        for agent in frame["agents"]:
            for zone in zones:
                key = (agent["id"], zone["id"])
                present = inside(agent, zone)
                if present and key not in active and key not in done:
                    active[key] = {"id": agent["id"], "zone": zone["id"],
                                   "entry_s": frame["offset_s"], "exit_s": None,
                                   "already_inside_at_origin": frame["offset_s"] == 0}
                    visits[key] = active[key]
                elif not present and key in active:
                    active[key]["exit_s"] = frame["offset_s"]
                    del active[key]; done.add(key)
    return visits
