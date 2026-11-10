#!/usr/bin/env python3
"""Generate a connected nine-zone warehouse with private task bays.

The 3x3 network has four two-way corners, four three-way T junctions,
and one four-way crossing. A zone owns three initially parked robots by
default. Each robot visits a pickup in its origin zone and then delivers
to a private bay in another zone. Payloads are experiment metadata; the
native simulator executes the two finite goal visits.
"""

import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ZONE_COLORS = ("#2563eb", "#0d9488", "#ea580c", "#9333ea", "#be123c",
               "#047857", "#0369a1", "#a16207", "#475569")
DIRECTIONS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
SPACING = 20
SIZE = 61
SAFETY = {
    "enabled": True,
    "robot_trigger": "body_in_outer", "inner_stop_enabled": False,
    "slow_front_m": 1.2, "slow_back_m": 0.2, "slow_half_width_m": 0.3,
    "stop_front_m": 0.3, "stop_back_m": 0.15, "stop_half_width_m": 0.18,
    "speed_scale": 0.2, "robot_radius_m": 0.085,
}


def build_case(robots_per_zone=3):
    """Return map, scenario, runner configuration, and task manifest."""
    if type(robots_per_zone) is not int or robots_per_zone not in (2, 3):
        raise ValueError("robots_per_zone must be 2 or 3")
    count = 9 * robots_per_zone
    name = f"nine_zones_{count}_robots"
    layout = [["@"] * SIZE for _ in range(SIZE)]
    zones, bays = [], {}

    def open_cell(cell):
        row, col = cell
        layout[row][col] = "."

    for zone_row in range(3):
        for zone_col in range(3):
            zone_id = f"Z{zone_row}{zone_col}"
            center = (10 + SPACING * zone_row, 10 + SPACING * zone_col)
            bounds = {"rows": [max(1, SPACING * zone_row), SPACING * (zone_row + 1) - 1],
                      "cols": [max(1, SPACING * zone_col), SPACING * (zone_col + 1) - 1]}
            neighbors, ports = {}, []
            for direction, (dr, dc) in DIRECTIONS.items():
                other_row, other_col = zone_row + dr, zone_col + dc
                if not (0 <= other_row < 3 and 0 <= other_col < 3):
                    continue
                other = f"Z{other_row}{other_col}"
                neighbors[direction] = other
                for distance in range(SPACING + 1):
                    for transverse in (-1, 0, 1):
                        open_cell((center[0] + distance * dr + transverse * dc,
                                   center[1] + distance * dc - transverse * dr))
                if dr:
                    distance = abs(bounds["rows"][0 if dr < 0 else 1] - center[0])
                else:
                    distance = abs(bounds["cols"][0 if dc < 0 else 1] - center[1])
                ports.append({
                    "direction": direction, "neighbor": other,
                    "outward_normal": [dr, dc],
                    "cells": [[center[0] + distance * dr + transverse * dc,
                               center[1] + distance * dc - transverse * dr]
                              for transverse in (-1, 0, 1)],
                })
            # Spacing between side bays keeps each bay a degree-one leaf.
            candidates = []
            for distance in (3, 5, 7, 9):
                for transverse in (-2, 2):
                    for direction in neighbors:
                        dr, dc = DIRECTIONS[direction]
                        candidates.append((center[0] + distance * dr + transverse * dc,
                                           center[1] + distance * dc - transverse * dr))
            owned = candidates[:3 * robots_per_zone]
            for cell in owned:
                open_cell(cell)
            bays[zone_id] = {
                "starts": owned[:robots_per_zone],
                "pickups": owned[robots_per_zone:2 * robots_per_zone],
                "dropoffs": owned[2 * robots_per_zone:],
            }
            zones.append({
                "id": zone_id, "grid_index": [zone_row, zone_col],
                "center": list(center), "bounds": bounds,
                "conflict_zone": {"rows": [center[0] - 1, center[0] + 1],
                                  "cols": [center[1] - 1, center[1] + 1]},
                "kind": {2: "two_way_corner", 3: "three_way_t", 4: "four_way"}[len(neighbors)],
                "degree": len(neighbors), "neighbors": neighbors, "ports": ports,
                "color": ZONE_COLORS[len(zones)],
                "bays": {kind: [list(cell) for cell in cells]
                         for kind, cells in bays[zone_id].items()},
            })

    zone_ids = [zone["id"] for zone in zones]
    matrix = [[0] * 9 for _ in range(9)]
    destination_slots = Counter()
    agents, task_records = [], []
    for source_index, zone in enumerate(zones):
        source = zone["id"]
        for local_index in range(robots_per_zone):
            destination_index = (source_index + 4 + local_index) % 9
            destination = zone_ids[destination_index]
            destination_slot = destination_slots[destination]
            destination_slots[destination] += 1
            matrix[source_index][destination_index] += 1
            robot_id = len(agents)
            start = list(bays[source]["starts"][local_index])
            pickup = list(bays[source]["pickups"][local_index])
            dropoff = list(bays[destination]["dropoffs"][destination_slot])
            task_id, payload_id = f"transport-{robot_id:03d}", f"payload-{robot_id:03d}"
            agents.append({
                "id": robot_id, "zone_id": source, "approach": source,
                "pickup_zone_id": source, "destination_zone_id": destination,
                "start": start, "goals": [pickup, dropoff], "release_at_s": 0,
                "task_id": task_id, "payload_id": payload_id, "color": zone["color"],
            })
            task_records.append({
                "id": task_id, "robot_id": robot_id, "payload_id": payload_id,
                "source_zone_id": source, "destination_zone_id": destination,
                "release_at_s": 0, "payload": {"type": "tote", "mass_kg": 5},
                "stages": [
                    {"index": 0, "operation": "pickup", "zone_id": source,
                     "location": pickup, "nominal_service_time_s": 2},
                    {"index": 1, "operation": "dropoff", "zone_id": destination,
                     "location": dropoff, "nominal_service_time_s": 2},
                ],
                "finish_behavior": "park_in_private_dropoff_bay",
            })

    map_document = {
        "name": name, "layout": ["".join(row) for row in layout],
        "milp_runtime": None, "optimize_wait": False, "weight": False,
        "n_row": SIZE, "n_col": SIZE, "n_endpoint": 0, "n_agent_loc": 0,
        "maxtime": 1000,
    }
    scenario = {
        "schema_version": 1, "name": name,
        "description": (
            f"Nine connected warehouse zones with {robots_per_zone} robots per zone. "
            "Four two-way corners, four T junctions, and one four-way crossing are "
            "joined by three-cell aisles. Each robot picks up locally, delivers "
            "to another zone, then parks in its private dropoff bay."
        ),
        "map_file": "map.json", "tasks_file": "tasks.json",
        "coordinate_order": ["row", "column"], "cell_size_m": 1,
        "initial_orientation": "north", "ticks_per_second": 10,
        "task_mode": "finite", "agents": agents, "zones": zones,
        "safety": dict(SAFETY),
        # Compatibility only: zone analysis uses the nine explicit regions.
        "conflict_zone": {"rows": [9, 51], "cols": [9, 51]},
        "approach_zones": {zone["id"]: zone["bounds"] for zone in zones},
        "network": {
            "shape": [3, 3], "aisle_width_cells": 3,
            "center_spacing_cells": SPACING, "robots_per_zone": robots_per_zone,
            "initial_robot_matrix": [[robots_per_zone] * 3 for _ in range(3)],
            "zone_order": zone_ids, "allocation_matrix": matrix,
            "edges": [[zone["id"], neighbor] for zone in zones
                      for neighbor in zone["neighbors"].values() if zone["id"] < neighbor],
        },
        "notes": [
            "Zone bounds and conflict regions use inclusive grid coordinates.",
            "The top-level conflict_zone is a compatibility bounding box, not one intersection.",
            "A two-way zone is a corner joining two incident aisles.",
            "All robots initially face north; coordinates are one metre apart.",
            "Start, pickup, and dropoff bays are distinct degree-one leaves.",
            "Every source and destination receives the same number of transport tasks.",
            "Payload identity and mass are metadata; they do not alter vehicle dynamics.",
            "Task goals are native finite visits; service timing is the controller's station action.",
            "Nominal two-second service times describe the current controller, not configurable dwell overrides.",
            "All robots are released at zero; destinations are private bays rather than aisle cells.",
            "Safety rectangles use robot-body forward/lateral dimensions in metres; all robots inherit the scenario policy.",
            "Only the outer slowdown rectangle is active by default; the retained inner dimensions are optional.",
            "The initial robot trigger is body_in_outer; compare outer_overlap with the matched runner experiment.",
        ],
    }
    config = {
        "image": "warehouse-congestion:local", "scenario": "scenario.json",
        "validation_mode": "stress", "default_runs": 1, "preview_port": 3004,
        "wall_timeout_s": 300, "record_plans": True,
        "parameters": {
            "num_agents": count, "seed": 42, "n_threads": 1,
            "sim_duration": 1800, "ticks_per_second": 10, "planner": "PBS",
            "task_assigner_type": "distinct_one_goal", "backup_solver": "PIBT",
            "single_agent_solver": "SIPP", "planner_invoke_policy": "default",
            "sim_window_tick": 20, "planning_window": 20, "cutoffTime": 1,
            "velocity": 100, "stop_at_congestion": False, "rotation": False,
            "screen": 0,
        },
    }
    tasks = {
        "schema_version": 1, "scenario": name, "zone_order": zone_ids,
        "allocation_matrix": matrix,
        "allocation_definition": "Rows are pickup/source zones; columns are delivery zones; entries count transport tasks.",
        "initial_robot_counts": {zone: robots_per_zone for zone in zone_ids},
        "initial_robot_matrix": [[robots_per_zone] * 3 for _ in range(3)],
        "tasks": task_records,
        "execution_note": (
            "scenario.json goals execute pickup then dropoff as two finite native goal visits. "
            "The robot remains in its private dropoff bay afterwards. Payload mass and identity "
            "are labels; loading physics and configurable service times are not implemented."
        ),
    }
    return map_document, scenario, config, tasks


def render_svg(map_document, scenario):
    """Render the network and its readable origin-destination allocation."""
    cell, left, top = 14, 30, 94
    map_width = map_document["n_col"] * cell
    width, height = left * 2 + map_width + 326, top + map_width + 60
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Nine-zone warehouse scenario</title>',
        f'<desc id="desc">{escape(scenario["description"])}</desc>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#0f172a">',
        f'<text x="{left}" y="35" font-size="24" font-weight="700">Nine connected warehouse zones</text>',
        f'<text x="{left}" y="60" font-size="13" fill="#475569">'
        f'{len(scenario["agents"])} robots · 3 m aisles · local pickup → cross-zone delivery · private bays</text>',
    ]
    for row, values in enumerate(map_document["layout"]):
        for col, value in enumerate(values):
            fill = "#dbe3ec" if value == "@" else "white"
            elements.append(f'<rect x="{left + col * cell}" y="{top + row * cell}" '
                            f'width="{cell}" height="{cell}" fill="{fill}"/>')
    for zone in scenario["zones"]:
        r0, r1 = zone["bounds"]["rows"]
        c0, c1 = zone["bounds"]["cols"]
        color = zone["color"]
        elements.append(f'<rect x="{left + c0 * cell}" y="{top + r0 * cell}" '
                        f'width="{(c1-c0+1)*cell}" height="{(r1-r0+1)*cell}" '
                        f'fill="none" stroke="{color}" stroke-width="1" stroke-dasharray="4 4"/>')
        label_x, label_y = left + (c0 + 1) * cell, top + (r0 + 2) * cell
        label = f'{zone["id"]}: {zone["degree"]}-way' + (' corner' if zone["degree"] == 2 else '')
        elements.append(f'<text x="{label_x}" y="{label_y}" font-size="12" font-weight="700" '
                        f'fill="{color}">{escape(label)}</text>')
        row, col = zone["center"]
        elements.append(f'<rect x="{left+(col-1)*cell}" y="{top+(row-1)*cell}" '
                        f'width="{3*cell}" height="{3*cell}" fill="#fbbf24" opacity=".45"/>')
        for port in zone["ports"]:
            row, col = port["cells"][1]
            elements.append(f'<circle cx="{left+(col+.5)*cell}" cy="{top+(row+.5)*cell}" '
                            f'r="2.5" fill="{color}"><title>{zone["id"]} → {port["neighbor"]}</title></circle>')
    for agent in scenario["agents"]:
        row, col = agent["start"]
        x, y = left + (col + .5) * cell, top + (row + .5) * cell
        color, robot_id = agent["color"], agent["id"]
        safety = {**scenario.get("safety", {}), **agent.get("safety", {})}
        if safety.get("enabled"):
            for prefix, polygon_color in (("slow", "#f59e0b"), ("stop", "#dc2626")):
                if prefix == "stop" and not safety.get("inner_stop_enabled", True):
                    continue
                half_width = safety[f"{prefix}_half_width_m"] * cell
                front, back = safety[f"{prefix}_front_m"] * cell, safety[f"{prefix}_back_m"] * cell
                elements.append(f'<rect x="{x-half_width}" y="{y-front}" '
                                f'width="{2*half_width}" height="{front+back}" '
                                f'fill="{polygon_color}" fill-opacity=".18" '
                                f'stroke="{polygon_color}" stroke-width=".8"/>')
        elements.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>'
                        f'<text x="{x}" y="{y+2.4}" font-size="6.9" text-anchor="middle" '
                        f'font-weight="700" fill="white">{robot_id}</text>')
        for prefix, goal in zip(("P", "D"), agent["goals"]):
            row, col = goal
            x, y = left + col * cell, top + row * cell
            elements.append(f'<rect x="{x+.5}" y="{y+.5}" width="13" height="13" rx="1.5" '
                            f'fill="white" stroke="{color}" stroke-width="1.1"/>'
                            f'<text x="{x+7}" y="{y+9.5}" font-size="6" text-anchor="middle" '
                            f'font-weight="700" fill="{color}">{prefix}{robot_id}</text>')
    side_x = left + map_width + 24
    elements.append(f'<text x="{side_x}" y="{top+15}" font-size="16" font-weight="700">Task allocation</text>')
    elements.append(f'<text x="{side_x}" y="{top+38}" font-size="11" fill="#475569">Source rows → destination columns</text>')
    table_y, table_cell = top + 70, 27
    zone_order = scenario["network"]["zone_order"]
    for col, zone_id in enumerate(zone_order):
        elements.append(f'<text x="{side_x+42+(col+.5)*table_cell}" y="{table_y-8}" font-size="8" '
                        f'text-anchor="middle">{zone_id}</text>')
    for row, zone_id in enumerate(zone_order):
        y = table_y + row * table_cell
        elements.append(f'<text x="{side_x}" y="{y+18}" font-size="10" font-weight="700">{zone_id}</text>')
        for col, value in enumerate(scenario["network"]["allocation_matrix"][row]):
            x = side_x + 42 + col * table_cell
            fill = "#dbeafe" if value else "#f1f5f9"
            elements.append(f'<rect x="{x}" y="{y}" width="26" height="26" fill="{fill}"/>'
                            f'<text x="{x+13}" y="{y+18}" font-size="11" '
                            f'text-anchor="middle">{value}</text>')
    y = table_y + 9 * table_cell + 36
    for index, line in enumerate((
            "Each row and column has the same total.",
            "Filled circle: robot start and ID",
            "P: pickup bay · D: dropoff / final parking",
            "Colors identify the robot's starting zone.",
            "Yellow: local intersection region",
            "Small dots: zone connection ports",
            "Every robot initially faces north.",
            "Amber outline: active outer slow rectangle",
            "Inner stopping rectangle is optional (off).",
            "Payload mass is metadata only.")):
        elements.append(f'<text x="{side_x}" y="{y+index*23}" font-size="11" '
                        f'fill="#475569">{escape(line)}</text>')
    elements.append(f'<text x="{left}" y="{top+map_width+31}" font-size="12" fill="#475569">'
                    'Map cells are 1 m. Junction degree counts connected aisles; bay spurs are task stations.</text>')
    elements.append('</g></svg>')
    return "\n".join(elements) + "\n"


def write_case(directory, robots_per_zone=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    map_document, scenario, config, tasks = build_case(robots_per_zone)
    for filename, document in (("map.json", map_document), ("scenario.json", scenario),
                               ("config.json", config), ("tasks.json", tasks)):
        (directory / filename).write_text(json.dumps(document, indent=2) + "\n")
    with (directory / "zone_allocation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_zone/destination_zone", *tasks["zone_order"]])
        for zone_id, counts in zip(tasks["zone_order"], tasks["allocation_matrix"]):
            writer.writerow([zone_id, *counts])
    (directory / "layout.svg").write_text(render_svg(map_document, scenario))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "nine_zones")
    parser.add_argument("--robots-per-zone", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    print(write_case(args.output_dir, args.robots_per_zone))


if __name__ == "__main__":
    main()
