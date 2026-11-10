#!/usr/bin/env python3
"""Generate matched, finite T and four-way junction experiments.

The center is either a three-cell-wide intersection or a single-cell crossing.
Approach aisles always remain three cells wide, and finished robots occupy
private side bays. Thus a hard case need not be an accidentally blocked map.
"""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DIRECTIONS = {
    "north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1),
}
COLORS = {
    "north": "#0d9488", "east": "#ea580c",
    "south": "#9333ea", "west": "#2563eb",
}
CASES = tuple(
    {"name": f"{geometry}_{width_name}_{count}", "geometry": geometry,
     "num_agents": count, "core_width": width}
    for geometry in ("t", "four_way")
    for width_name, count, width in (("wide", 12, 3), ("wide", 24, 3),
                                     ("narrow", 24, 1), ("narrow", 12, 1))
)


def build_case(geometry="t", num_agents=24, core_width=3, arm_length=22,
               name=None):
    """Return map, scenario, and runner config; coordinates are [row, column].

    Geometry and load can be changed independently. Bays and starts use the
    same coordinates across the matched wide/narrow pair. Every incoming arm
    sends equal-sized streams to the other arms, as evenly as divisibility
    permits. At least one empty lane remains beside all initial robots.
    """
    if geometry not in ("t", "four_way"):
        raise ValueError("geometry must be 't' or 'four_way'")
    approaches = tuple(direction for direction in DIRECTIONS
                       if geometry == "four_way" or direction != "south")
    if type(num_agents) is not int or num_agents < len(approaches) or num_agents % len(approaches):
        raise ValueError("num_agents must be a positive multiple of the number of arms")
    if core_width not in (1, 3):
        raise ValueError("core_width must be 1 or 3")
    per_arm = num_agents // len(approaches)
    if type(arm_length) is not int or arm_length < 2 * per_arm + 4:
        raise ValueError("arm_length must leave space for all starts and private bays")
    name = name or f"{geometry}_{'wide' if core_width == 3 else 'narrow'}_{num_agents}"
    center = (arm_length + 2, arm_length + 2)
    rows = center[0] + 4 if geometry == "t" else 2 * arm_length + 5
    cols = 2 * arm_length + 5
    layout = [["@"] * cols for _ in range(rows)]

    def point(approach, distance, transverse=0):
        # Positive transverse is the right side when looking out of the arm.
        dr, dc = DIRECTIONS[approach]
        return (center[0] + distance * dr + transverse * dc,
                center[1] + distance * dc - transverse * dr)

    def open_cell(position):
        row, col = position
        layout[row][col] = "."

    for approach in approaches:
        for distance in range(arm_length + 1):
            for transverse in (-1, 0, 1):
                open_cell(point(approach, distance, transverse))
    if core_width == 1:
        # Retain only each arm's centerline inside the central 5x5 square.
        for row in range(center[0] - 2, center[0] + 3):
            for col in range(center[1] - 2, center[1] + 3):
                layout[row][col] = "@"
        for approach in approaches:
            for distance in range(3):
                open_cell(point(approach, distance))

    bays = {}
    for approach in approaches:
        bays[approach] = [point(approach, 5 + 2 * index, 2)
                          for index in range(per_arm)]
        for position in bays[approach]:
            open_cell(position)
    next_bay = Counter()
    agents = []
    for arm_index, approach in enumerate(approaches):
        for index in range(per_arm):
            destination = approaches[(arm_index + 1 + index % (len(approaches) - 1))
                                     % len(approaches)]
            goal = bays[destination][next_bay[destination]]
            next_bay[destination] += 1
            agents.append({
                "id": len(agents), "approach": approach,
                "destination_arm": destination,
                "start": list(point(approach, 4 + 2 * index, -1)),
                "goals": [list(goal)], "release_at_s": 0,
                "color": COLORS[approach],
            })

    zone = {"rows": [center[0] - 2, center[0] + 2],
            "cols": [center[1] - 2, center[1] + 2]}
    approach_zones = {}
    ports = []
    for approach in approaches:
        corners = [point(approach, distance, transverse)
                   for distance in (3, arm_length) for transverse in (-1, 1)]
        approach_zones[approach] = {
            "rows": [min(p[0] for p in corners), max(p[0] for p in corners)],
            "cols": [min(p[1] for p in corners), max(p[1] for p in corners)],
        }
        ports.append({
            "name": approach, "outward_normal": list(DIRECTIONS[approach]),
            "cells": [list(point(approach, arm_length, transverse))
                      for transverse in (-1, 0, 1)],
            "nominal_inbound_cell": list(point(approach, arm_length, -1)),
            "nominal_outbound_cell": list(point(approach, arm_length, 1)),
        })
    core_free_cells = sum(layout[row][col] == "."
                          for row in range(zone["rows"][0], zone["rows"][1] + 1)
                          for col in range(zone["cols"][0], zone["cols"][1] + 1))
    map_document = {
        "name": name, "layout": ["".join(row) for row in layout],
        "milp_runtime": None, "optimize_wait": False, "weight": False,
        "n_row": rows, "n_col": cols, "n_endpoint": 0, "n_agent_loc": 0,
        "maxtime": 1000,
    }
    scenario = {
        "schema_version": 1, "name": name,
        "description": (
            f"{num_agents} simultaneous robots at a {geometry.replace('_', '-')} "
            f"junction with a {core_width}-cell crossing and three-cell aisles. "
            "Every route crosses the center and ends in a private side bay."
        ),
        "map_file": "map.json", "coordinate_order": ["row", "column"],
        "initial_orientation": "north", "ticks_per_second": 10,
        "task_mode": "finite", "agents": agents, "conflict_zone": zone,
        "approach_zones": approach_zones,
        "benchmark": {
            "family": geometry, "num_agents": num_agents,
            "robots_per_approach": per_arm, "core_width_cells": core_width,
            "arm_width_cells": 3, "arm_length_cells": arm_length,
            "core_free_cells": core_free_cells,
            "free_cells": sum(row.count(".") for row in layout),
            "release_pattern": "simultaneous",
            "movement_counts": dict(sorted(Counter(
                f"{agent['approach']}->{agent['destination_arm']}"
                for agent in agents).items())),
        },
        "module": {
            "geometry": geometry, "anchor": list(center),
            "bounds": {"rows": [0, rows - 1], "cols": [0, cols - 1]},
            "ports": ports, "cell_size_m": 1,
            "composition_note": (
                "Ports terminate at free aisle end cells behind the boundary wall. "
                "A future composer must open that wall, join matching ports, "
                "transform every grid coordinate, and replace finite parking tasks."
            ),
        },
        "notes": [
            "Zone bounds are inclusive; only free cells can contain a robot.",
            "Wide and narrow variants retain identical starts and destinations.",
            "All initial headings are north, as required by current runtime initialization.",
            "Lane directions are nominal; PBS may use any free cell.",
            "Private goal bays have degree one and never obstruct through lanes.",
            "Every individual route remains feasible with all other starts and goals blocked.",
            "Difficulty is established by measured outcomes, not the case name.",
        ],
    }
    config = {
        "image": "warehouse-congestion:local", "scenario": "scenario.json",
        "validation_mode": "stress", "default_runs": 1,
        "preview_port": 3003, "wall_timeout_s": 300, "record_plans": True,
        "parameters": {
            "num_agents": num_agents, "seed": 42, "n_threads": 1,
            "sim_duration": 1200, "ticks_per_second": 10, "planner": "PBS",
            "task_assigner_type": "distinct_one_goal", "backup_solver": "PIBT",
            "single_agent_solver": "SIPP", "planner_invoke_policy": "default",
            "sim_window_tick": 20, "planning_window": 20, "cutoffTime": 1,
            "velocity": 100, "stop_at_congestion": False, "rotation": False,
            "screen": 0,
        },
    }
    return map_document, scenario, config


def render_svg(map_document, scenario):
    """Draw the walls, actual free core, robot starts, and private destinations."""
    cell, left, top = 17, 38, 96
    width = left * 2 + map_document["n_col"] * cell
    height = top + map_document["n_row"] * cell + 74
    zone = scenario["conflict_zone"]
    benchmark = scenario["benchmark"]
    title = scenario["name"].replace("_", " ")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(scenario["description"])}</desc>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#0f172a">',
        f'<text x="{left}" y="36" font-size="24" font-weight="700">{escape(title)}</text>',
        f'<text x="{left}" y="61" font-size="13" fill="#475569">'
        f'{len(scenario["agents"])} robots · {benchmark["core_width_cells"]}-cell crossing '
        '· three-cell aisles · private parking bays</text>',
    ]
    for row, cells in enumerate(map_document["layout"]):
        for col, value in enumerate(cells):
            inside = (zone["rows"][0] <= row <= zone["rows"][1]
                      and zone["cols"][0] <= col <= zone["cols"][1])
            fill = "#d4dde7" if value == "@" else "#fef3c7" if inside else "white"
            elements.append(
                f'<rect x="{left + col * cell}" y="{top + row * cell}" '
                f'width="{cell}" height="{cell}" fill="{fill}" stroke="#cbd5e1" stroke-width=".4"/>')
    for agent in scenario["agents"]:
        row, col = agent["start"]
        x, y = left + (col + .5) * cell, top + (row + .5) * cell
        color, robot_id = agent["color"], agent["id"]
        elements.append(
            f'<circle cx="{x}" cy="{y}" r="6.8" fill="{color}"/>'
            f'<text x="{x}" y="{y + 2.8}" font-size="7.7" text-anchor="middle" '
            f'font-weight="700" fill="white">{robot_id}</text>')
        row, col = agent["goals"][0]
        x, y = left + col * cell, top + row * cell
        elements.append(
            f'<rect x="{x + 1}" y="{y + 1}" width="15" height="15" rx="2" '
            f'fill="white" stroke="{color}" stroke-width="1.5"/>'
            f'<text x="{x + 8.5}" y="{y + 11.2}" font-size="7" text-anchor="middle" '
            f'font-weight="700" fill="{color}">G{robot_id}</text>')
    for port in scenario["module"]["ports"]:
        row, col = port["cells"][1]
        x, y = left + (col + .5) * cell, top + (row + .5) * cell
        elements.append(
            f'<circle cx="{x}" cy="{y}" r="3" fill="#475569"/>'
            f'<title>Composition port: {escape(port["name"])}</title>')
    baseline = top + map_document["n_row"] * cell
    elements.extend([
        f'<text x="{left}" y="{baseline + 27}" font-size="12" fill="#475569">'
        'Filled circle: robot ID · G: parking goal · yellow: conflict zone · small gray dot: port</text>',
        f'<text x="{left}" y="{baseline + 49}" font-size="12" fill="#475569">'
        'All robots initially face north. Colors identify the origin arm, including at destinations.</text>',
        '</g></svg>',
    ])
    return "\n".join(elements) + "\n"


def write_case(directory, case):
    directory.mkdir(parents=True, exist_ok=True)
    map_document, scenario, config = build_case(**case)
    for filename, document in (("map.json", map_document), ("scenario.json", scenario),
                               ("config.json", config)):
        (directory / filename).write_text(json.dumps(document, indent=2) + "\n")
    (directory / "layout.svg").write_text(render_svg(map_document, scenario))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "cases")
    parser.add_argument("--case", choices=[case["name"] for case in CASES], action="append",
                        help="Generate selected defaults; omit to generate all eight.")
    args = parser.parse_args()
    for case in CASES:
        if args.case is None or case["name"] in args.case:
            print(write_case(args.output_dir / case["name"], case))


if __name__ == "__main__":
    main()
