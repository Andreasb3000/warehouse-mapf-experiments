#!/usr/bin/env python3
"""Generate a compact 1,000-robot T-junction designed to overload the crossing.

Every robot starts in an approach arm, targets a different arm, and is released
at time zero. Goals occupy travel lanes, so finished robots can worsen the jam.
This workload is intentionally separate from the six-robot correctness fixture.
"""

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import random


ROWS, COLS = 81, 121
N_ROBOTS = 1000
SEED = 42
COLORS = {"west": "#2563eb", "east": "#e67e22", "north": "#16a085"}
ZONE = {"rows": [61, 77], "cols": [52, 68]}
APPROACHES = {
    "west": {"rows": [61, 77], "cols": [1, 51]},
    "east": {"rows": [61, 77], "cols": [69, 119]},
    "north": {"rows": [1, 60], "cols": [52, 68]},
}


def cells(zone):
    return [(row, col)
            for row in range(zone["rows"][0], zone["rows"][1] + 1)
            for col in range(zone["cols"][0], zone["cols"][1] + 1)]


def components(free_cells):
    """Label free-cell connected components with four-neighbor movement."""
    remaining = set(free_cells)
    labels = {}
    label = 0
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = deque([first])
        labels[first] = label
        while queue:
            row, col = queue.popleft()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    labels[neighbor] = label
                    queue.append(neighbor)
        label += 1
    return labels


def validate_fixture(map_document, scenario):
    """Check the large fixture's physical and traffic invariants before saving."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    layout = map_document["layout"]
    require(len(layout) == ROWS and all(len(row) == COLS for row in layout),
            "Map dimensions differ from the stress geometry")
    free = {(row, col) for row, line in enumerate(layout)
            for col, value in enumerate(line) if value == "."}
    require(len(free) == 3043, "Unexpected free-cell count")
    require(len(set(components(free).values())) == 1, "Free space is disconnected")
    central = set(cells(scenario["conflict_zone"]))
    arm_labels = components(free - central)
    require(len(set(arm_labels.values())) == 3, "Expected exactly three approach arms")
    agents = scenario["agents"]
    require(len(agents) == N_ROBOTS, "Expected exactly 1,000 robots")
    require([agent["id"] for agent in agents] == list(range(N_ROBOTS)),
            "Robot IDs must be contiguous and ordered")
    starts = [tuple(agent["start"]) for agent in agents]
    goals = [tuple(agent["goals"][0]) for agent in agents]
    require(len(set(starts)) == N_ROBOTS, "Duplicate robot starts")
    require(len(set(goals)) == N_ROBOTS, "Duplicate robot goals")
    require(set(starts + goals) <= free, "Start or goal is an obstacle")
    require(not (set(starts + goals) & central), "Starts and goals must lie in arms")
    for agent, start, goal in zip(agents, starts, goals):
        require(len(agent["goals"]) == 1, "Expected one finite goal per robot")
        require(start != goal, "A robot starts at its own goal")
        require(arm_labels[start] != arm_labels[goal],
                "Every route must pass through the central junction")
        require(start in cells(APPROACHES[agent["approach"]]),
                "Start is outside its labeled approach")
        require(goal in cells(APPROACHES[agent["destination_arm"]]),
                "Goal is outside its labeled destination arm")
        require(agent["release_at_s"] == 0, "All robots must be released together")
    require(Counter(agent["approach"] for agent in agents) ==
            {"west": 334, "east": 333, "north": 333}, "Unbalanced approach counts")
    require(scenario["initial_orientation"] == "north", "Unsupported initial heading")
    return {
        "robots": N_ROBOTS,
        "free_cells": len(free),
        "robot_density": round(N_ROBOTS / len(free), 6),
        "connected_arms": 3,
        "all_routes_require_conflict_zone": True,
        "approach_counts": dict(Counter(agent["approach"] for agent in agents)),
        "destination_counts": dict(Counter(agent["destination_arm"] for agent in agents)),
    }


def build_fixture():
    free = set(cells(ZONE))
    for zone in APPROACHES.values():
        free.update(cells(zone))
    layout = ["".join("." if (row, col) in free else "@" for col in range(COLS))
              for row in range(ROWS)]
    # Restrict starts to the nearest forty cells of each arm. This creates
    # approximately 49% occupancy near the crossing while leaving maneuvering
    # space farther away. All central cells initially remain empty.
    start_regions = {
        "west": {"rows": [61, 77], "cols": [12, 51]},
        "east": {"rows": [61, 77], "cols": [69, 108]},
        "north": {"rows": [21, 60], "cols": [52, 68]},
    }
    counts = {"west": 334, "east": 333, "north": 333}
    destinations = {"west": ("east", "north"),
                    "east": ("west", "north"),
                    "north": ("west", "east")}
    starts_rng = random.Random(SEED)
    goals_rng = random.Random(SEED + 1)
    goal_cells = {arm: iter(goals_rng.sample(cells(APPROACHES[arm]), count))
                  for arm, count in counts.items()}
    agents = []
    for approach, count in counts.items():
        starts = starts_rng.sample(cells(start_regions[approach]), count)
        for index, start in enumerate(starts):
            destination = destinations[approach][index % 2]
            agents.append({
                "id": len(agents),
                "approach": approach,
                "destination_arm": destination,
                "start": list(start),
                "goals": [list(next(goal_cells[destination]))],
                "release_at_s": 0,
                "color": COLORS[approach],
            })

    map_document = {
        "name": "three_way_stress_81x121",
        "layout": layout,
        "milp_runtime": None,
        "optimize_wait": False,
        "weight": False,
        "n_row": ROWS,
        "n_col": COLS,
        "n_endpoint": 0,
        "n_agent_loc": 0,
        "maxtime": 1000,
    }
    scenario = {
        "schema_version": 1,
        "name": "three_way_stress_1000_robots",
        "description": (
            "1,000 robots converge on a broad T-junction at time zero. Every "
            "goal lies in a different arm. Goals occupy travel lanes, so "
            "completed robots can deliberately amplify congestion."
        ),
        "map_file": "three_way.map.json",
        "coordinate_order": ["row", "column"],
        "initial_orientation": "north",
        "ticks_per_second": 10,
        "task_mode": "finite",
        "workload_kind": "stress",
        "seed": SEED,
        "agents": agents,
        "conflict_zone": ZONE,
        "approach_zones": APPROACHES,
        "start_regions": start_regions,
        "notes": [
            "This is an overload experiment; finishing every task is not expected.",
            "All 1,000 robots are released at time zero and initially face north.",
            "Start occupancy is 32.9% of all free cells and 49.0% of start regions.",
            "Goals are unique travel-lane cells; parked robots can block traffic.",
            "All starts and goals are outside the central 17-by-17 conflict zone.",
            "Every start-to-goal route must traverse the central conflict zone.",
            "The planner may use any free cell; there are no enforced one-way lanes.",
            "Zone bounds include both endpoint cells.",
        ],
    }
    scenario["geometry_validation"] = validate_fixture(map_document, scenario)
    return map_document, scenario


def render_svg(scenario):
    cell, left, top = 6, 70, 108
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="866" height="716" '
        'viewBox="0 0 866 716" role="img" aria-labelledby="title desc">',
        '<title id="title">1,000-robot T-junction stress test</title>',
        '<desc id="desc">Dense groups of robots start in the west, east, and '
        'north arms of a T-junction. Every destination is in another arm.</desc>',
        '<rect width="866" height="716" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#0f172a">',
        '<text x="38" y="40" font-size="25" font-weight="700">'
        '1,000 robots. One T-junction.</text>',
        '<text x="38" y="65" font-size="14" fill="#475569">'
        '334 from the west · 333 from the east · 333 from the north · released together'
        '</text>',
        f'<rect x="{left}" y="{top}" width="{COLS * cell}" '
        f'height="{ROWS * cell}" fill="#d4dde7"/>',
    ]
    for zone in [*APPROACHES.values(), ZONE]:
        row, col = zone["rows"][0], zone["cols"][0]
        height = (zone["rows"][1] - row + 1) * cell
        width = (zone["cols"][1] - col + 1) * cell
        color = "#fef3c7" if zone == ZONE else "#ffffff"
        elements.append(
            f'<rect x="{left + col * cell}" y="{top + row * cell}" '
            f'width="{width}" height="{height}" fill="{color}"/>'
        )
    # Goals are subtle rings; solid dots above them show the starting population.
    for agent in scenario["agents"]:
        row, col = agent["goals"][0]
        elements.append(
            f'<circle cx="{left + (col + .5) * cell}" '
            f'cy="{top + (row + .5) * cell}" r="2.0" fill="none" '
            'stroke="#94a3b8" stroke-width="0.6"/>'
        )
    for agent in scenario["agents"]:
        row, col = agent["start"]
        elements.append(
            f'<circle cx="{left + (col + .5) * cell}" '
            f'cy="{top + (row + .5) * cell}" r="2.25" '
            f'fill="{agent["color"]}"/>'
        )
    elements.extend([
        '<text x="433" y="96" text-anchor="middle" font-size="15" '
        'font-weight="700" fill="#16a085">NORTH · 333</text>',
        '<text x="38" y="526" text-anchor="middle" font-size="16" '
        'font-weight="700" fill="#2563eb">W</text>',
        '<text x="827" y="526" text-anchor="middle" font-size="16" '
        'font-weight="700" fill="#e67e22">E</text>',
        '<text x="38" y="622" font-size="13">'
        'Solid dots = robot starts     Hollow dots = destinations     Yellow = conflict zone'
        '</text>',
        '<text x="38" y="648" font-size="13" fill="#475569">'
        '81 rows × 121 columns · 17-cell-wide aisles · 3,043 free cells · 32.9% occupied'
        '</text>',
        '<text x="38" y="674" font-size="13" fill="#475569">'
        'Every route crosses the junction. Parking in travel lanes is part of this overload test.'
        '</text>',
        '<text x="38" y="698" font-size="12" fill="#64748b">'
        'Seed 42 · all robots initially face north · complete clearance is not required'
        '</text>',
        '</g></svg>',
    ])
    return "\n".join(elements) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_document, scenario = build_fixture()
    for name, document in (("three_way.map.json", map_document),
                           ("three_way.scenario.json", scenario)):
        path = args.output_dir / name
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(path)
    svg_path = args.output_dir / "three_way.svg"
    svg_path.write_text(render_svg(scenario), encoding="utf-8")
    print(svg_path)
    print(json.dumps(scenario["geometry_validation"], indent=2))


if __name__ == "__main__":
    main()
