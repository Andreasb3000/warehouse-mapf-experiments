#!/usr/bin/env python3
"""Generate the small, finite three-way T-junction fixture and its diagram.

Coordinates in scenario files are zero-based [row, column]. A robot crosses
the central junction once and parks in its own side bay. The existing runtime
initializes every robot facing north; route arrows do not override that pose.
"""

import argparse
from html import escape
import json
from pathlib import Path


ROWS = 16
COLS = 25
COLORS = {
    "west": "#2563eb",
    "east": "#e67e22",
    "north": "#16a085",
}


def build_fixture():
    """Return map and scenario documents without reading or changing files."""
    movements = [
        ("west", (13, 6), (14, 18)),
        ("west", (13, 3), (3, 14)),
        ("east", (11, 18), (10, 6)),
        ("east", (11, 21), (6, 14)),
        ("north", (6, 11), (14, 21)),
        ("north", (3, 11), (10, 3)),
    ]
    layout = [["@"] * COLS for _ in range(ROWS)]
    for row in range(1, ROWS - 1):
        for col in range(1, COLS - 1):
            if 11 <= row <= 13 or (row <= 13 and 11 <= col <= 13):
                layout[row][col] = "."

    agents = []
    for robot_id, (approach, start, goal) in enumerate(movements):
        layout[goal[0]][goal[1]] = "."
        agents.append({
            "id": robot_id,
            "approach": approach,
            "start": list(start),
            "goals": [list(goal)],
            "color": COLORS[approach],
        })

    map_document = {
        "name": "three_way_16x25",
        "layout": ["".join(row) for row in layout],
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
        "name": "three_way_6_robots",
        "description": (
            "Two robots approach from each of the west, east, and north arms. "
            "Straight and turning routes cross the T-junction, then finish "
            "in distinct side bays. All robots initially face north."
        ),
        "map_file": "three_way.map.json",
        "coordinate_order": ["row", "column"],
        "initial_orientation": "north",
        "ticks_per_second": 10,
        "task_mode": "finite",
        "agents": agents,
        "conflict_zone": {"rows": [11, 13], "cols": [11, 13]},
        "approach_zones": {
            "west": {"rows": [11, 13], "cols": [1, 10]},
            "east": {"rows": [11, 13], "cols": [14, 23]},
            "north": {"rows": [1, 10], "cols": [11, 13]},
        },
        "notes": [
            "Zone bounds include both endpoint cells.",
            "Side bays keep finished robots outside the through aisles.",
            "Travel lanes are nominal; the planner may use any free cell.",
            "Starts are simultaneous; release schedules are not modeled.",
            "Initial heading follows the existing simulator's north-facing pose.",
        ],
    }
    return map_document, scenario


def render_svg(map_document, scenario):
    """Render the fixture using only SVG primitives; no third-party packages."""
    cell, left, top = 24, 96, 112
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="792" height="634" '
        'viewBox="0 0 792 634" role="img" aria-labelledby="title desc">',
        '<title id="title">Three-way warehouse T-junction: six robots</title>',
        '<desc id="desc">A three-cell-wide T-junction with two robots per '
        'approach and a separate parking bay for each destination.</desc>',
        '<rect width="792" height="634" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#0f172a">',
        '<text x="48" y="42" font-size="25" font-weight="700">'
        'Three-way T-junction</text>',
        '<text x="48" y="68" font-size="14" fill="#475569">'
        '6 robots · 2 per approach · straight and turning routes · private parking'
        '</text>',
        '<defs>',
    ]
    for approach, color in COLORS.items():
        elements.append(
            f'<marker id="arrow-{approach}" markerWidth="7" '
            f'markerHeight="7" refX="5" refY="3.5" orient="auto">'
            f'<path d="M0 0 L7 3.5 L0 7Z" fill="{color}"/></marker>'
        )
    elements.append('</defs>')

    for row, cells in enumerate(map_document["layout"]):
        for col, value in enumerate(cells):
            fill = "#d4dde7" if value == "@" else "#ffffff"
            if 11 <= row <= 13 and 11 <= col <= 13:
                fill = "#fef3c7"
            elements.append(
                f'<rect x="{left + col * cell}" y="{top + row * cell}" '
                f'width="{cell}" height="{cell}" fill="{fill}" '
                'stroke="#cbd5e1" stroke-width="0.5"/>'
            )

    for index in range(COLS):
        elements.append(
            f'<text x="{left + (index + .5) * cell}" y="{top - 8}" '
            f'text-anchor="middle" font-size="9" fill="#64748b">{index}</text>'
        )
    for index in range(ROWS):
        elements.append(
            f'<text x="{left - 9}" y="{top + (index + .5) * cell + 3}" '
            f'text-anchor="end" font-size="9" fill="#64748b">{index}</text>'
        )

    directions = {"west": (1, 0), "east": (-1, 0), "north": (0, 1)}
    for agent in scenario["agents"]:
        robot_id, color = agent["id"], agent["color"]
        row, col = agent["start"]
        x, y = left + (col + .5) * cell, top + (row + .5) * cell
        dx, dy = directions[agent["approach"]]
        elements.append(
            f'<line x1="{x + dx * 12}" y1="{y + dy * 12}" '
            f'x2="{x + dx * 30}" y2="{y + dy * 30}" '
            f'stroke="{color}" stroke-width="2" '
            f'marker-end="url(#arrow-{escape(agent["approach"])})"/>'
        )
        elements.append(
            f'<circle cx="{x}" cy="{y}" r="10" fill="{color}"/>'
            f'<text x="{x}" y="{y + 4}" text-anchor="middle" '
            f'font-size="11" font-weight="700" fill="white">{robot_id}</text>'
        )
        goal_row, goal_col = agent["goals"][0]
        gx, gy = left + goal_col * cell, top + goal_row * cell
        elements.append(
            f'<rect x="{gx + 1.5}" y="{gy + 1.5}" width="21" height="21" '
            f'rx="3" fill="white" stroke="{color}" stroke-width="2"/>'
            f'<text x="{gx + 12}" y="{gy + 16}" text-anchor="middle" '
            f'font-size="10" font-weight="700" fill="{color}">G{robot_id}</text>'
        )

    for label, x, y in (("N", 396, 86), ("W", 52, 417), ("E", 733, 417)):
        elements.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" '
            f'font-size="18" font-weight="700">{label}</text>'
        )
    elements.extend([
        '<circle cx="58" cy="551" r="9" fill="#2563eb"/>',
        '<text x="77" y="556" font-size="13">Robot start / ID</text>',
        '<rect x="232" y="542" width="19" height="19" rx="3" '
        'fill="white" stroke="#2563eb" stroke-width="2"/>',
        '<text x="260" y="556" font-size="13">G = final parking goal</text>',
        '<rect x="473" y="542" width="19" height="19" '
        'fill="#fef3c7" stroke="#eab308"/>',
        '<text x="501" y="556" font-size="13">Conflict zone</text>',
        '<text x="48" y="587" font-size="12" fill="#475569">'
        'Arrows show intended travel direction. All robots initially face north.'
        '</text>',
        '<text x="48" y="607" font-size="12" fill="#475569">'
        '16 rows × 25 columns · three-cell-wide aisles · coordinates: [row, column]'
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
    for filename, document in (("three_way.map.json", map_document),
                               ("three_way.scenario.json", scenario)):
        path = args.output_dir / filename
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(path)
    svg_path = args.output_dir / "three_way.svg"
    svg_path.write_text(render_svg(map_document, scenario), encoding="utf-8")
    print(svg_path)


if __name__ == "__main__":
    main()
