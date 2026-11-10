#!/usr/bin/env python3
"""Render the executable map, not the separate proposed blueprint, as vector PDF."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT.parent / "simulator/experiments/zones/nine_zones"


def main():
    document = json.loads((CASE / "map.json").read_text())
    scenario = json.loads((CASE / "scenario.json").read_text())
    tasks = json.loads((CASE / "tasks.json").read_text())
    lines = [r"\documentclass[tikz,border=3pt]{standalone}",
             r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
             r"\usetikzlibrary{arrows.meta}", r"\begin{document}",
             r"\begin{tikzpicture}[font=\sffamily\small]",
             r"\definecolor{road}{HTML}{F8FAFC}",
             r"\definecolor{obstacle}{HTML}{CBD5E1}",
             r"\definecolor{robot}{HTML}{2563EB}",
             r"\definecolor{pickup}{HTML}{0F766E}",
             r"\definecolor{dropoff}{HTML}{C026D3}",
             r"\definecolor{safety}{HTML}{D97706}",
             r"\begin{scope}[x=0.132cm,y=-0.132cm]",
             r"\fill[obstacle] (-.5,-.5) rectangle (60.5,60.5);"]
    for r, row in enumerate(document["layout"]):
        # Contiguous white row runs produce a small, accurate planning-mask PDF.
        start = None
        for c, ch in enumerate(row + "@"):
            if ch == "." and start is None:
                start = c
            elif ch != "." and start is not None:
                lines.append(fr"\fill[road] ({start-.5},{r-.5}) rectangle ({c-.5},{r+.5});")
                start = None
    for cut in (19.5, 39.5):
        lines.extend([fr"\draw[gray!70,densely dashed,line width=.3pt] ({cut},-.5)--({cut},60.5);",
                      fr"\draw[gray!70,densely dashed,line width=.3pt] (-.5,{cut})--(60.5,{cut});"])
    for zone in scenario["zones"]:
        r, c = zone["center"]
        kind = {2: "corner", 3: "T", 4: "cross"}[zone["degree"]]
        lines.extend([fr"\fill[safety!30] ({c-1.5},{r-1.5}) rectangle ({c+1.5},{r+1.5});",
                      fr"\node[anchor=center,font=\sffamily\scriptsize] at ({c},{r-8.5}) {{{zone['id']} ({kind})}};"])
        for r, c in zone["bays"]["pickups"]:
            lines.append(fr"\fill[pickup] ({c-.35},{r-.35}) rectangle ({c+.35},{r+.35});")
        for r, c in zone["bays"]["dropoffs"]:
            lines.append(fr"\draw[dropoff,line width=.8pt] ({c-.35},{r-.35}) rectangle ({c+.35},{r+.35});")
    for agent in scenario["agents"]:
        r, c = agent["start"]
        lines.append(fr"\fill[robot] ({c},{r}) circle[radius=.38];")
    lines.extend([r"\draw[black!60,line width=.5pt] (-.5,-.5) rectangle (60.5,60.5);",
                  r"\draw[|-|,line width=.7pt] (3,57) -- (13,57);",
                  r"\node[font=\sffamily\scriptsize,above] at (8,56.5) {10 m};",
                  r"\draw[-{Latex[length=1.5mm]}] (56,57)--(56,52);",
                  r"\node[font=\sffamily\scriptsize,above] at (56,52) {N};",
                  r"\end{scope}",
                  r"\node[anchor=north west,text width=5.2cm,align=left,font=\sffamily\bfseries] at (8.35,0) {Executable reference map};",
                  r"\node[anchor=north west,text width=5.2cm,align=left,font=\sffamily\small] at (8.35,-.55) {61 $\times$ 61 cells; 1 m spacing.\\9 junctions, 12 aisle links.\\27 robots, three per zone.};",
                  r"\fill[robot] (8.55,-2.1) circle[radius=.055];",
                  r"\node[anchor=west] at (8.8,-2.1) {Initial robot position};",
                  r"\fill[pickup] (8.50,-2.55) rectangle (8.60,-2.45);",
                  r"\node[anchor=west] at (8.8,-2.5) {Pickup bay};",
                  r"\draw[dropoff,line width=.9pt] (8.50,-2.95) rectangle (8.60,-2.85);",
                  r"\node[anchor=west] at (8.8,-2.9) {Private dropoff / final parking};",
                  r"\fill[safety!30] (8.49,-3.35) rectangle (8.61,-3.23);",
                  r"\node[anchor=west] at (8.8,-3.3) {3 $\times$ 3 m junction region};",
                  r"\node[anchor=north west,text width=5.2cm,align=left,font=\sffamily\bfseries] at (8.35,-3.9) {Safety field (separate scale)};",
                  r"\begin{scope}[shift={(9.5,-6.75)},x=1.5cm,y=1.5cm]",
                  r"\filldraw[fill=safety!10,draw=safety,line width=.8pt] (-.3,-.2) rectangle (.3,1.2);",
                  r"\fill[robot] (0,0) circle[radius=.085];",
                  r"\draw[-{Latex[length=1.5mm]},robot] (0,.10)--(0,.65);",
                  r"\draw[<->] (.5,0)--(.5,1.2);",
                  r"\node[right,font=\sffamily\scriptsize] at (.5,.6) {front 1.2 m};",
                  r"\draw[<->] (-.3,-.4)--(.3,-.4);",
                  r"\node[below,font=\sffamily\scriptsize] at (0,-.4) {width 0.6 m};",
                  r"\node[anchor=west,font=\sffamily\scriptsize] at (.5,-.12) {back 0.2 m};",
                  r"\end{scope}",
                  r"\node[anchor=north west,text width=5.2cm,align=left,font=\sffamily\scriptsize] at (8.35,-7.65) {Robot diameter 0.17 m. Contact triggers a 20\% MOVE speed cap. Inner stopping is disabled. Map markers are enlarged for visibility.};",
                  r"\end{tikzpicture}", r"\end{document}"])
    figure = ROOT / "figures/map_overview.tex"
    figure.write_text("\n".join(lines) + "\n")
    result = subprocess.run(["pdflatex", "-halt-on-error", "-interaction=nonstopmode", figure.name],
                            cwd=figure.parent, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-5000:])
    metadata = {
        "source": str(CASE.relative_to(ROOT.parent)),
        "source_sha256": {name: hashlib.sha256((CASE/name).read_bytes()).hexdigest()
                          for name in ("map.json", "scenario.json", "tasks.json", "config.json")},
        "grid": [document["n_row"], document["n_col"]],
        "cell_spacing_m": scenario["cell_size_m"],
        "free_planning_cells": sum(row.count(".") for row in document["layout"]),
        "zones": len(scenario["zones"]),
        "junction_types": dict(collections.Counter(z["kind"] for z in scenario["zones"])),
        "undirected_links": len(scenario["network"]["edges"]),
        "directed_approaches": 2 * len(scenario["network"]["edges"]),
        "approach_length_m": 17,
        "aisle_width_free_planning_cells": 3,
        "approximate_wall_face_clearance_m": 3.2,
        "junction_spacing_m": 20,
        "robots": len(scenario["agents"]),
        "initial_counts": tasks["initial_robot_matrix"],
        "missions": len(tasks["tasks"]),
        "mission": "private start -> local pickup -> cross-zone private dropoff; park after delivery",
        "release_s": 0,
        "nominal_service_s_per_goal": 2,
        "payload_mass_kg_metadata_only": 5,
        "safety": scenario["safety"],
        "caveats": [
            "This executable nine-junction fixture differs from the separate unimplemented sixteen-junction blueprint.",
            "Map shows blocked planning cells; native boxes are 0.8m wide on a 1m grid.",
            "Robots are finite tasks and may park before evaluation ends; stationary agents can inflate aggregate accuracy.",
            "No load-dependent dynamics, restricted radio visibility, or closed-loop congestion rerouting in these recordings.",
        ],
    }
    (ROOT / "data/map_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(figure.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
