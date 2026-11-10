"""Validate explicit starts and finite goals before launching native processes."""

import json
from pathlib import Path


def load_scenario(filename, layout, num_agents):
    scenario_path = Path(filename)
    scenario = json.loads(scenario_path.read_text())
    source_map = json.loads((scenario_path.parent / scenario["map_file"]).read_text())
    if source_map["layout"] != layout:
        raise ValueError("Scenario geometry does not match the simulation map")
    if not layout or any(len(row) != len(layout[0]) for row in layout):
        raise ValueError("Map must be a nonempty rectangular grid")
    if source_map["n_row"] != len(layout) or source_map["n_col"] != len(layout[0]):
        raise ValueError("Map dimensions do not match its layout")
    if scenario.get("initial_orientation", "north") != "north":
        raise ValueError("Current controller/server initialization requires north-facing robots")
    agents = scenario["agents"]
    if len(agents) != num_agents or [agent["id"] for agent in agents] != list(range(num_agents)):
        raise ValueError("Scenario agent IDs must be ordered 0..num_agents-1")
    rows, cols = len(layout), len(layout[0])

    def cell(point):
        if not isinstance(point, list) or len(point) != 2 or any(type(v) is not int for v in point):
            raise ValueError("Coordinates must be [row, column] integer pairs")
        row, col = point
        if not (0 <= row < rows and 0 <= col < cols) or layout[row][col] in "@T":
            raise ValueError(f"Scenario point {point} is outside the free map")
        return row * cols + col

    starts = [cell(agent["start"]) for agent in agents]
    if len(set(starts)) != num_agents:
        raise ValueError("Robot starts must be unique")
    goals = []
    for agent in agents:
        if not agent.get("goals"):
            raise ValueError("Each robot needs at least one finite goal")
        goals.append([cell(goal) for goal in agent["goals"]])
    all_goals = [goal for sequence in goals for goal in sequence]
    if len(set(all_goals)) != len(all_goals):
        raise ValueError("Finite PBS destinations must be unique across all tasks")
    if any(agent.get("release_at_s", 0) != 0 for agent in agents):
        raise ValueError("This first fixture releases all robots at time zero")
    # ARGoS accepts (column, row); server task files use flattened grid cells.
    robot_starts = [(str(agent["start"][1]), str(agent["start"][0])) for agent in agents]
    task_rows = [str(num_agents)] + [";".join(f"{goal},-1,0,0" for goal in sequence) for sequence in goals]
    return robot_starts, "\n".join(task_rows) + "\n"
