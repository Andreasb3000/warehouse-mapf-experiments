"""Geometry checks guard against impossible or misleading junction cases."""

from collections import Counter, deque
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from experiments.junction.benchmark.generate import CASES, build_case, render_svg, write_case
from experiments.junction.scenario import load_scenario


def distance(start, goal, free):
    if start not in free or goal not in free:
        return None
    queue, visited = deque([(start, 0)]), {start}
    while queue:
        cell, length = queue.popleft()
        if cell == goal:
            return length
        row, col = cell
        for neighbor in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
            if neighbor in free and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, length + 1))
    return None


class GeneratedJunctionTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case["name"]: build_case(**case) for case in CASES}

    def cells(self, map_document):
        return {(row, col) for row, values in enumerate(map_document["layout"])
                for col, value in enumerate(values) if value == "."}

    def test_dimensions_boundaries_and_balanced_demand(self):
        for name, (map_document, scenario, config) in self.cases.items():
            with self.subTest(case=name):
                layout, agents = map_document["layout"], scenario["agents"]
                self.assertLess(max(map_document["n_row"], map_document["n_col"]), 70)
                self.assertEqual(len(layout), map_document["n_row"])
                self.assertTrue(all(len(row) == map_document["n_col"] for row in layout))
                self.assertTrue(all(value == "@" for value in layout[0] + layout[-1]))
                self.assertTrue(all(row[0] == row[-1] == "@" for row in layout))
                starts = {tuple(agent["start"]) for agent in agents}
                goals = {tuple(agent["goals"][0]) for agent in agents}
                self.assertEqual(len(starts), len(agents))
                self.assertEqual(len(goals), len(agents))
                self.assertFalse(starts & goals)
                self.assertTrue((starts | goals) <= self.cells(map_document))
                self.assertEqual(Counter(agent["approach"] for agent in agents),
                                 Counter(agent["destination_arm"] for agent in agents))
                self.assertTrue(all(agent["approach"] != agent["destination_arm"] for agent in agents))
                self.assertEqual(config["parameters"]["num_agents"], len(agents))

    def test_every_route_requires_center(self):
        for name, (map_document, scenario, _) in self.cases.items():
            free = self.cells(map_document)
            zone = scenario["conflict_zone"]
            core = {(row, col) for row in range(zone["rows"][0], zone["rows"][1] + 1)
                    for col in range(zone["cols"][0], zone["cols"][1] + 1)}
            for agent in scenario["agents"]:
                with self.subTest(case=name, robot=agent["id"]):
                    start, goal = tuple(agent["start"]), tuple(agent["goals"][0])
                    self.assertIsNotNone(distance(start, goal, free))
                    self.assertIsNone(distance(start, goal, free - core))
                    if scenario["benchmark"]["core_width_cells"] == 1:
                        self.assertIsNone(distance(start, goal, free - {tuple(scenario["module"]["anchor"])}))

    def test_private_goals_and_constructive_sequential_solution(self):
        for name, (map_document, scenario, _) in self.cases.items():
            free = self.cells(map_document)
            starts = {tuple(agent["start"]) for agent in scenario["agents"]}
            goals = {tuple(agent["goals"][0]) for agent in scenario["agents"]}
            for agent in scenario["agents"]:
                with self.subTest(case=name, robot=agent["id"]):
                    start, goal = tuple(agent["start"]), tuple(agent["goals"][0])
                    row, col = goal
                    neighbors = {(row - 1, col), (row + 1, col),
                                 (row, col - 1), (row, col + 1)}
                    self.assertEqual(len(neighbors & free), 1)
                    self.assertEqual(distance(start, goal, free - (goals - {goal})),
                                     distance(start, goal, free))
                    # Hold every other robot still, at either its start or its
                    # destination. A route still exists, so a one-at-a-time
                    # schedule can solve the finite instance in any robot order.
                    blocked = (starts - {start}) | (goals - {goal})
                    self.assertIsNotNone(distance(start, goal, free - blocked))

    def test_wide_narrow_pairs_change_only_center_geometry(self):
        for geometry in ("t", "four_way"):
            for count in (12, 24):
                with self.subTest(geometry=geometry, count=count):
                    wide_map, wide, _ = self.cases[f"{geometry}_wide_{count}"]
                    narrow_map, narrow, _ = self.cases[f"{geometry}_narrow_{count}"]
                    self.assertEqual(wide["agents"], narrow["agents"])
                    self.assertEqual(wide["module"], narrow["module"])
                    wide_free, narrow_free = self.cells(wide_map), self.cells(narrow_map)
                    self.assertLess(len(narrow_free), len(wide_free))
                    self.assertTrue(narrow_free <= wide_free)
                    zone = wide["conflict_zone"]
                    self.assertTrue(all(zone["rows"][0] <= row <= zone["rows"][1]
                                        and zone["cols"][0] <= col <= zone["cols"][1]
                                        for row, col in wide_free - narrow_free))

    def test_module_ports_are_three_free_cells_with_outward_walls(self):
        for name, (map_document, scenario, _) in self.cases.items():
            free = self.cells(map_document)
            for port in scenario["module"]["ports"]:
                with self.subTest(case=name, port=port["name"]):
                    self.assertEqual(len(port["cells"]), 3)
                    dr, dc = port["outward_normal"]
                    for row, col in port["cells"]:
                        self.assertIn((row, col), free)
                        self.assertNotIn((row + dr, col + dc), free)

    def test_saved_cases_load_in_launcher_and_svg_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            for case in CASES:
                with self.subTest(case=case["name"]):
                    directory = write_case(Path(temporary) / case["name"], case)
                    map_document = json.loads((directory / "map.json").read_text())
                    starts, tasks = load_scenario(directory / "scenario.json",
                                                  map_document["layout"], case["num_agents"])
                    self.assertEqual(len(starts), case["num_agents"])
                    self.assertEqual(len(tasks.splitlines()), case["num_agents"] + 1)
                    ET.fromstring((directory / "layout.svg").read_text())

    def test_rejects_unsupported_or_overcrowded_dimensions(self):
        for options in ({"geometry": "roundabout"}, {"num_agents": 13},
                        {"core_width": 2}, {"arm_length": 8}, {"num_agents": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_case(**options)


if __name__ == "__main__":
    unittest.main()
