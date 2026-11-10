"""Check that the saved fixture forces a crossing without blocking exits."""

from collections import Counter, deque
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .scenario import load_scenario
except ImportError:
    from scenario import load_scenario


ROOT = Path(__file__).parent


def shortest_distance(start, goal, free_cells):
    if start not in free_cells or goal not in free_cells:
        return None
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == goal:
            return distance
        for neighbor in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
            if neighbor in free_cells and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


class ThreeWayScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = json.loads((ROOT / "three_way.scenario.json").read_text())
        cls.map = json.loads((ROOT / cls.scenario["map_file"]).read_text())
        cls.agents = cls.scenario["agents"]
        cls.free = {(row, col)
                    for row, cells in enumerate(cls.map["layout"])
                    for col, value in enumerate(cells) if value == "."}
        zone = cls.scenario["conflict_zone"]
        cls.conflict = {(row, col)
                        for row in range(zone["rows"][0], zone["rows"][1] + 1)
                        for col in range(zone["cols"][0], zone["cols"][1] + 1)}

    def test_unique_free_starts_and_finite_goals(self):
        self.assertEqual(len(self.agents), 6)
        self.assertEqual({agent["id"] for agent in self.agents}, set(range(6)))
        self.assertEqual(Counter(agent["approach"] for agent in self.agents),
                         {"west": 2, "east": 2, "north": 2})
        self.assertTrue(all(len(agent["goals"]) == 1 for agent in self.agents))
        starts = {tuple(agent["start"]) for agent in self.agents}
        goals = {tuple(agent["goals"][0]) for agent in self.agents}
        self.assertEqual(len(starts), 6)
        self.assertEqual(len(goals), 6)
        self.assertTrue((starts | goals) <= self.free)
        self.assertFalse(starts & goals)
        self.assertFalse((starts | goals) & self.conflict)
        self.assertEqual(self.scenario["task_mode"], "finite")
        self.assertEqual(self.scenario["initial_orientation"], "north")

    def test_map_dimensions_and_boundary(self):
        layout = self.map["layout"]
        self.assertEqual(len(layout), self.map["n_row"])
        self.assertTrue(all(len(row) == self.map["n_col"] for row in layout))
        self.assertTrue(all(value == "@" for value in layout[0] + layout[-1]))
        self.assertTrue(all(row[0] == row[-1] == "@" for row in layout))
        self.assertTrue(self.conflict <= self.free)

    def test_every_route_must_cross_conflict_zone(self):
        for agent in self.agents:
            with self.subTest(robot=agent["id"]):
                start, goal = tuple(agent["start"]), tuple(agent["goals"][0])
                self.assertIsNotNone(shortest_distance(start, goal, self.free))
                self.assertIsNone(shortest_distance(start, goal,
                                                    self.free - self.conflict))

    def test_junction_has_exactly_three_connected_arms(self):
        outside = self.free - self.conflict
        components = []
        for cell in sorted(outside):
            if all(shortest_distance(cell, representative, outside) is None
                   for representative in components):
                components.append(cell)
        self.assertEqual(len(components), 3)
        self.assertEqual(set(self.scenario["approach_zones"]),
                         {"west", "east", "north"})

    def test_parked_robots_cannot_obstruct_remaining_routes(self):
        goals = {tuple(agent["goals"][0]) for agent in self.agents}
        for agent in self.agents:
            with self.subTest(robot=agent["id"]):
                start, goal = tuple(agent["start"]), tuple(agent["goals"][0])
                original = shortest_distance(start, goal, self.free)
                occupied = goals - {goal}
                self.assertEqual(shortest_distance(start, goal,
                                                   self.free - occupied), original)
                row, col = goal
                neighbors = {(row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)}
                self.assertEqual(len(neighbors & self.free), 1,
                                 "Each destination should be a private side bay")

    def test_initial_approach_regions_match_robot_labels(self):
        for agent in self.agents:
            with self.subTest(robot=agent["id"]):
                row, col = agent["start"]
                zone = self.scenario["approach_zones"][agent["approach"]]
                self.assertLessEqual(zone["rows"][0], row)
                self.assertLessEqual(row, zone["rows"][1])
                self.assertLessEqual(zone["cols"][0], col)
                self.assertLessEqual(col, zone["cols"][1])

    def test_launcher_converts_grid_coordinates_consistently(self):
        starts, task_text = load_scenario(ROOT / "three_way.scenario.json",
                                          self.map["layout"], 6)
        self.assertEqual(starts[0], ("6", "13"))
        self.assertEqual(starts[4], ("11", "6"))
        self.assertEqual(task_text.splitlines()[0], "6")
        self.assertEqual(task_text.splitlines()[1], "368,-1,0,0")
        self.assertEqual(len(task_text.splitlines()), 7)

    def test_launcher_rejects_invalid_or_unsupported_scenarios(self):
        cases = [
            ("robot_count", lambda scenario: None, 5, "agent IDs"),
            ("obstacle_start",
             lambda scenario: scenario["agents"][0].update(start=[0, 0]),
             6, "outside the free map"),
            ("duplicate_start",
             lambda scenario: scenario["agents"][1].update(start=[13, 6]),
             6, "starts must be unique"),
            ("delayed_release",
             lambda scenario: scenario["agents"][0].update(release_at_s=5),
             6, "time zero"),
            ("unsupported_orientation",
             lambda scenario: scenario.update(initial_orientation="east"),
             6, "north-facing"),
            ("duplicate_parking",
             lambda scenario: scenario["agents"][1].update(goals=[[14, 18]]),
             6, "destinations must be unique"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / self.scenario["map_file"]).write_text(json.dumps(self.map))
            fixture_path = directory / "scenario.json"
            for name, mutate, count, message in cases:
                with self.subTest(case=name):
                    scenario = deepcopy(self.scenario)
                    mutate(scenario)
                    fixture_path.write_text(json.dumps(scenario))
                    with self.assertRaisesRegex(ValueError, message):
                        load_scenario(fixture_path, self.map["layout"], count)


if __name__ == "__main__":
    unittest.main()
