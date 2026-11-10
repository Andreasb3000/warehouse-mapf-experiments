"""Guard network topology, task balance, and feasibility of the zone fixture."""

from collections import Counter, deque
import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from experiments.junction.scenario import load_scenario
from experiments.zones.generate import build_case, write_case


def neighbors(cell):
    row, col = cell
    return {(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)}


def reachable(start, free):
    visited, queue = {start}, deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in neighbors(cell) & free - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


class ZoneGenerationTests(unittest.TestCase):
    def setUp(self):
        self.map, self.scenario, self.config, self.tasks = build_case()
        self.free = {(row, col) for row, values in enumerate(self.map["layout"])
                     for col, value in enumerate(values) if value == "."}

    def test_compact_connected_network_and_correct_junction_degrees(self):
        layout = self.map["layout"]
        self.assertEqual((len(layout), len(layout[0])), (61, 61))
        self.assertTrue(all(len(row) == 61 for row in layout))
        self.assertTrue(all(cell == "@" for cell in layout[0] + layout[-1]))
        self.assertTrue(all(row[0] == row[-1] == "@" for row in layout))
        self.assertEqual(reachable(next(iter(self.free)), self.free), self.free)
        zones = self.scenario["zones"]
        self.assertEqual(Counter(zone["degree"] for zone in zones), {2: 4, 3: 4, 4: 1})
        self.assertEqual(len(self.scenario["network"]["edges"]), 12)
        self.assertEqual([zone["id"] for zone in zones],
                         [f"Z{row}{col}" for row in range(3) for col in range(3)])
        for zone in zones:
            self.assertEqual(zone["degree"], len(zone["neighbors"]))

    def test_private_bays_and_constructive_sequential_solution(self):
        agents = self.scenario["agents"]
        starts = {tuple(agent["start"]) for agent in agents}
        goals = {tuple(goal) for agent in agents for goal in agent["goals"]}
        self.assertEqual(len(starts), 27)
        self.assertEqual(len(goals), 54)
        self.assertFalse(starts & goals)
        self.assertTrue((starts | goals) <= self.free)
        for cell in starts | goals:
            with self.subTest(bay=cell):
                self.assertEqual(len(neighbors(cell) & self.free), 1)
        # Every route remains available when all other robots occupy either
        # starts or stations. A one-at-a-time solution therefore exists.
        for agent in agents:
            own = {tuple(agent["start"]), *(tuple(goal) for goal in agent["goals"])}
            blocked = (starts | goals) - own
            available = self.free - blocked
            with self.subTest(robot=agent["id"]):
                self.assertTrue(own <= reachable(tuple(agent["start"]), available))

    def test_matrix_and_task_manifest_match_two_stage_agent_goals(self):
        agents = self.scenario["agents"]
        self.assertEqual(len(agents), 27)
        self.assertEqual(self.tasks["initial_robot_matrix"], [[3, 3, 3]] * 3)
        self.assertEqual(len(self.tasks["tasks"]), 27)
        self.assertEqual(Counter(agent["zone_id"] for agent in agents),
                         Counter(agent["destination_zone_id"] for agent in agents))
        self.assertEqual(len({agent["payload_id"] for agent in agents}), 27)
        matrix = self.tasks["allocation_matrix"]
        self.assertTrue(all(sum(row) == 3 for row in matrix))
        self.assertTrue(all(sum(row[col] for row in matrix) == 3 for col in range(9)))
        self.assertTrue(all(matrix[index][index] == 0 for index in range(9)))
        zones = {zone["id"]: zone for zone in self.scenario["zones"]}
        for agent, task in zip(agents, self.tasks["tasks"]):
            self.assertEqual(agent["id"], task["robot_id"])
            self.assertEqual(agent["task_id"], task["id"])
            self.assertEqual(agent["payload_id"], task["payload_id"])
            self.assertEqual([stage["location"] for stage in task["stages"]], agent["goals"])
            self.assertEqual([stage["operation"] for stage in task["stages"]], ["pickup", "dropoff"])
            self.assertIn(agent["start"], zones[agent["zone_id"]]["bays"]["starts"])
            self.assertIn(agent["goals"][0], zones[agent["zone_id"]]["bays"]["pickups"])
            self.assertIn(agent["goals"][1], zones[agent["destination_zone_id"]]["bays"]["dropoffs"])
            self.assertEqual(agent["release_at_s"], 0)

    def test_zone_bounds_are_disjoint_and_ports_connect_neighboring_zones(self):
        zones = {zone["id"]: zone for zone in self.scenario["zones"]}
        for zone in zones.values():
            for port in zone["ports"]:
                dr, dc = port["outward_normal"]
                opposite = next(other for other in zones[port["neighbor"]]["ports"]
                                if other["neighbor"] == zone["id"])
                self.assertEqual({(row + dr, col + dc) for row, col in port["cells"]},
                                 {tuple(cell) for cell in opposite["cells"]})
                self.assertTrue({tuple(cell) for cell in port["cells"]} <= self.free)
        for cell in self.free:
            owners = [zone["id"] for zone in zones.values()
                      if zone["bounds"]["rows"][0] <= cell[0] <= zone["bounds"]["rows"][1]
                      and zone["bounds"]["cols"][0] <= cell[1] <= zone["bounds"]["cols"][1]]
            self.assertEqual(len(owners), 1)

    def test_saved_fixtures_load_and_svg_and_matrix_are_parseable(self):
        for per_zone in (2, 3):
            with self.subTest(robots_per_zone=per_zone), tempfile.TemporaryDirectory() as temporary:
                directory = write_case(Path(temporary), per_zone)
                map_document = json.loads((directory / "map.json").read_text())
                starts, tasks = load_scenario(directory / "scenario.json", map_document["layout"], 9 * per_zone)
                self.assertEqual(len(starts), 9 * per_zone)
                self.assertEqual(len(tasks.splitlines()), 1 + 9 * per_zone)
                self.assertTrue(all(len(line.split(";")) == 2 for line in tasks.splitlines()[1:]))
                ET.fromstring((directory / "layout.svg").read_text())
                with (directory / "zone_allocation.csv").open() as handle:
                    rows = list(csv.reader(handle))
                self.assertEqual(len(rows), 10)
                self.assertTrue(all(len(row) == 10 for row in rows))

    def test_rejects_unsupported_counts(self):
        for count in (0, 1, 4, -1, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_case(count)

    def test_outer_only_default_retains_optional_nested_stop_geometry(self):
        safety = self.scenario["safety"]
        self.assertTrue(safety["enabled"])
        self.assertEqual(safety["robot_trigger"], "body_in_outer")
        self.assertIs(safety["inner_stop_enabled"], False)
        self.assertGreater(safety["speed_scale"], 0)
        self.assertLess(safety["speed_scale"], 1)
        for suffix in ("front_m", "back_m", "half_width_m"):
            self.assertGreater(safety[f"stop_{suffix}"], 0)
            self.assertLess(safety[f"stop_{suffix}"], safety[f"slow_{suffix}"])
        self.assertGreater(safety["robot_radius_m"], 0)


if __name__ == "__main__":
    unittest.main()
