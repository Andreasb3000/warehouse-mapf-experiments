import json
from pathlib import Path
import tempfile
import unittest

from metrics import measure, shortest_distance


class DifficultyTests(unittest.TestCase):
    def test_distance_respects_walls(self):
        self.assertEqual(shortest_distance([".....", ".@@@.", "....."], [1, 0], [1, 4]), 6)
        with self.assertRaises(ValueError):
            shortest_distance([".@."], [0, 0], [0, 2])

    def test_unfinished_trips_remain_in_ratios_and_turning_is_not_queueing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run-01"
            run.mkdir()
            docs = {
                "config.json": {"scenario": "scenario.json", "parameters": {"ticks_per_second": 1, "sim_duration": 10, "velocity": 100}},
                "map.json": {"layout": ["........."]},
                "scenario.json": {"name": "test", "map_file": "map.json", "conflict_zone": {"rows": [0, 0], "cols": [4, 4]}, "agents": [
                    {"id": 0, "approach": "west", "start": [0, 0], "goals": [[0, 8]]},
                    {"id": 1, "approach": "east", "start": [0, 8], "goals": [[0, 0]]}]},
            }
            for filename, value in docs.items():
                (root / filename).write_text(json.dumps(value))
            (run / "junction_metrics.json").write_text(json.dumps({
                "valid": True, "robots": [{"parked_from_s": None, "first_entry_s": None}] * 2,
                "all_parked_by_s": None, "peak_junction_occupancy": 0,
                "baseline": {"n_mapf_calls": 4, "n_rule_based_calls": 2}, "wall_runtime_s": 1,
            }))
            frames = [{"clock": t, "agents": [
                {"id": 0, "x": 0, "y": 0, "rz": 0},
                {"id": 1, "x": 0, "y": -8, "rz": t * .2}]} for t in range(10)]
            (run / "trajectories.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
            result = measure(run)
            self.assertEqual(result["completion_fraction"], 0)
            self.assertIsNone(result["clearance_time_s"])
            self.assertTrue(result["unfinished_ratios_are_lower_bounds"])
            self.assertEqual(result["p90_observed_time_ratio"], 9 / 8)
            self.assertEqual(result["fallback_fraction"], .5)
            self.assertGreater(result["approaches"]["west"]["stationary_robot_seconds"], 0)
            self.assertEqual(result["approaches"]["east"]["stationary_robot_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
