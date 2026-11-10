import math
import unittest
from experiments.prediction.flow_metrics import approaches_for, crossing_events, queue_samples


class FlowMetricsTests(unittest.TestCase):
    def test_censored_visit_and_completed_visit(self):
        zones = [{"id": "j", "conflict_zone": {"rows": [4, 4], "cols": [4, 4]}}]
        frames = [{"offset_s": t, "agents": [{"id": 0, "row": row, "col": 4, "yaw": 0}]}
                  for t, row in enumerate([5, 4, 3])]
        visit = crossing_events(frames, zones)[(0, "j")]
        self.assertEqual((visit["entry_s"], visit["exit_s"]), (1, 2))
        self.assertIsNone(crossing_events(frames[:2], zones)[(0, "j")]["exit_s"])

    def test_slow_following_group_counts_but_service_and_turn_do_not(self):
        scenario = {"agents": [{"id": i, "goals": [[1, 1]]} for i in range(2)],
                    "zones": [{"id": name, "center": [10, col],
                               "conflict_zone": {"rows": [9, 11], "cols": [col-1, col+1]}}
                              for name, col in [("a", 10), ("b", 30)]],
                    "network": {"edges": [["a", "b"]]}}
        approaches = approaches_for(scenario, [])
        frames = [{"offset_s": k*.1, "agents": [
            {"id": i, "row": 10, "col": 27-i+.02*k, "yaw": -math.pi/2}
            for i in range(2)]} for k in range(21)]
        measured = queue_samples(frames, approaches, scenario)[-1]
        self.assertEqual(measured["max_queue"], 2)
        self.assertTrue(measured["multi_robot_queue"])
        for frame in frames:
            frame["agents"][0]["yaw"] += frame["offset_s"]
        self.assertLess(queue_samples(frames, approaches, scenario)[-1]["max_queue"], 2)

    def test_single_junction_approaches_skip_shelf_directions(self):
        scenario = {"conflict_zone": {"rows": [2, 2], "cols": [2, 2]}}
        approaches = approaches_for(scenario, ["@@@@@", "@@.@@", "@...@", "@@@@@"])
        self.assertEqual(len(approaches), 3)
        self.assertTrue(all(a["length_m"] == 1 for a in approaches))


if __name__ == "__main__":
    unittest.main()
