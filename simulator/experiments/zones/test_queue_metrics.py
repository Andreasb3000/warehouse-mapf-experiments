import math
import unittest

from experiments.zones.queue_metrics import build_approaches, measure_approach, summarize_queues


class ApproachQueueTests(unittest.TestCase):
    def setUp(self):
        scenario = {"zones": [{"id": name, "center": [10, col],
                               "conflict_zone": {"rows": [9, 11], "cols": [col-1, col+1]}}
                              for name, col in (("A", 10), ("B", 30))],
                    "network": {"edges": [["A", "B"]]}}
        self.approaches = build_approaches(scenario)
        self.approach = self.approaches[0]

    def pose(self, robot_id, distance, *, row=10, heading=-math.pi/2):
        return {"id": robot_id, "x": -row, "y": -(28.5-distance), "rz": heading}

    def measure(self, poses, eligible=None):
        return measure_approach(self.approach, poses,
                                {pose["id"] for pose in poses} if eligible is None else eligible,
                                {pose["id"]: .085 for pose in poses})

    def test_approach_boundaries_are_seventeen_by_three_meters(self):
        self.assertEqual(len(self.approaches), 2)
        self.assertEqual(self.approach["length_m"], 17)
        self.assertEqual(self.approach["width_m"], 3)
        self.assertEqual(self.approach["upstream_exit_row_col"], [10, 11.5])
        self.assertEqual(self.approach["downstream_entry_row_col"], [10, 28.5])
        self.assertEqual(self.approaches[1]["unit_row_col"], [0, -1])
        self.assertEqual(self.measure([self.pose(0, 1, row=8.5)])["queued_robots"], 1)
        self.assertEqual(self.measure([self.pose(0, 1, row=11.5)])["queued_robots"], 0)
        self.assertEqual(self.measure([self.pose(0, 0)])["queued_robots"], 0)

    def test_reversed_heading_and_ineligible_service_or_completed_robots_excluded(self):
        poses = [self.pose(0, .5, heading=math.pi/2), self.pose(1, 1.0), self.pose(2, 1.5)]
        result = self.measure(poses, eligible={0, 2})
        self.assertEqual(result["queued_robot_ids"], [2])
        self.assertEqual(self.measure(poses, eligible=set())["queued_robots"], 0)

    def test_heading_transform_works_for_vertical_approach(self):
        scenario = {"zones": [{"id": name, "center": [row, 10],
                               "conflict_zone": {"rows": [row-1, row+1], "cols": [9, 11]}}
                              for name, row in (("A", 10), ("B", 30))], "network": {"edges": [["A", "B"]]}}
        south = build_approaches(scenario)[0]
        poses = [{"id": 0, "x": -27.5, "y": -10, "rz": math.pi},
                 {"id": 1, "x": -27.0, "y": -10, "rz": 0}]
        self.assertEqual(measure_approach(south, poses, {0, 1}, {})["queued_robot_ids"], [0])

    def test_far_upstream_isolated_robot_does_not_form_a_queue(self):
        self.assertEqual(self.measure([self.pose(0, 7)])["queued_robots"], 0)

    def test_disconnected_groups_are_not_merged(self):
        poses = [self.pose(index, distance) for index, distance in enumerate((1, 3, 5.01, 6))]
        result = self.measure(poses)
        self.assertEqual(result["queued_robot_ids"], [0, 1])
        self.assertAlmostEqual(result["queue_tail_extent_m"], 3.085)
        self.assertAlmostEqual(result["queue_body_span_m"], 2.17)

    def test_queue_spillback_warning_needs_a_connected_tail_near_upstream_exit(self):
        long_queue = self.measure([self.pose(i, .5+2*i) for i in range(9)])
        short_queue = self.measure([self.pose(i, 1+2*i) for i in range(8)])
        self.assertTrue(long_queue["spillback_warning"])
        self.assertAlmostEqual(long_queue["tail_to_upstream_exit_m"], .415)
        self.assertFalse(short_queue["spillback_warning"])
        samples = [{"clock": 0, **long_queue}, {"clock": 1, **short_queue}]
        summary = summarize_queues(samples, [self.approach], .1)
        self.assertAlmostEqual(summary["queued_robot_seconds"], 1.7)
        self.assertEqual(summary["max_queue_robots"], 9)
        self.assertAlmostEqual(summary["spillback_warning_approach_seconds"], .1)
        self.assertEqual(summary["max_simultaneous_spillback_warning_approaches"], 1)


if __name__ == "__main__":
    unittest.main()
