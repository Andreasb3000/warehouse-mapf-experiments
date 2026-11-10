import unittest
import numpy as np
from event_metrics import confusion, score_windows, sum_zones, zone_masks


class HotspotMetricTest(unittest.TestCase):
    def test_never_alerting_has_zero_recall_and_undefined_precision(self):
        result = confusion(np.array([False, False, False]), np.array([True, False, True]))
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (0, 0, 2, 1))
        self.assertEqual(result["recall"], 0)
        self.assertIsNone(result["precision"])

    def test_current_clear_cohort_does_not_reward_repeating_existing_hotspots(self):
        predicted = np.array([[[2., 0.], [2., 0.], [2., 0.]]])
        actual = np.ones((1, 3, 2))
        result = score_windows(predicted, actual, np.array([[False, True]]))[1]
        self.assertEqual(result["all_zones"]["tp"], 1)
        self.assertEqual(result["currently_clear_zones"]["tp"], 0)
        self.assertEqual(result["currently_clear_zones"]["fn"], 1)

    def test_zone_sums_conserve_robot_occupancy_and_exclude_obstacles(self):
        road = np.ones((2, 4), dtype=bool)
        road[0, 3] = False
        scenario = {"zones": [{"id": "left", "kind": "test", "bounds": {"rows": [0, 1], "cols": [0, 1]}},
                              {"id": "right", "kind": "test", "bounds": {"rows": [0, 1], "cols": [2, 3]}}]}
        _, masks = zone_masks(scenario, road)
        field = np.ones((2, 4))
        field[0, 3] = 100
        np.testing.assert_equal(sum_zones(field, masks), [4, 3])

    def test_overlapping_zones_are_rejected(self):
        road = np.ones((2, 2), dtype=bool)
        zone = {"id": "z", "kind": "test", "bounds": {"rows": [0, 1], "cols": [0, 1]}}
        with self.assertRaises(ValueError):
            zone_masks({"zones": [zone, zone]}, road)


if __name__ == "__main__":
    unittest.main()
