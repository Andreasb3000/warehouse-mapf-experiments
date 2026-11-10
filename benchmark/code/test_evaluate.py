"""Small cases showing what spatial and aggregate forecast errors measure."""
import unittest

import numpy as np

from evaluate import measure


class EvaluationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.road = np.ones((2, 2), dtype=bool)
        self.roi = np.array([[True, True], [False, False]])
        self.truth = np.zeros((1, 3, 2, 2, 2), dtype=float)
        self.prediction = np.zeros_like(self.truth)

    def test_displacement_penalizes_fixed_masks_despite_total_cancellation(self):
        # One slow robot is forecast in the wrong cell, with correct total mass.
        self.truth[:, :, :, 0, 0] = 1
        self.prediction[:, :, :, 0, 1] = 1
        rows = measure(self.prediction, self.truth, self.road, self.roi)
        for row in rows:
            for channel in ("occupancy", "slow_occupancy"):
                with self.subTest(window=row["window_s"], channel=channel):
                    scores = row["channels"][channel]
                    self.assertEqual(scores["regions"]["roi"]["mae"], 1.)
                    self.assertEqual(scores["regions"]["roi"]["rmse"], 1.)
                    self.assertEqual(scores["regions"]["roads"]["mae"], .5)
                    self.assertAlmostEqual(scores["regions"]["roads"]["rmse"], np.sqrt(.5))
                    self.assertEqual(scores["regions"]["positive_roi"]["cells"], 1)
                    self.assertEqual(scores["regions"]["positive_roi"]["mae"], 1.)
                    self.assertEqual(scores["total_robot_seconds_mae"], 0.)

    def test_larger_false_alarm_changes_fixed_score_not_positive_only_score(self):
        self.truth[:, :, :, 0, 0] = 1
        self.prediction[:, :, :, 0, 1] = 2
        rows = measure(self.prediction, self.truth, self.road, self.roi)
        for row in rows:
            scores = row["channels"]["slow_occupancy"]
            self.assertEqual(scores["regions"]["roi"]["mae"], 1.5)
            self.assertEqual(scores["regions"]["positive_roi"]["mae"], 1.)
            self.assertEqual(scores["regions"]["positive_roi"]["cells"], 1)
            duration = row["window_s"][1] - row["window_s"][0]
            self.assertEqual(scores["total_robot_seconds_mae"], duration)

    def test_empty_positive_subset_is_unavailable_while_false_alarms_are_scored(self):
        self.prediction[:, :, :, 0, 1] = 1
        rows = measure(self.prediction, self.truth, self.road, self.roi)
        for row in rows:
            for channel in ("occupancy", "slow_occupancy"):
                scores = row["channels"][channel]
                for subset in ("positive", "positive_roi"):
                    region = scores["regions"][subset]
                    self.assertEqual(region["cells"], 0)
                    for key in ("mae", "rmse", "p95_absolute", "bias"):
                        self.assertIsNone(region[key])
                self.assertEqual(scores["regions"]["roads"]["mae"], .25)
                self.assertEqual(scores["regions"]["roi"]["mae"], .5)


if __name__ == "__main__":
    unittest.main()
