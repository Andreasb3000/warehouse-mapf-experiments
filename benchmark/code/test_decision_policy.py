"""The common routing adapter must not silently reward risky forecast fields."""
import unittest
import numpy as np
from decision_policy import cell_penalties, forecast, POLICY


class RoutingCostTest(unittest.TestCase):
    def setUp(self):
        self.road = np.ones((7, 7), dtype=bool)
        self.prediction = np.zeros((3, 2, 7, 7), dtype=np.float32)

    def test_zero_prediction_is_exactly_unpenalized_route_cost(self):
        self.assertTrue(np.array_equal(cell_penalties(self.prediction, self.road), np.zeros((7, 7))))

    def test_only_slow_occupancy_changes_cost_not_total_occupancy(self):
        self.prediction[:, 0] = 27
        self.assertFalse(cell_penalties(self.prediction, self.road).any())

    def test_window_weights_and_neighborhood_are_common_and_exact(self):
        self.prediction[:, 1, 3, 3] = [1, 2, 3]
        result = cell_penalties(self.prediction, self.road)
        expected = np.zeros((7, 7))
        expected[2:5, 2:5] = 8 * (.5 + .3 * 2 + .2 * 3)
        np.testing.assert_allclose(result, expected)

    def test_obstacles_never_receive_cost_and_edges_do_not_wrap(self):
        self.road[0, 1] = False
        self.prediction[:, 1, 0, 0] = 1
        result = cell_penalties(self.prediction, self.road)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(result[-1, -1], 0)
        self.assertEqual(result[1, 1], 8)

    def test_negative_nan_or_incorrect_shape_is_rejected(self):
        for value in (-1, np.nan):
            bad = self.prediction.copy()
            bad[0, 1, 3, 3] = value
            with self.assertRaises(ValueError):
                cell_penalties(bad, self.road)
        with self.assertRaises(ValueError):
            cell_penalties(self.prediction[:2], self.road)

    def test_cell_cost_cap_limits_extreme_forecasts(self):
        self.prediction[:, 1, 3, 3] = 10000
        result = cell_penalties(self.prediction, self.road)
        self.assertEqual(result.max(), POLICY["max_cell_penalty"])

    def test_zero_predictor_needs_no_future_or_planner_state(self):
        static = np.ones((3, 7, 7))
        result = forecast({}, None, {}, {}, static, "zero")
        self.assertEqual(result.shape, (3, 2, 7, 7))
        self.assertFalse(result.any())


if __name__ == "__main__":
    unittest.main()
