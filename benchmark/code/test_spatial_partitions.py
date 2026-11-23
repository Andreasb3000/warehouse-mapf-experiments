"""Regression checks for spatial region aggregation."""
import unittest
import numpy as np
from spatial_partitions import partition_labels, aggregate_spatial_heat
from fleet_cases import make_fleet_workload
from route_search_policy import forecast_cost_menu, road_mask


class SpatialPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_doc, cls.scenario, *_ = make_fleet_workload(8401, 40, 27, 'outer_overlap')
        cls.road = road_mask(cls.map_doc)

    def test_partition_counts_and_full_road_coverage(self):
        for partition, expected in [('regions9', 9), ('regions36', 36), ('regions81', 33), ('regions144', 84)]:
            with self.subTest(partition=partition):
                labels = partition_labels(self.road, self.scenario, partition)
                self.assertEqual(len(np.unique(labels[self.road])), expected)
                self.assertTrue((labels[self.road] >= 0).all())
                self.assertTrue((labels[~self.road] == -1).all())

    def test_average_only_traversable_cells_and_no_added_mass(self):
        heat = np.zeros(self.road.shape)
        rr, cc = np.argwhere(self.road)[0]
        heat[rr, cc] = 2.0
        for partition in ['regions9', 'regions36', 'regions81', 'regions144']:
            with self.subTest(partition=partition):
                out = aggregate_spatial_heat(heat, self.road, self.scenario, partition)
                self.assertAlmostEqual(out.sum(), 2.0, places=10)
                self.assertEqual(out[~self.road].sum(), 0.)

    def test_original_cell_mode_bit_identical(self):
        fields = np.zeros((3, 2, *self.road.shape))
        fields[:,1,30,30] = [1., 2., 3.]
        old = forecast_cost_menu(fields, self.road)
        new = forecast_cost_menu(fields, self.road, 'cells', self.scenario)
        self.assertEqual(old, new)
        self.assertEqual(len(forecast_cost_menu(fields, self.road, 'regions9', self.scenario)), 3)
        self.assertNotEqual(new[1]['cell_penalties'],
                            forecast_cost_menu(fields, self.road, 'regions9', self.scenario)[1]['cell_penalties'])


if __name__ == '__main__':
    unittest.main()
