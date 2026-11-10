"""Stress demand must preserve geometry, safety initialization, and escape routes."""
from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from congestion_cases import (ARMS, CENTER, _geometry, _initial_safety,
                              distances, make_congestion_workload, neighbors)
from record_dataset import make_workload, write_json
from experiments.junction.scenario import load_scenario
from experiments.zones.generate import build_case


class ConcentratedDemandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {(load, burst): make_congestion_workload(8201, center_load=load, burst=burst)
                     for load in (12, 24) for burst in ('compact', 'spread')}

    def test_floor_map_and_safety_dimensions_are_unchanged(self):
        original_map, original_scenario, _, _ = build_case(3)
        for case in self.cases.values():
            self.assertEqual(json.dumps(case[0]), json.dumps(original_map))
            self.assertEqual(case[1]['safety'], original_scenario['safety'])
            self.assertEqual(case[1]['initial_orientation'], 'north')
            self.assertEqual({a['release_at_s'] for a in case[1]['agents']}, {0})

    def test_exact_demand_per_arm_with_no_mandatory_center_waypoint(self):
        free, _, _, _, _ = _geometry()
        for (load, _), case in self.cases.items():
            records = case[4]['assignments']
            active = [r for r in records if r['role'] == 'center_crossing']
            self.assertEqual(len(active), load)
            self.assertEqual(Counter(r['approach'] for r in active), Counter({a: load // 4 for a in ARMS}))
            for record in active:
                pickup, dropoff = record['pickup'], record['dropoff']
                shortest = distances(pickup, free)[dropoff]
                bypass = distances(pickup, free - CENTER)[dropoff]
                self.assertGreater(bypass, shortest)
                self.assertEqual(record['center_preference_gap_m'], bypass - shortest)
            self.assertTrue(all(len(a['goals']) == 2 for a in case[1]['agents']))

    def test_all_terminals_are_unique_private_bays_and_starts_do_not_block_them(self):
        free, _, _, _, _ = _geometry()
        for case in self.cases.values():
            agents = case[1]['agents']
            starts = {tuple(a['start']) for a in agents}
            goals = {tuple(g) for a in agents for g in a['goals']}
            self.assertEqual(len(starts), 27)
            self.assertEqual(len(goals), 54)
            self.assertFalse(starts & goals)
            self.assertTrue(all(sum(n in free for n in neighbors(g)) == 1 for g in goals))
            for agent in agents:
                own = {tuple(agent['start']), *(tuple(g) for g in agent['goals'])}
                self.assertTrue(own <= distances(tuple(agent['start']), free - ((starts | goals) - own)).keys())

    def test_neither_safety_rule_is_triggered_at_initialization(self):
        for case in self.cases.values():
            starts = [tuple(a['start']) for a in case[1]['agents']]
            self.assertTrue(all(math.dist(a, b) >= 2 for i, a in enumerate(starts) for b in starts[i + 1:]))
            for trigger in ('body_in_outer', 'outer_overlap'):
                decisions = _initial_safety(case[1], case[0], starts, trigger)
                self.assertTrue(all(mode == 'clear' and scale == 1 and not reasons
                                    for mode, scale, reasons in decisions.values()))

    def test_compact_and_spread_change_staging_not_delivery_assignments(self):
        for load in (12, 24):
            compact = self.cases[(load, 'compact')][4]['assignments']
            spread = self.cases[(load, 'spread')][4]['assignments']
            self.assertEqual([(r['pickup'], r['dropoff']) for r in compact],
                             [(r['pickup'], r['dropoff']) for r in spread])
            a = [r['nominal_center_arrival_s'] for r in compact[:load]]
            b = [r['nominal_center_arrival_s'] for r in spread[:load]]
            self.assertTrue(all(v > 12 for v in a + b))
            self.assertLess(np.std(a), np.std(b))
            self.assertLessEqual(max(abs(v - 26) for v in a), 3)

    def test_task_manifest_and_allocations_match_actual_agents(self):
        for case in self.cases.values():
            _, scenario, _, tasks, assignment = case
            zone_index = {z: i for i, z in enumerate(tasks['zone_order'])}
            allocation = [[0] * 9 for _ in range(9)]
            populations = Counter()
            for agent, task in zip(scenario['agents'], tasks['tasks']):
                self.assertEqual(agent['goals'], [s['location'] for s in task['stages']])
                self.assertEqual(agent['id'], task['robot_id'])
                populations[agent['zone_id']] += 1
                allocation[zone_index[agent['zone_id']]][zone_index[agent['destination_zone_id']]] += 1
            self.assertEqual(allocation, assignment['allocation_matrix'])
            self.assertEqual(allocation, tasks['allocation_matrix'])
            self.assertEqual({z: populations[z] for z in tasks['zone_order']}, tasks['initial_robot_counts'])

    def test_deterministic_reproducible_and_load_zero_preserves_original_baseline(self):
        self.assertEqual(make_congestion_workload(8201, center_load=12), self.cases[(12, 'compact')])
        self.assertEqual(make_congestion_workload(8201, center_load=0), make_workload(8201, 240))
        self.assertNotEqual(make_congestion_workload(8202, center_load=12)[4]['assignments'],
                            self.cases[(12, 'compact')][4]['assignments'])

    def test_saved_fixture_passes_native_scenario_loader(self):
        case = self.cases[(24, 'compact')]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename, value in zip(('map.json', 'scenario.json', 'config.json', 'tasks.json', 'assignment.json'), case):
                write_json(directory / filename, value)
            starts, rows = load_scenario(directory / 'scenario.json', case[0]['layout'], 27)
            self.assertEqual(len(starts), 27)
            self.assertEqual(len(rows.splitlines()), 28)
            self.assertTrue(all(len(row.split(';')) == 2 for row in rows.splitlines()[1:]))

    def test_invalid_or_unsupported_parameters_are_rejected(self):
        for load in (-1, 1, 13, 28, True):
            with self.assertRaises(ValueError):
                make_congestion_workload(8201, center_load=load)
        for arguments in ({'burst': 'delayed'}, {'trigger': 'invalid'}, {'duration_s': 0},
                          {'nominal_arrival_s': 10}, {'nominal_arrival_s': float('nan')}):
            with self.assertRaises(ValueError):
                make_congestion_workload(8201, **arguments)


if __name__ == '__main__':
    unittest.main()
