"""Larger fleets must be real, nested, and independently feasible."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from fleet_cases import (CENTER, FLEET_SIZES, _shared_floor, distances,
                          make_fleet_workload, neighbors, validate_fleet_case)
from record_dataset import write_json
from experiments.junction.scenario import load_scenario
from experiments.zones.generate import build_case
from experiments.prediction.joint import _Safety, _fields


class NestedFleetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {n: make_fleet_workload(8301, fleet_size=n) for n in FLEET_SIZES}

    def test_real_fleet_sizes_and_balanced_sources(self):
        for size, (_, scenario, config, tasks, assignment) in self.cases.items():
            self.assertEqual(len(scenario['agents']), size)
            self.assertEqual(len(tasks['tasks']), size)
            self.assertEqual(config['parameters']['num_agents'], size)
            self.assertEqual([a['id'] for a in scenario['agents']], list(range(size)))
            self.assertEqual(Counter(a['zone_id'] for a in scenario['agents']),
                             Counter({f'Z{r}{c}': size // 9 for r in range(3) for c in range(3)}))
            self.assertEqual(assignment['fleet_size'], size)

    def test_one_common_floor_extends_only_private_leaf_bays(self):
        maps = [case[0] for case in self.cases.values()]
        self.assertTrue(all(json.dumps(m) == json.dumps(maps[0]) for m in maps))
        original = build_case(3)[0]
        old = {(r, c) for r, row in enumerate(original['layout']) for c, v in enumerate(row) if v == '.'}
        new = {(r, c) for r, row in enumerate(maps[0]['layout']) for c, v in enumerate(row) if v == '.'}
        old_leaves = {p for p in old if sum(n in old for n in neighbors(p)) == 1}
        new_leaves = {p for p in new if sum(n in new for n in neighbors(p)) == 1}
        self.assertEqual(len(new), 881)
        self.assertEqual(len(new_leaves), 192)
        self.assertEqual(len(new - old), 111)
        self.assertEqual(old - old_leaves, new - new_leaves)
        self.assertTrue(old <= new)
        self.assertEqual((maps[0]['n_row'], maps[0]['n_col']), (61, 61))

    def test_small_fleets_are_exact_prefixes_of_larger_fleet_records(self):
        full = self.cases[81]
        for size, case in self.cases.items():
            self.assertEqual(case[1]['agents'], full[1]['agents'][:size])
            self.assertEqual(case[3]['tasks'], full[3]['tasks'][:size])
            self.assertEqual(case[4]['assignments'], full[4]['assignments'][:size])

    def test_all_goals_are_private_unique_and_other_robots_do_not_block_routes(self):
        free = _shared_floor()[4]
        for size, case in self.cases.items():
            agents = case[1]['agents']
            starts = {tuple(a['start']) for a in agents}
            goals = {tuple(g) for a in agents for g in a['goals']}
            self.assertEqual(len(starts), size)
            self.assertEqual(len(goals), 2 * size)
            self.assertFalse(starts & goals)
            self.assertTrue(all(sum(n in free for n in neighbors(g)) == 1 for g in goals))
            for agent in agents:
                own = {tuple(agent['start']), *(tuple(g) for g in agent['goals'])}
                self.assertTrue(own <= distances(tuple(agent['start']), free - ((starts | goals) - own)).keys())

    def test_initial_fields_are_clear_under_both_unchanged_safety_rules(self):
        original_safety = build_case(3)[1]['safety']
        for case in self.cases.values():
            self.assertEqual(case[1]['safety'], original_safety)
            self.assertEqual(case[1]['initial_orientation'], 'north')
            self.assertEqual({a['release_at_s'] for a in case[1]['agents']}, {0})
            for trigger in ('body_in_outer', 'outer_overlap'):
                scenario = deepcopy(case[1]); scenario['safety']['robot_trigger'] = trigger
                states = {a['id']: {'row': a['start'][0], 'col': a['start'][1], 'yaw': 0.0} for a in scenario['agents']}
                response = _Safety(_fields(scenario, list(states), True), case[0]).evaluate(states)
                self.assertTrue(all(mode == 'clear' and scale == 1 and not reasons for mode, scale, reasons in response.values()))

    def test_center_preferences_do_not_remove_bypass_routes(self):
        free = _shared_floor()[4]
        for record in self.cases[81][4]['assignments']:
            pickup, goal = tuple(record['pickup']), tuple(record['dropoff'])
            self.assertNotEqual(record['source_zone'], record['destination_zone'])
            self.assertIn(goal, distances(pickup, free - CENTER))
            self.assertGreaterEqual(record['center_preference_gap_m'], 0)
            self.assertEqual(record['center_preference_gap_m'],
                             record['center_avoiding_distance_m'] - record['shortest_delivery_distance_m'])
        self.assertEqual(self.cases[81][4]['strictly_center_preferring_tasks'], 81)

    def test_tasks_allocation_and_native_loader_match_actual_fleet(self):
        for size, case in self.cases.items():
            _, scenario, _, tasks, assignment = case
            matrix = [[0] * 9 for _ in range(9)]
            zi = {z: i for i, z in enumerate(tasks['zone_order'])}
            for agent, task in zip(scenario['agents'], tasks['tasks']):
                self.assertEqual(agent['id'], task['robot_id'])
                self.assertEqual(agent['task_id'], task['id'])
                self.assertEqual(agent['payload_id'], task['payload_id'])
                self.assertEqual(agent['goals'], [s['location'] for s in task['stages']])
                matrix[zi[agent['zone_id']]][zi[agent['destination_zone_id']]] += 1
            self.assertEqual(matrix, assignment['allocation_matrix'])
            self.assertEqual(matrix, tasks['allocation_matrix'])
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for filename, value in zip(('map.json', 'scenario.json', 'config.json', 'tasks.json', 'assignment.json'), case):
                    write_json(directory / filename, value)
                starts, rows = load_scenario(directory / 'scenario.json', case[0]['layout'], size)
                self.assertEqual(len(starts), size)
                self.assertEqual(len(rows.splitlines()), size + 1)
                self.assertTrue(all(len(row.split(';')) == 2 for row in rows.splitlines()[1:]))

    def test_reproducible_and_returned_data_cannot_mutate_cached_template(self):
        fresh = make_fleet_workload(8301, fleet_size=27)
        self.assertEqual(fresh, self.cases[27])
        fresh[1]['agents'][0]['start'][0] = -100
        self.assertEqual(make_fleet_workload(8301, fleet_size=27), self.cases[27])
        self.assertNotEqual(make_fleet_workload(8302, fleet_size=27)[4]['assignments'], self.cases[27][4]['assignments'])

    def test_unsupported_or_inconsistent_fleet_data_are_rejected(self):
        for size in (0, 28, 108, True):
            with self.assertRaises(ValueError):
                make_fleet_workload(8301, fleet_size=size)
        for arguments in ({'trigger': 'bad'}, {'duration_s': 0}, {'duration_s': float('nan')}):
            with self.assertRaises(ValueError):
                make_fleet_workload(8301, **arguments)
        bad = deepcopy(self.cases[54]); bad[2]['parameters']['num_agents'] = 27
        with self.assertRaises(ValueError):
            validate_fleet_case(*bad)
        bad = deepcopy(self.cases[54]); bad[3]['tasks'].pop()
        with self.assertRaises(ValueError):
            validate_fleet_case(*bad)


if __name__ == '__main__':
    unittest.main()
