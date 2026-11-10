"""Outcome-accounting cases that can reverse a gate-policy conclusion.

These tiny synthetic native logs test bookkeeping, not the motion model.
They intentionally include unfinished work, same-boundary completions,
fallback use, and gate calls that do not actually stop a robot.
"""
import json
from pathlib import Path
import tempfile
import unittest

from fleet_execution import SAFETY_DEFAULTS, write_json
from run_entry_gates import score


class EntryGateMetricsTest(unittest.TestCase):
    def fixture(self, root, completions=(8.0, None)):
        count, horizon = 2, 20
        stages = [
            [{'operation': 'pickup', 'location': [0, 0]},
             {'operation': 'dropoff', 'location': [4, 0]}],
            [{'operation': 'pickup', 'location': [0, 4]},
             {'operation': 'dropoff', 'location': [4, 4]}],
        ]
        scenario = {'agents': [{'id': i, 'goals': [s['location'] for s in stages[i]]}
                               for i in range(count)],
                    'safety': {'robot_trigger': 'outer_overlap', 'inner_stop_enabled': False}}
        tasks = [{'id': f'task-{i}', 'robot_id': i, 'release_at_s': 0,
                  'stages': stages[i]} for i in range(count)]
        config = {'parameters': {'num_agents': count, 'sim_duration': horizon * 10,
                                 'ticks_per_second': 10}}
        spec = {'run_id': 'synthetic-gate', 'num_agents': count,
                'workload_seed': 1, 'trigger': 'outer_overlap',
                'method': 'forecast', 'horizon_s': horizon}
        for name, value in [('scenario.json', scenario), ('tasks.json', {'tasks': tasks}),
                            ('config.json', config), ('run_spec.json', spec),
                            ('map.json', {'layout': ['.....'] * 5})]:
            write_json(root / name, value)
        events = [{'type': 'safety_config', 'robots': [
            {**SAFETY_DEFAULTS, 'agent_id': i, 'robot_trigger': 'outer_overlap',
             'inner_stop_enabled': False, 'enabled': True} for i in range(count)]}]
        for i in range(count):
            row, col = stages[i][0]['location']
            events.append({'type': 'goal_reached', 'clock': 0, 'agent': i,
                           'row': row, 'col': col})
        queries = []
        for clock in range(horizon * 10):
            if clock % 20 == 0:
                success = clock != 20
                events.extend([
                    {'type': 'routing_barrier_complete', 'clock': clock, 'wall_s': .25},
                    {'type': 'routing_plan', 'clock': clock, 'success': success,
                     'planner_wall_s': .2, 'nonzero_penalty_cells': 0},
                    {'type': 'plan_snapshot', 'clock': clock,
                     'planner_source': 'primary' if success else 'fallback',
                     'robots': []},
                ])
                queries.append({'clock': clock, 'origin_s': clock / 10,
                                'host_query_s': .05, 'reason': 'decision',
                                'held_robot_ids': [0, 1] if clock in (20, 40, 60) else []})
            # Control clock 40 first affects physical interval 41 -> 42.
            # One of its ten inferred idle ticks must fail the stationary check.
            events.append({'type': 'tick', 'clock': clock, 'agents': [
                {'id': 0, 'x': -1., 'y': -1., 'rz': 0.},
                {'id': 1, 'x': -2. - (.02 if clock >= 42 else 0), 'y': -2., 'rz': 0.},
            ]})
        for robot, completed in enumerate(completions):
            if completed is not None:
                row, col = stages[robot][1]['location']
                events.append({'type': 'goal_reached', 'clock': round(completed * 10),
                               'agent': robot, 'row': row, 'col': col})
        blocked = [
            (20, 0, 0, 0), (20, 0, 0, 0),  # duplicate RPC in one native tick
            (30, 1, 1, 0),                  # old queued action can execute
            (40, 1, 0, 0),                  # one tick has observable motion
            (50, 0, 0, 1),                  # earlier action in this batch dispatches
            (60, 0, 0, 0),
        ]
        events.extend({'type': 'entry_gate_blocked', 'clock': clock, 'robot_id': robot,
                       'queued_action_count': queued, 'dispatchable_before_gate': before}
                      for clock, robot, queued, before in blocked)
        (root / 'host_queries.jsonl').write_text(
            ''.join(json.dumps(query) + '\n' for query in queries))
        self.save(root, events)
        return events

    def save(self, root, events):
        (root / 'simulator.log').write_text(
            ''.join('native: ' + json.dumps(event) + '\n' for event in events))

    def test_unfinished_work_and_same_tick_barrier_stay_in_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = score(root)
            self.assertEqual(result['deliveries_completed'], 1)
            self.assertEqual(result['deliveries_unfinished'], 1)
            self.assertEqual(result['restricted_mean_flow_s'], 14.)
            # The completed task pays barriers at t=0,2,4,6,8, including its own
            # completion boundary. The unfinished task pays all ten barriers.
            self.assertAlmostEqual(result['task_outcomes'][0]['synchronous_query_pause_s'], 1.25)
            self.assertAlmostEqual(result['task_outcomes'][1]['synchronous_query_pause_s'], 2.5)
            self.assertAlmostEqual(result['charged_restricted_mean_flow_s'], 15.875)
            self.assertAlmostEqual(result['equal_charged_horizon_mean_flow_s'], 14.625)
            self.assertEqual(result['deliveries_by_equal_charged_horizon'], 1)
            self.assertEqual(result['primary_planner_failures'], 1)
            self.assertEqual(result['planning_calls'], 10)

    def test_equal_charged_horizon_censors_late_native_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, completions=(17.5, 19.8))
            result = score(root)
            self.assertEqual(result['deliveries_completed'], 2)
            self.assertEqual(result['deliveries_by_equal_charged_horizon'], 1)
            # 17.5 + 9*.25 = 19.75; 19.8 + 10*.25 = 22.3 -> censored at20.
            self.assertAlmostEqual(result['equal_charged_horizon_mean_flow_s'], 19.875)

    def test_requested_blocks_and_observed_stops_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = score(root)
            self.assertEqual(result['requested_hold_robot_s'], 12.)
            self.assertEqual(result['dispatch_block_events'], 5)
            self.assertEqual(result['empty_queue_block_events'], 3)
            self.assertAlmostEqual(result['empty_queue_gate_wait_robot_s'], 3.)
            self.assertAlmostEqual(result['verified_stationary_gate_wait_robot_s'], 2.9)
            self.assertEqual(result['blocked_robots'], 2)
            self.assertAlmostEqual(result['max_robot_gate_wait_s'], 2.)

    def test_last_poll_wait_is_clipped_and_unobserved_motion_not_assumed_stationary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = [e for e in self.fixture(root, completions=(None, None))
                      if e['type'] != 'entry_gate_blocked']
            events.append({'type': 'entry_gate_blocked', 'clock': 190, 'robot_id': 1,
                           'queued_action_count': 0, 'dispatchable_before_gate': 0})
            self.save(root, events)
            result = score(root)
            self.assertAlmostEqual(result['empty_queue_gate_wait_robot_s'], 1.)
            # Movement after control 198 and 199 would require unrecorded
            # physical poses200 and201. They cannot be verified stationary.
            self.assertAlmostEqual(result['verified_stationary_gate_wait_robot_s'], .8)

    def test_primary_failure_must_match_native_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self.fixture(root)
            next(e for e in events if e['type'] == 'plan_snapshot' and e['clock'] == 20)[
                'planner_source'] = 'primary'
            self.save(root, events)
            with self.assertRaisesRegex(ValueError, 'primary/fallback'):
                score(root)

    def test_missing_plan_boundary_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self.fixture(root)
            events = [e for e in events if not (e['type'] == 'routing_plan' and e['clock'] == 40)]
            self.save(root, events)
            with self.assertRaisesRegex(ValueError, 'missing routing_plan'):
                score(root)

    def test_duplicate_plan_record_is_not_collapsed_by_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self.fixture(root)
            events.append(dict(next(e for e in events
                                    if e['type'] == 'routing_plan' and e['clock'] == 40)))
            self.save(root, events)
            with self.assertRaisesRegex(ValueError, 'Duplicate or missing routing_plan'):
                score(root)

    def test_nonzero_routing_cost_invalidates_admission_only_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = self.fixture(root)
            next(e for e in events if e['type'] == 'routing_plan' and e['clock'] == 40)[
                'nonzero_penalty_cells'] = 1
            self.save(root, events)
            with self.assertRaisesRegex(ValueError, 'routing penalties'):
                score(root)


if __name__ == '__main__':
    unittest.main()
