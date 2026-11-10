"""Bounded upstream holds respect direct clearers and retained commitments."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from entry_gate_policy import EntryGatePolicy, POLICY


def move(node, start, goal, dependencies=()):
    return {'type': 'M', 'node_id': node, 'start': list(start), 'goal': list(goal),
            'orientation': 0, 'dependencies': [list(d) for d in dependencies]}


def fixture(clock=120):
    positions = [(30, 30), (26, 30), (30, 26)]
    paths = [[move(1, positions[0], (34, 30))],
             [move(1, positions[1], (27, 30)), move(2, (27, 30), (34, 30))],
             [move(1, positions[2], (30, 27)), move(2, (30, 27), (30, 34))]]
    pose = {'clock': clock + 1, 'agents': [dict(id=i, x=-r, y=-c, rz=0.)
                                         for i, (r, c) in enumerate(positions)]}
    event = {'clock': clock, 'observation_clock': clock + 1, 'pose_frame': pose,
             'snapshot': {'clock': clock + 1, 'server_clock': clock,
                          'robots': [dict(id=i, finished_node_id=0, committed_node_id=0,
                                          enqueued_node_id=0, actions=path)
                                     for i, path in enumerate(paths)]},
             'controllers': [dict(id=i, clock=clock + 1) for i in range(3)]}
    past = np.tile(np.array([[r, c, 0] for r, c in positions], dtype=float), (21, 1, 1))
    scenario = {'agents': [dict(id=i, goals=[[2, i + 2]]) for i in range(3)]}
    map_doc = {'layout': ['.' * 61 for _ in range(61)]}
    return event, past, scenario, map_doc


def advance(event, clock):
    event = copy.deepcopy(event)
    event['clock'] = clock
    event['observation_clock'] = clock + 1
    event['pose_frame']['clock'] = clock + 1
    event['snapshot'].update(clock=clock + 1, server_clock=clock)
    for controller in event['controllers']:
        controller['clock'] = clock + 1
    return event


def stationary_forecast(past):
    return {'frames': [dict(offset_s=t / 10, agents=[dict(id=i, row=float(r), col=float(c), yaw=float(y))
                                                   for i, (r, c, y) in enumerate(past[-1])])
                       for t in range(51)]}


class EntryGateTest(unittest.TestCase):
    def test_no_gate_needs_no_history_or_forecast(self):
        event, _, _, _ = fixture()
        with patch('entry_gate_policy.rollout', side_effect=AssertionError('not needed')):
            payload, diagnostics = EntryGatePolicy().decide(event, None, {}, {}, 'none')
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(payload['until_clock'], 140)
        self.assertIsNone(diagnostics['pressure_robots'])

    def test_slow_interior_receives_priority_over_more_candidates(self):
        args = fixture()
        payload, diagnostics = EntryGatePolicy().decide(*args, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [1, 2])
        self.assertEqual(diagnostics['admitted_robot_ids'], [])
        self.assertEqual(diagnostics['protected_robot_ids'], [0])
        self.assertEqual(diagnostics['pressure_robots'], 1)
        json.dumps(diagnostics, allow_nan=False)

    def test_no_pressure_releases_all_candidates(self):
        event, past, scenario, map_doc = fixture()
        past[:, 0, 0] = np.linspace(28, 30, 21)
        payload, diagnostics = EntryGatePolicy().decide(event, past, scenario, map_doc, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(diagnostics['pressure_robots'], 0)

    def test_forecast_can_anticipate_pressure_before_current_slow_detector(self):
        event, past, scenario, map_doc = fixture()
        past[:, 0, 0] = np.linspace(28, 30, 21)
        with patch('entry_gate_policy.rollout', return_value=stationary_forecast(past)) as mocked:
            payload, diagnostics = EntryGatePolicy().decide(event, past, scenario, map_doc, 'forecast')
        self.assertEqual(diagnostics['current_pressure_robots'], 0)
        self.assertGreaterEqual(diagnostics['pressure_robots'], .5)
        self.assertEqual(payload['held_robot_ids'], [1, 2])
        self.assertIs(mocked.call_args.kwargs['controllers'], event['controllers'])
        self.assertEqual(mocked.call_args.kwargs['horizon_s'], 5)
        self.assertEqual(mocked.call_args.kwargs['dt'], .1)

    def test_identical_pressure_produces_same_gate_for_both_information_sources(self):
        args = fixture()
        reactive, _ = EntryGatePolicy().decide(*args, 'reactive')
        with patch('entry_gate_policy.rollout', return_value=stationary_forecast(args[1])):
            forecast, _ = EntryGatePolicy().decide(*args, 'forecast')
        self.assertEqual(reactive, forecast)

    def test_queued_or_committed_entry_is_never_held(self):
        for cut_name in ('committed_node_id', 'enqueued_node_id'):
            with self.subTest(cut_name=cut_name):
                event, past, scenario, map_doc = fixture()
                event['snapshot']['robots'][2][cut_name] = 1
                payload, diagnostics = EntryGatePolicy().decide(event, past, scenario, map_doc, 'reactive')
                self.assertEqual(payload['held_robot_ids'], [1])
                self.assertIn(2, diagnostics['committed_entry_robot_ids'])

    def test_strict_mode_releases_transitive_prerequisites_of_clearer(self):
        event, past, scenario, map_doc = fixture()
        event['snapshot']['robots'][0]['actions'][0]['dependencies'] = [[1, 2]]
        # Dependency on node 2 also requires node 1 of that robot.
        event['snapshot']['robots'][1]['actions'][0]['dependencies'] = [[2, 1]]
        payload, diagnostics = EntryGatePolicy(protect_uncommitted_clearance=True).decide(
            event, past, scenario, map_doc, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(diagnostics['protected_robot_ids'], [0, 1, 2])

    def test_future_dependencies_after_clearer_exit_do_not_disable_gate(self):
        event, past, scenario, map_doc = fixture()
        event['snapshot']['robots'][0]['actions'].append(move(2, (34, 30), (35, 30), [[2, 1]]))
        payload, _ = EntryGatePolicy(protect_uncommitted_clearance=True).decide(
            event, past, scenario, map_doc, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [1, 2])

    def test_strict_mode_unknown_unfinished_dependency_fails_open(self):
        event, past, scenario, map_doc = fixture()
        event['snapshot']['robots'][0]['actions'][0]['dependencies'] = [[2, 99]]
        payload, diagnostics = EntryGatePolicy(protect_uncommitted_clearance=True).decide(
            event, past, scenario, map_doc, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(diagnostics['unknown_dependencies'], [[2, 99]])

    def test_six_seconds_of_holding_force_one_interval_of_release(self):
        event, past, scenario, map_doc = fixture()
        policy = EntryGatePolicy()
        held = []
        diagnostics = None
        for clock in (120, 140, 160, 180, 200):
            payload, diagnostics = policy.decide(advance(event, clock), past, scenario, map_doc, 'reactive')
            held.append(payload['held_robot_ids'])
            if clock == 180:
                self.assertEqual(diagnostics['forced_release_robot_ids'], [1, 2])
        self.assertEqual(held, [[1, 2], [1, 2], [1, 2], [], [1, 2]])

    def test_gate_queue_outside_does_not_become_pressure_due_to_cell_rounding(self):
        event, past, scenario, map_doc = fixture()
        event['pose_frame']['agents'][0].update(x=-20, y=-20)
        event['snapshot']['robots'][0]['actions'] = []
        past[:, 0, :2] = [20, 20]
        event['pose_frame']['agents'][1]['x'] = -26.5
        past[:, 1, 0] = 26.5  # Existing raster convention maps this to row 27.
        payload, diagnostics = EntryGatePolicy().decide(event, past, scenario, map_doc, 'reactive')
        self.assertEqual(diagnostics['pressure_robots'], 0)
        self.assertEqual(payload['held_robot_ids'], [])

    def test_current_pose_at_service_is_excluded_by_shared_slow_detector(self):
        event, past, scenario, map_doc = fixture()
        scenario['agents'][0]['goals'] = [[30, 30]]
        payload, diagnostics = EntryGatePolicy().decide(event, past, scenario, map_doc, 'reactive')
        self.assertEqual(diagnostics['pressure_robots'], 0)
        self.assertEqual(payload['held_robot_ids'], [])

    def test_oldest_candidate_priority_changes_when_a_new_candidate_arrives(self):
        event, past, scenario, map_doc = fixture()
        # Empty now, but a future interior slowdown is predicted. The gate must
        # admit one candidate instead of freezing everyone ahead of that event.
        event['pose_frame']['agents'][0].update(x=-20, y=-20)
        event['snapshot']['robots'][0]['actions'] = []
        past[:, 0, :2] = [20, 20]
        future = stationary_forecast(past)
        for frame in future['frames'][1:]:
            frame['agents'][0].update(row=30., col=30.)
        policy = EntryGatePolicy()
        first = copy.deepcopy(event)
        first['snapshot']['robots'][1]['actions'] = []
        with patch('entry_gate_policy.rollout', return_value=future):
            policy.decide(first, past, scenario, map_doc, 'forecast')
            payload, diagnostics = policy.decide(advance(event, 140), past, scenario, map_doc, 'forecast')
        self.assertEqual(payload['held_robot_ids'], [1])
        self.assertEqual(diagnostics['admitted_robot_ids'], [2])

    def test_candidate_requires_nearby_accepted_entry(self):
        event, past, scenario, map_doc = fixture()
        event['snapshot']['robots'][2]['actions'] = [move(1, (30, 26), (30, 25))]
        payload, diagnostics = EntryGatePolicy().decide(event, past, scenario, map_doc, 'reactive')
        self.assertEqual(diagnostics['candidate_ids'], [1])
        self.assertEqual(payload['held_robot_ids'], [1])

    def test_invalid_history_clock_or_method_rejected(self):
        args = fixture()
        with self.assertRaises(ValueError):
            EntryGatePolicy().decide(*args, 'unknown')
        with self.assertRaises(ValueError):
            EntryGatePolicy().decide(args[0], args[1][:-1], args[2], args[3], 'reactive')
        policy = EntryGatePolicy()
        policy.decide(*args, 'reactive')
        with self.assertRaises(ValueError):
            policy.decide(*args, 'reactive')
        with self.assertRaises(ValueError):
            policy.decide(advance(args[0], 140), *args[1:], 'forecast')

    def test_constants_are_provisional_and_frozen_in_export(self):
        self.assertFalse(POLICY['parameters_validated'])
        self.assertEqual(POLICY['interval_ticks'], 20)
        self.assertEqual(POLICY['version'], 'central-entry-gate-v3')
        self.assertFalse(POLICY['protect_uncommitted_clearance'])

    def test_bounded_mode_can_hold_prerequisite_of_uncommitted_future_exit(self):
        event, past, scenario, map_doc = fixture()
        event['snapshot']['robots'][0]['actions'][0]['dependencies'] = [[2, 1]]
        payload, diagnostics = EntryGatePolicy().decide(event, past, scenario, map_doc, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [1, 2])
        self.assertEqual(diagnostics['protected_robot_ids'], [0])
        self.assertFalse(payload['protect_clearance'])

    def test_strict_mode_is_explicit_in_native_payload_and_diagnostics(self):
        payload, diagnostics = EntryGatePolicy(protect_uncommitted_clearance=True).decide(*fixture(), 'reactive')
        self.assertTrue(payload['protect_clearance'])
        self.assertTrue(diagnostics['protect_uncommitted_clearance'])
        with self.assertRaises(ValueError):
            EntryGatePolicy(protect_uncommitted_clearance='false')

    def test_commitment_promoted_after_a_hold_clears_that_requested_hold(self):
        event, past, scenario, map_doc = fixture()
        policy = EntryGatePolicy()
        policy.decide(event, past, scenario, map_doc, 'reactive')
        updated = advance(event, 140)
        updated['snapshot']['robots'][2]['committed_node_id'] = 1
        payload, _ = policy.decide(updated, past, scenario, map_doc, 'reactive')
        self.assertNotIn(2, payload['held_robot_ids'])
        self.assertNotIn(2, policy.held_since)

    def test_queue_promotion_after_a_hold_clears_that_requested_hold(self):
        event, past, scenario, map_doc = fixture()
        policy = EntryGatePolicy()
        policy.decide(event, past, scenario, map_doc, 'reactive')
        updated = advance(event, 140)
        updated['snapshot']['robots'][2]['enqueued_node_id'] = 1
        payload, _ = policy.decide(updated, past, scenario, map_doc, 'reactive')
        self.assertNotIn(2, payload['held_robot_ids'])
        self.assertNotIn(2, policy.held_since)

    def test_robot_that_enters_square_is_never_held_directly(self):
        event, past, scenario, map_doc = fixture()
        policy = EntryGatePolicy()
        policy.decide(event, past, scenario, map_doc, 'reactive')
        updated = advance(event, 140)
        updated['pose_frame']['agents'][2]['y'] = -27.5
        past[:, 2, 1] = 27.5
        payload, diagnostics = policy.decide(updated, past, scenario, map_doc, 'reactive')
        self.assertNotIn(2, payload['held_robot_ids'])
        self.assertIn(2, diagnostics['inside_robot_ids'])

    def test_withdrawn_entry_removes_hold_and_candidate_age(self):
        event, past, scenario, map_doc = fixture()
        policy = EntryGatePolicy()
        policy.decide(event, past, scenario, map_doc, 'reactive')
        updated = advance(event, 140)
        updated['snapshot']['robots'][2]['actions'] = []
        payload, _ = policy.decide(updated, past, scenario, map_doc, 'reactive')
        self.assertNotIn(2, payload['held_robot_ids'])
        self.assertNotIn(2, policy.held_since)
        self.assertNotIn(2, policy.first_candidate_clock)

    def test_missing_refresh_breaks_continuous_requested_holding(self):
        event, past, scenario, map_doc = fixture()
        policy = EntryGatePolicy()
        policy.decide(event, past, scenario, map_doc, 'reactive')
        policy.decide(advance(event, 140), past, scenario, map_doc, 'reactive')
        # No decision at 160: the previous request expires then.
        payload, diagnostics = policy.decide(advance(event, 180), past, scenario, map_doc, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [1, 2])
        self.assertEqual(diagnostics['forced_release_robot_ids'], [])
        self.assertEqual(diagnostics['continuous_hold_s'], {'1': 0., '2': 0.})

    def test_loss_of_pressure_breaks_continuous_requested_holding(self):
        event, past, scenario, map_doc = fixture()
        policy = EntryGatePolicy()
        policy.decide(event, past, scenario, map_doc, 'reactive')
        moving = past.copy()
        moving[:, 0, 0] = np.linspace(28, 30, 21)
        released, _ = policy.decide(advance(event, 140), moving, scenario, map_doc, 'reactive')
        self.assertEqual(released['held_robot_ids'], [])
        payload, diagnostics = policy.decide(advance(event, 160), past, scenario, map_doc, 'reactive')
        self.assertEqual(payload['held_robot_ids'], [1, 2])
        self.assertEqual(diagnostics['continuous_hold_s'], {'1': 0., '2': 0.})


if __name__ == '__main__':
    unittest.main()
