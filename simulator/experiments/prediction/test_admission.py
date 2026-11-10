"""Native-style bounded admission changes dispatch, never execution history."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

try:
    from .admission import approaches, contains, enters
    from .joint import rollout
    from .test_joint import scenario, controller, sample
    from .test_predict import action, robot, snapshot, frame
except ImportError:
    from admission import approaches, contains, enters
    from joint import rollout
    from test_joint import scenario, controller, sample
    from test_predict import action, robot, snapshot, frame


def gate(ids=(0,), until=1., **kwargs):
    return dict(center=[4., 7.], half_width=1., held_robot_ids=list(ids),
                until_offset_s=until, protect_clearance=False, **kwargs)


def run(state, poses, controls, admission, horizon=3, settings=None):
    return rollout(state, poses, settings or scenario(enabled=False), {}, controllers=controls,
                   horizon_s=horizon, admission=admission)


class AdmissionGeometryTests(unittest.TestCase):
    def test_native_closed_square_and_outside_crossing(self):
        self.assertTrue(contains([4, 6], [4, 7], 1))
        self.assertTrue(enters([4, 5], [4, 9], [4, 7], 1))
        self.assertTrue(approaches([4, 5], [4, 9], [4, 7], 1))
        self.assertFalse(approaches([4, 6], [4, 5], [4, 7], 1))
        self.assertFalse(approaches([4, 7], [4, 8], [4, 7], 1))
        self.assertTrue(approaches([2, 5], [3, 5], [4, 7], 1))  # Distance decreases.
        self.assertFalse(approaches([3, 5], [4, 5], [4, 7], 1))  # Parallel outside edge.
        self.assertFalse(approaches([4, 5], [4, 4], [4, 7], 1))


class AdmissionRolloutTests(unittest.TestCase):
    def test_expiry_at_poll_releases_without_completing_the_action(self):
        state = snapshot([robot(0, [action(7, [4, 4], [4, 5])])])
        poses, controls = frame(0, [(4, 4)]), [controller(0)]
        result = run(state, poses, controls, gate(), horizon=1.2)
        self.assertEqual(sample(result, .9)['action_type'], 'WAIT')
        self.assertEqual(sample(result, 1)['action_type'], 'M')
        self.assertEqual(sample(result, 1)['col'], 4)
        self.assertGreater(sample(result, 1.1)['col'], 4)
        self.assertEqual(result['completions'], [])
        detail = result['diagnostics']['admission']
        self.assertEqual(detail['dispatch_block_events'], 1)
        self.assertEqual(detail['empty_queue_wait_robot_seconds'], 1.)
        self.assertEqual(detail['events'][0]['node_id'], 7)

    def test_nonzero_poll_phase_and_no_velocity_override(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 5])])])
        result = run(state, frame(0, [(4, 4)]), [controller(0, poll_phase=5)], gate(), horizon=1.6)
        self.assertEqual(result['diagnostics']['admission']['events'][0]['offset_s'], .5)
        self.assertEqual(sample(result, 1.4)['action_type'], 'WAIT')
        self.assertEqual(sample(result, 1.5)['action_type'], 'M')
        self.assertGreater(sample(result, 1.6)['col'], 4)

    def test_existing_queue_moves_and_new_prefetch_is_denied(self):
        known = [action(0, [4, 4], [4, 4.5]), action(1, [4, 4.5], [4, 5])]
        state = snapshot([robot(0, known, enqueued=0)])
        queue = [{'type': 'M', 'node_ids': [0], 'goal': [4, 4.5], 'orientation_rad': 0,
                  'remaining_ticks': -1}]
        result = run(state, frame(0, [(4, 4)]),
                     [controller(0, queue, speed_m_s=.4, previous_move_speed_m_s=.4)], gate(until=2), horizon=.2)
        self.assertGreater(sample(result, .1)['col'], 4)
        event = result['diagnostics']['admission']['events'][0]
        self.assertEqual((event['node_id'], event['queued_action_count'], event['dispatchable_before_gate']), (1, 1, 0))
        self.assertEqual(result['diagnostics']['admission']['empty_queue_wait_robot_seconds'], 0.)

    def test_committed_action_dispatches_before_gate_without_fake_empty_wait(self):
        known = [action(0, [4, 4], [4, 4.5]), action(1, [4, 4.5], [4, 5])]
        state = snapshot([robot(0, known, committed=0)])
        result = run(state, frame(0, [(4, 4)]), [controller(0)], gate(until=2), horizon=.2)
        self.assertGreater(sample(result, .1)['col'], 4)
        event = result['diagnostics']['admission']['events'][0]
        self.assertEqual((event['node_id'], event['queued_action_count'], event['dispatchable_before_gate']), (1, 0, 1))
        self.assertEqual(result['diagnostics']['admission']['empty_queue_wait_robot_seconds'], 0.)

    def test_transitive_committed_prerequisite_is_never_held(self):
        state = snapshot([
            robot(0, [action(0, [4, 4], [4, 4.5])]),
            robot(1, [action(0, [10, 10], [10, 10.5], dependencies=[[0, 0]]),
                      action(1, [10, 10.5], [10, 11])]),
            robot(2, [action(0, [15, 15], [15, 16], dependencies=[[1, 1]])], committed=0),
        ])
        poses, controls = frame(0, [(4, 4), (10, 10), (15, 15)]), [controller(i) for i in range(3)]
        result = run(state, poses, controls, gate(until=2))
        baseline = run(state, poses, controls, None)
        self.assertEqual(result['frames'], baseline['frames'])
        self.assertEqual(result['completions'], baseline['completions'])
        self.assertGreater(result['diagnostics']['admission']['commitment_prerequisite_bypasses'], 0)
        self.assertEqual(result['diagnostics']['admission']['dispatch_block_events'], 0)

    def test_uncommitted_dependent_waits_for_real_completion_after_expiry(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 4.5])]),
                          robot(1, [action(0, [4, 7], [4, 9], dependencies=[[0, 0]])])])
        poses, controls = frame(0, [(4, 4), (4, 7)]), [controller(i) for i in range(2)]
        result = run(state, poses, controls, gate(), horizon=4)
        self.assertEqual(sample(result, 1, 1)['action_type'], 'WAIT')
        completion = next(c['offset_s'] for c in result['completions'] if c['id'] == 0)
        self.assertGreater(completion, 1)
        self.assertTrue(all(a['col'] == 7 for f in result['frames'] if f['offset_s'] < completion
                            for a in f['agents'] if a['id'] == 1))
        self.assertGreater(sample(result, 4, 1)['col'], 7)

    def test_unselected_inside_away_and_turn_actions_remain_available(self):
        state = snapshot([robot(0, [action(0, [4, 7], [4, 9])]),
                          robot(1, [action(0, [4, 5], [4, 4])]),
                          robot(2, [action(0, [10, 10], [10, 10], kind='T', orientation=1)]),
                          robot(3, [action(0, [4, 4], [4, 5])])])
        poses = frame(0, [(4, 7), (4, 5), (10, 10), (4, 4)])
        result = run(state, poses, [controller(i) for i in range(4)], gate(ids=(0, 1, 2)))
        self.assertEqual(result['diagnostics']['admission']['dispatch_block_events'], 0)
        self.assertEqual(result['frames'], run(state, poses, [controller(i) for i in range(4)], None)['frames'])

    def test_holding_changes_downstream_safety_without_disabling_it(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 7])]), robot(1, [])])
        poses, controls = frame(0, [(4, 4), (4, 6)]), [controller(i) for i in range(2)]
        settings = scenario('outer_overlap')
        result = run(state, poses, controls, gate(until=2), horizon=4, settings=settings)
        baseline = run(state, poses, controls, None, horizon=4, settings=settings)
        self.assertTrue(result['diagnostics']['safety_enabled'])
        self.assertNotEqual(result['frames'], baseline['frames'])
        self.assertTrue(any(a['mode'] == 'slow' for f in result['frames'] for a in f['agents']))

    def test_inputs_unchanged_empty_selection_equivalent_and_validation(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 5])])])
        poses, controls, admission = frame(0, [(4, 4)]), [controller(0)], gate(ids=())
        saved = deepcopy((state, poses, controls, admission))
        result = run(state, poses, controls, admission)
        self.assertEqual((state, poses, controls, admission), saved)
        self.assertEqual(result['frames'], run(state, poses, controls, None)['frames'])
        invalid = [dict(admission, protect_clearance=True), dict(admission, held_robot_ids=[0, 0]),
                   dict(admission, held_robot_ids=[9]), dict(admission, until_offset_s=float('nan')),
                   dict(admission, half_width=-1), dict(admission, until_offset_s=0),
                   dict(admission, until_offset_s=.15)]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                run(state, poses, controls, item)
        with self.assertRaisesRegex(ValueError, 'requires native controller'):
            rollout(state, poses, scenario(), {}, admission=gate())


class ArchivedDefaultRegression(unittest.TestCase):
    def test_none_matches_archived_rollout_bytes_on_27_54_81_robot_states(self):
        root = Path(__file__).resolve().parents[3]
        study = root / 'benchmark/data/entry_gates/central_gate_v3'
        archived = study / 'source_snapshot/simulator/experiments/prediction/joint.py'
        if not archived.exists():
            self.skipTest('Archived native study is not present in this checkout')
        spec = importlib.util.spec_from_file_location('joint_archived_v3', archived)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for count, clock in [(27, 600), (54, 1200), (81, 1800)]:
            with self.subTest(count=count, clock=clock):
                directory = study / f'w8302_fleet{count}_none'
                event = json.loads((directory / 'bridge' / f'request_{clock}.json').read_text())
                settings = json.loads((directory / 'scenario.json').read_text())
                floor = json.loads((directory / 'map.json').read_text())
                args = (event['snapshot'], event['pose_frame'], settings, floor)
                kwargs = dict(controllers=event['controllers'], horizon_s=5, dt=.1)
                before = json.dumps(old.rollout(*args, **kwargs), allow_nan=False).encode()
                after = json.dumps(rollout(*args, **kwargs, admission=None), allow_nan=False).encode()
                self.assertEqual(after, before)


class NativePollingOrderTests(unittest.TestCase):
    def test_origin_prefetch_respects_queue_before_passed_node_acknowledgement(self):
        known = [action(i, [4, 4 + .5 * i], [4, 4.5 + .5 * i]) for i in range(6)]
        state = snapshot([robot(0, known, enqueued=3, committed=4)])
        queue = [{'type': 'M', 'node_ids': [0, 1, 2, 3], 'goal': [4, 6],
                  'orientation_rad': 0, 'remaining_ticks': -1}]
        args = (state, frame(0, [(4, 4.55)]), scenario(enabled=False), {})
        control = [controller(0, queue, speed_m_s=.8, previous_move_speed_m_s=.8)]
        admission = dict(gate(until=2), center=[4., 9.])
        historical = rollout(*args, controllers=control, admission=admission, horizon_s=.1)
        ordered = rollout(*args, controllers=control, admission=admission, horizon_s=.1,
                          native_poll_before_completion=True)
        self.assertEqual(historical['diagnostics']['admission']['events'][0]['node_id'], 5)
        self.assertEqual(ordered['diagnostics']['admission']['events'], [])
        self.assertEqual(ordered['completions'][0]['node_id'], 0)
        self.assertEqual(ordered['diagnostics']['dispatch_order'], 'native_poll_before_completion')

    def test_later_poll_does_not_dispatch_dependency_completed_at_same_boundary(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 4], kind='S')], enqueued=0),
                          robot(1, [action(0, [10, 10], [10, 11], dependencies=[[0, 0]])])])
        queue = [{'type': 'S', 'node_ids': [0], 'goal': [4, 4], 'orientation_rad': 0, 'remaining_ticks': 10}]
        args = (state, frame(0, [(4, 4), (10, 10)]), scenario(enabled=False), {})
        controls = [controller(0, queue), controller(1)]
        ordered = rollout(*args, controllers=controls, horizon_s=2.1, native_poll_before_completion=True)
        self.assertEqual(next(c['offset_s'] for c in ordered['completions'] if c['id'] == 0), 1.)
        self.assertEqual(sample(ordered, 1.9, 1)['action_type'], 'WAIT')
        self.assertEqual(sample(ordered, 2, 1)['action_type'], 'M')
        self.assertGreater(sample(ordered, 2.1, 1)['col'], 10)

    def test_noop_and_empty_gate_have_identical_trajectories_in_new_mode(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 5])])])
        args = (state, frame(0, [(4, 4)]), scenario(enabled=False), {})
        kwargs = dict(controllers=[controller(0)], horizon_s=3, native_poll_before_completion=True,
                      record_progress=True)
        baseline = rollout(*args, **kwargs)
        empty = rollout(*args, **kwargs, admission=gate(ids=()))
        empty['diagnostics'].pop('admission')
        self.assertEqual(empty, baseline)
        with self.assertRaisesRegex(ValueError, 'requires native controller'):
            rollout(*args, native_poll_before_completion=True)


class AcceptedProgressTests(unittest.TestCase):
    def test_progress_reports_move_turn_service_and_stops_at_prefix_end(self):
        state = snapshot([robot(0, [action(0, [4, 4], [3, 4]),
                                   action(1, [3, 4], [3, 4], kind='T', orientation=1),
                                   action(2, [3, 4], [3, 4], kind='S')])])
        result = rollout(state, frame(0, [(4, 4)]), scenario(enabled=False), {}, horizon_s=6,
                         record_progress=True)
        self.assertEqual(sample(result, 0)['accepted_work_done_s'], 0)
        self.assertAlmostEqual(sample(result, 1)['accepted_work_done_s'], 1)
        self.assertAlmostEqual(sample(result, 2)['accepted_work_done_s'], 2)
        self.assertAlmostEqual(sample(result, 4)['accepted_work_done_s'], 4)
        self.assertAlmostEqual(sample(result, 6)['accepted_work_done_s'], 4)
        self.assertEqual(sample(result, 6)['action_type'], 'EXHAUSTED')

    def test_progress_does_not_change_trajectories_or_credit_holding(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 5])])])
        args = (state, frame(0, [(4, 4)]), scenario(enabled=False), {})
        kwargs = dict(horizon_s=3, controllers=[controller(0)], admission=gate())
        result = rollout(*args, **kwargs, record_progress=True)
        original = rollout(*args, **kwargs)
        self.assertEqual(sample(result, 1)['accepted_work_done_s'], 0)
        self.assertGreater(sample(result, 2)['accepted_work_done_s'], 0)
        result['diagnostics'].pop('progress_definition')
        for entry in result['frames']:
            for agent in entry['agents']:
                agent.pop('accepted_work_done_s')
        self.assertEqual(result, original)

    def test_reverse_to_accepted_goal_is_negative_progress_not_wiggle_credit(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 5])], enqueued=0)])
        queue = [{'type': 'M', 'node_ids': [0], 'goal': [4, 5], 'orientation_rad': 0, 'remaining_ticks': -1}]
        result = rollout(state, frame(0, [(4, 5.2)]), scenario(enabled=False), {}, horizon_s=.1,
                         controllers=[controller(0, queue)], record_progress=True)
        self.assertLess(sample(result, .1)['accepted_work_done_s'], 0)

    def test_native_service_uses_only_remaining_timer_and_zero_rates_rejected(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 4], kind='S')], enqueued=0)])
        queue = [{'type': 'S', 'node_ids': [0], 'goal': [4, 4], 'orientation_rad': 0, 'remaining_ticks': 3}]
        args = (state, frame(0, [(4, 4)]), scenario(enabled=False), {})
        result = rollout(*args, horizon_s=2, controllers=[controller(0, queue)], record_progress=True)
        self.assertAlmostEqual(sample(result, 2)['accepted_work_done_s'], .3)
        for kwargs in [dict(turn_rate_deg_s=0), dict(speed_m_s=0), dict(controllers=[controller(0, queue, nominal_speed_m_s=0)])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rollout(*args, record_progress=True, **kwargs)


if __name__ == '__main__':
    unittest.main()
