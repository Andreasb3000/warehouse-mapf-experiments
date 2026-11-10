"""Alternative routes pay travel time to identical service targets."""
from copy import deepcopy
import json
import unittest

from route_eta import service_targets, score_route_eta
from experiments.prediction.joint import rollout
from experiments.prediction.test_joint import controller, scenario
from experiments.prediction.test_predict import action, robot, snapshot, frame


def service(node, location, task_id=17):
    value = action(node, location, location, kind='S')
    value['task_id'] = task_id
    return value


def routes():
    short = snapshot([robot(0, [action(0, [10, 10], [8, 10], orientation=0), service(1, [8, 10])])])
    detour = snapshot([robot(0, [
        action(0, [10, 10], [10, 10], kind='T', orientation=1),
        action(1, [10, 10], [10, 12], orientation=1),
        action(2, [10, 12], [10, 12], kind='T', orientation=0),
        action(3, [10, 12], [8, 12], orientation=0),
        action(4, [8, 12], [8, 12], kind='T', orientation=3),
        action(5, [8, 12], [8, 10], orientation=3), service(6, [8, 10])])])
    return short, detour


def score(state, targets, *, poses=None, controls=None, mode='nominal', horizon=30):
    return score_route_eta(state, poses or frame(0, [(10, 10)]), controls or [controller(0)],
                           scenario(enabled=False), {}, targets, mode=mode, horizon_s=horizon)


class RouteEtaTests(unittest.TestCase):
    def test_nominal_and_rollout_charge_longer_detour_to_same_service(self):
        short, detour = routes()
        targets = service_targets(short)
        a, b = score(short, targets), score(detour, targets)
        self.assertAlmostEqual(a['mean_eta_s'], 4.)
        self.assertAlmostEqual(b['mean_eta_s'], 11.)
        for horizon in (1., 30.):
            with self.subTest(horizon=horizon):
                a = score(short, targets, mode='rollout', horizon=horizon)
                b = score(detour, targets, mode='rollout', horizon=horizon)
                self.assertGreater(b['mean_eta_s'], a['mean_eta_s'])
                self.assertEqual(a['mean_nominal_eta_s'], 4.)
                self.assertEqual(b['mean_nominal_eta_s'], 11.)

    def test_partial_or_empty_route_never_earns_prefix_exhaustion_bonus(self):
        short, _ = routes()
        targets = service_targets(short)
        partial = deepcopy(short)
        partial['robots'][0]['actions'].pop()
        with self.assertRaisesRegex(ValueError, 'terminal service'):
            score(partial, targets)
        with self.assertRaisesRegex(ValueError, 'terminal service'):
            service_targets(snapshot([robot(0, [])]))
        with self.assertRaisesRegex(ValueError, 'terminal service'):
            score(snapshot([robot(0, [])]), targets)

    def test_all_remaining_services_are_common_not_just_last_destination(self):
        short, _ = routes()
        both = snapshot([robot(0, [service(0, [10, 10], task_id=16),
                                  action(1, [10, 10], [8, 10]), service(2, [8, 10])])])
        targets = service_targets(both)
        self.assertEqual(len(targets[0]['services']), 2)
        with self.assertRaisesRegex(ValueError, 'service IDs/goals'):
            score(short, targets)
        changed = deepcopy(both)
        changed['robots'][0]['actions'][0]['task_id'] = 99
        with self.assertRaisesRegex(ValueError, 'service IDs/goals'):
            score(changed, targets)

    def test_other_goal_or_task_id_is_not_comparable(self):
        short, _ = routes()
        targets = service_targets(short)
        for field, value in [('task_id', 99), ('goal', [8, 11])]:
            candidate = deepcopy(short)
            candidate['robots'][0]['actions'][-1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'service IDs/goals'):
                score(candidate, targets)

    def test_only_confirmed_parked_robot_gets_zero_fixed_denominator_retained(self):
        short, _ = routes()
        short['robots'].append(robot(1, []))
        targets = service_targets(short, completed_robot_ids=[1])
        result = score(short, targets, poses=frame(0, [(10, 10), (20, 20)]),
                       controls=[controller(0), controller(1)])
        self.assertEqual(result['per_robot_eta_s'], {'0': 4., '1': 0.})
        self.assertEqual(result['mean_eta_s'], 2.)
        self.assertEqual(result['parked_robot_ids'], [1])
        changed = deepcopy(short)
        changed['robots'].pop()
        with self.assertRaisesRegex(ValueError, 'denominator'):
            score(changed, targets)
        with self.assertRaisesRegex(ValueError, 'still has accepted actions'):
            service_targets(short, completed_robot_ids=[0, 1])

    def test_terminal_service_completion_uses_actual_remaining_timer(self):
        state = snapshot([robot(0, [service(0, [10, 10])], enqueued=0)])
        queue = [{'type': 'S', 'node_ids': [0], 'goal': [10, 10],
                  'orientation_rad': 0., 'remaining_ticks': 3}]
        result = score(state, service_targets(state), controls=[controller(0, queue)], mode='rollout', horizon=1)
        self.assertEqual(result['completed_terminal_ids'], [0])
        self.assertEqual(result['optimistic_tail_ids'], [])
        self.assertAlmostEqual(result['mean_eta_s'], .3)
        self.assertAlmostEqual(result['mean_nominal_eta_s'], .3)

    def test_earlier_pickup_service_completion_does_not_finish_terminal_goal(self):
        state = snapshot([robot(0, [service(0, [10, 10], 16),
                                   action(1, [10, 10], [8, 10]), service(2, [8, 10])], enqueued=0)])
        queue = [{'type': 'S', 'node_ids': [0], 'goal': [10, 10],
                  'orientation_rad': 0., 'remaining_ticks': 3}]
        result = score(state, service_targets(state), controls=[controller(0, queue)], mode='rollout', horizon=1)
        self.assertEqual(result['completed_terminal_ids'], [])
        self.assertEqual(result['optimistic_tail_ids'], [0])
        self.assertGreater(result['mean_eta_s'], 3)
        self.assertAlmostEqual(result['mean_nominal_eta_s'], 4.3)

    def test_tail_matches_horizon_plus_remaining_work_and_is_nonnegative(self):
        state, _ = routes()
        result = score(state, service_targets(state), mode='rollout', horizon=1)
        tail = result['diagnostics']['rollout_diagnostics']['nominal_remaining_work_s']['terminal']['0']
        self.assertAlmostEqual(result['mean_eta_s'], 1 + tail)
        self.assertGreaterEqual(tail, 0)
        state['robots'][0]['enqueued_node_id'] = 0
        queue = [{'type': 'M', 'node_ids': [0], 'goal': [8, 10],
                  'orientation_rad': 0., 'remaining_ticks': -1}]
        overshot = score(state, service_targets(state), poses=frame(0, [(7.5, 10)]), controls=[controller(0, queue)])
        self.assertAlmostEqual(overshot['mean_eta_s'], 2.5)

    def test_missing_dependency_is_rejected_instead_of_a_fake_completion_estimate(self):
        state, _ = routes()
        targets = service_targets(state)
        state['robots'][0]['actions'][0]['dependencies'] = [[9, 99]]
        for mode in ('nominal', 'rollout'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'missing ADG dependencies'):
                score(state, targets, mode=mode, horizon=1)

    def test_safety_rollout_can_prefer_costlier_clear_detour_without_a_gate(self):
        short, detour = routes()
        for state in (short, detour):
            state['robots'].append(robot(1, []))
        targets = service_targets(short, completed_robot_ids=[1])
        poses, controls = frame(0, [(10, 10), (9, 10)]), [controller(0), controller(1)]
        settings = scenario(inner_stop_enabled=True)
        short_eta = score_route_eta(short, poses, controls, settings, {}, targets, horizon_s=30)
        detour_eta = score_route_eta(detour, poses, controls, settings, {}, targets, horizon_s=30)
        self.assertGreater(detour_eta['mean_nominal_eta_s'], short_eta['mean_nominal_eta_s'])
        self.assertLess(detour_eta['mean_eta_s'], short_eta['mean_eta_s'])
        self.assertEqual(short_eta['optimistic_tail_ids'], [0])
        self.assertEqual(detour_eta['completed_terminal_ids'], [0])
        self.assertNotIn('admission', detour_eta['diagnostics']['rollout_diagnostics'])

    def test_remaining_work_metadata_does_not_change_any_default_trajectory(self):
        state, _ = routes()
        args = (state, frame(0, [(10, 10)]), scenario(enabled=False), {})
        kwargs = dict(controllers=[controller(0)], native_poll_before_completion=True, horizon_s=3)
        baseline = rollout(*args, **kwargs)
        with_work = rollout(*args, **kwargs, record_remaining_work=True)
        with_work['diagnostics'].pop('nominal_remaining_work_s')
        self.assertEqual(with_work, baseline)

    def test_no_input_mutation_and_json_serializable_scores(self):
        state, _ = routes()
        targets, poses, controls = service_targets(state), frame(0, [(10, 10)]), [controller(0)]
        before = deepcopy((state, targets, poses, controls))
        result = score(state, targets, poses=poses, controls=controls, mode='rollout', horizon=3)
        self.assertEqual((state, targets, poses, controls), before)
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
