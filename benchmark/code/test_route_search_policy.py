"""Causal cost proposals and safe selection among actual planner previews."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from route_search_policy import POLICY, RouteSearchPolicy, forecast_cost_menu, road_mask


def fixture():
    observed, clock = 121, 120
    pose = {'clock': observed, 'agents': [dict(id=0, x=-2., y=-2., rz=0.)]}
    snapshot = {'clock': observed, 'server_clock': clock, 'robots': [dict(id=0, actions=[])]}
    candidates = [dict(index=i, name=name, evaluated=True, success=True,
                       planner_wall_s=.01, nonzero_penalty_cells=0, max_cell_penalty=0.,
                       plan={'candidate_index': i}, preview_snapshot=dict(deepcopy(snapshot), branch=i))
                  for i, name in enumerate(('zero', 'mild', 'strong'))]
    event = dict(clock=clock, observation_clock=observed, snapshot=snapshot, pose_frame=pose,
                 controllers=[dict(id=0, clock=observed)], candidates=candidates,
                 boundary='post_commit_before_plan_selection')
    past = np.broadcast_to(np.array([[[2., 2., 0.]]]), (121, 1, 3)).copy()
    scenario = {'agents': [{'id': 0, 'goals': [[4., 4.]]}]}
    return event, past, scenario, {'layout': ['.....'] * 5}


def eta_stub(values):
    def score(snapshot, pose, controllers, scenario, map_doc, targets, **kwargs):
        value = values[snapshot['branch']]
        if isinstance(value, Exception):
            raise value
        return dict(mean_eta_s=value, per_robot_eta_s={'0': value},
                    completed_terminal_ids=[], optimistic_tail_ids=[], diagnostics={})
    return score


def real_route_fixture():
    event, past, scenario, map_doc = fixture()
    scenario.update(ticks_per_second=10, cell_size_m=1., safety={'enabled': False})
    scenario['agents'][0]['goals'] = [[2., 4.]]
    event['controllers'][0].update(queue=[], poll_phase=0, nominal_speed_m_s=1.,
                                    speed_m_s=0., previous_move_speed_m_s=0., acceleration_m_s2=2.)

    def actions(segments):
        result, previous = [], [2., 2.]
        for node, (kind, goal, orientation) in enumerate(segments, 1):
            result.append(dict(node_id=node, type=kind, start=previous, goal=list(goal),
                               orientation=orientation, dependencies=[], task_id=7 if kind == 'S' else -1))
            previous = list(goal)
        return result

    detour = actions([('M', (1., 2.), 0), ('T', (1., 2.), 1),
                      ('M', (1., 4.), 1), ('T', (1., 4.), 2),
                      ('M', (2., 4.), 2), ('S', (2., 4.), 2)])
    direct = actions([('T', (2., 2.), 1), ('M', (2., 4.), 1), ('S', (2., 4.), 1)])
    for candidate, route in zip(event['candidates'], (detour, direct, direct)):
        robot = candidate['preview_snapshot']['robots'][0]
        robot.update(finished_node_id=0, enqueued_node_id=0, committed_node_id=0, actions=route)
        candidate['plan'] = {'actions': route}
    event['snapshot'] = deepcopy(event['candidates'][0]['preview_snapshot'])
    return event, past, scenario, map_doc


class RouteSearchTests(unittest.TestCase):
    def test_forecast_costs_use_slow_layer_windows_and_spatial_expansion(self):
        fields = np.zeros((3, 2, 5, 5))
        fields[:, 0] = 1000.  # ordinary occupancy contributes no cost
        fields[:, 1, 2, 2] = [1., 2., 3.]
        road = np.ones((5, 5), dtype=bool)
        road[2, 2] = False
        menu = forecast_cost_menu(fields, road)
        self.assertEqual([c['name'] for c in menu], ['zero', 'mild', 'strong'])
        expected = np.zeros((5, 5)); expected[1:4, 1:4] = 3.4; expected[2, 2] = 0.
        np.testing.assert_allclose(np.array(menu[1]['cell_penalties']).reshape(5, 5), expected)
        np.testing.assert_allclose(menu[2]['cell_penalties'], 4 * expected.ravel())
        self.assertEqual(sum(menu[0]['cell_penalties']), 0.)

    def test_costs_are_row_major_bounded_and_never_enter_obstacles(self):
        fields = np.zeros((3, 2, 3, 4)); fields[:, 1, 0, 0] = 1e6
        road = road_mask({'layout': ['.@..', '.T..', '....']})
        menu = forecast_cost_menu(fields, road)
        self.assertEqual(menu[2]['cell_penalties'][0], 1000.)
        self.assertEqual(menu[2]['cell_penalties'][1], 0.)
        self.assertEqual(menu[2]['cell_penalties'][4], 1000.)
        self.assertEqual(menu[2]['cell_penalties'][5], 0.)
        self.assertLessEqual(max(menu[2]['cell_penalties']), 1000.)

    def test_invalid_fields_rejected(self):
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                forecast_cost_menu(np.full((3, 2, 2, 2), value), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            forecast_cost_menu(np.zeros((2, 2, 2, 2)), np.ones((2, 2)))

    def test_noop_proposal_needs_no_history_or_rollout(self):
        event, _, scenario, map_doc = fixture()
        with patch('route_search_policy.rollout') as predict:
            payload, diagnostics = RouteSearchPolicy().propose(event, None, scenario, map_doc, 'none')
        self.assertEqual(len(payload['route_candidates']), 1)
        self.assertEqual(payload['cell_penalties'], [0.] * 25)
        self.assertEqual(diagnostics['forecast_wall_s'], 0.)
        predict.assert_not_called()

    def test_both_selectors_receive_the_same_three_forecast_cost_fields(self):
        fields = np.zeros((201, 2, 5, 5)); fields[:, 1, 2, 2] = 1.
        with patch('route_search_policy.rollout', return_value={}) as predict, \
                patch('route_search_policy.predicted_rasters', return_value=(fields, None)):
            nominal, _ = RouteSearchPolicy().propose(*fixture(), method='nominal')
            rollout, diagnostics = RouteSearchPolicy().propose(*fixture(), method='rollout')
        self.assertEqual(nominal, rollout)
        self.assertEqual(diagnostics['distinct_penalty_fields'], 3)
        self.assertTrue(diagnostics['forecast_is_shared_by_selectors'])
        self.assertEqual(predict.call_args.kwargs['horizon_s'], 20.)
        self.assertTrue(predict.call_args.kwargs['native_poll_before_completion'])
        json.dumps(diagnostics, allow_nan=False)

    def test_zero_heat_retains_and_diagnoses_duplicate_candidate_fields(self):
        fields = np.zeros((201, 2, 5, 5))
        with patch('route_search_policy.rollout', return_value={}), \
                patch('route_search_policy.predicted_rasters', return_value=(fields, None)):
            payload, diagnostics = RouteSearchPolicy().propose(*fixture(), method='nominal')
        self.assertEqual(len(payload['route_candidates']), 3)
        self.assertEqual(diagnostics['distinct_penalty_fields'], 1)

    def test_forecast_rejects_short_stale_or_nonfinite_history(self):
        for kind in ('short', 'stale', 'nonfinite', 'clock'):
            event, past, scenario, map_doc = fixture()
            if kind == 'short': past = past[-20:]
            if kind == 'stale': past[-1, 0, 0] += .1
            if kind == 'nonfinite': past[0, 0, 0] = np.nan
            if kind == 'clock': event['controllers'][0]['clock'] -= 1
            with self.subTest(kind=kind), patch('route_search_policy.rollout') as predict, \
                    self.assertRaises(ValueError):
                RouteSearchPolicy().propose(event, past, scenario, map_doc)
            predict.assert_not_called()

    def select(self, values, method='rollout', event=None, completed=()):
        source, _, scenario, map_doc = fixture()
        if event is not None: source = event
        with patch('route_search_policy.service_targets', return_value={'0': [[17, [4., 4.]]]}) as targets, \
                patch('route_search_policy.score_route_eta', side_effect=eta_stub(values)) as scores:
            result = RouteSearchPolicy().select(source, scenario, map_doc, method, completed)
        return *result, targets, scores

    def test_zero_wins_exact_ties_and_gain_at_or_below_margin(self):
        for alternate in (10., 9.9, 9.5):
            with self.subTest(alternate=alternate):
                payload, diagnostics, _, _ = self.select([10., alternate, 11.])
                self.assertEqual(payload['selected_index'], 0)
                self.assertEqual(diagnostics['selection_reason'], 'zero_wins_tie_or_margin')

    def test_choose_best_eligible_actual_plan_above_strict_mean_eta_margin(self):
        payload, diagnostics, targets, scores = self.select([10., 9.4, 8.])
        self.assertEqual(payload, {'clock': 120, 'selected_index': 2})
        self.assertEqual(diagnostics['selected_mean_eta_gain_s'], 2.)
        self.assertEqual(diagnostics['selected_name'], 'strong')
        self.assertEqual(scores.call_count, 3)
        for call in scores.call_args_list:
            self.assertEqual(call.kwargs, {'mode': 'rollout', 'horizon_s': 30.})
        self.assertEqual(targets.call_count, 1)
        self.assertEqual(diagnostics['candidates'][2]['eta']['per_robot_eta_s'], {'0': 8.})
        json.dumps(diagnostics, allow_nan=False)

    def test_nominal_uses_same_selection_but_nominal_scoring(self):
        payload, _, _, scores = self.select([10., 8., 9.], method='nominal')
        self.assertEqual(payload['selected_index'], 1)
        self.assertTrue(all(call.kwargs['mode'] == 'nominal' for call in scores.call_args_list))

    def test_real_route_eta_penalizes_detours_in_both_selectors(self):
        # Prescribed routes isolate the ranking objective; this is not a native
        # PBS study or evidence of a congestion-management improvement.
        event, _, scenario, map_doc = real_route_fixture()
        for method in ('nominal', 'rollout'):
            with self.subTest(method=method):
                payload, diagnostics = RouteSearchPolicy().select(event, scenario, map_doc, method)
                self.assertEqual(payload['selected_index'], 1)
                self.assertGreater(diagnostics['selected_mean_eta_gain_s'], .5)
                self.assertAlmostEqual(diagnostics['candidates'][0]['mean_nominal_eta_s'], 8.)
                self.assertAlmostEqual(diagnostics['candidates'][1]['nominal_detour_mean_s'], -3.)
                self.assertTrue(diagnostics['candidates'][1]['common_service_targets_validated'])
                if method == 'rollout':
                    self.assertEqual(diagnostics['candidates'][1]['eta']['completed_terminal_ids'], [0])

    def test_real_service_validator_rejects_a_shorter_route_to_wrong_task(self):
        event, _, scenario, map_doc = real_route_fixture()
        event['candidates'][1]['preview_snapshot']['robots'][0]['actions'][-1]['task_id'] = 999
        event['candidates'][2]['success'] = False
        payload, diagnostics = RouteSearchPolicy().select(event, scenario, map_doc, 'nominal')
        self.assertEqual(payload['selected_index'], 0)
        self.assertIn('service IDs/goals', diagnostics['candidates'][1]['rejection_reason'])

    def test_alternate_tie_prefers_lower_cost_menu_index(self):
        payload, *_ = self.select([10., 8., 8.])
        self.assertEqual(payload['selected_index'], 1)

    def test_none_keeps_zero_without_eta_or_task_suffix_validation(self):
        payload, diagnostics, targets, scores = self.select([], method='none')
        self.assertEqual(payload['selected_index'], 0)
        self.assertEqual(diagnostics['selection_reason'], 'zero_only')
        targets.assert_not_called(); scores.assert_not_called()

    def test_failed_zero_preserves_native_fallback_without_validation(self):
        event, *_ = fixture(); event['candidates'][0]['success'] = False
        event['candidates'][0]['preview_snapshot'] = None
        payload, diagnostics, targets, scores = self.select([], event=event)
        self.assertEqual(payload['selected_index'], 0)
        self.assertEqual(diagnostics['selection_reason'], 'zero_planner_failed_native_fallback')
        targets.assert_not_called(); scores.assert_not_called()

    def test_failed_or_skipped_alternative_never_scored_or_selected(self):
        event, *_ = fixture()
        event['candidates'][1]['success'] = False
        event['candidates'][1]['preview_snapshot'] = None
        event['candidates'][2]['evaluated'] = False
        payload, diagnostics, _, scores = self.select([10., 0., 0.], event=event)
        self.assertEqual(payload['selected_index'], 0)
        self.assertEqual(scores.call_count, 1)
        self.assertFalse(diagnostics['candidates'][1]['eligible'])

    def test_service_mismatch_excludes_alt_and_is_audited(self):
        payload, diagnostics, _, _ = self.select([10., ValueError('service suffix differs'), 9.])
        self.assertEqual(payload['selected_index'], 2)
        self.assertEqual(diagnostics['candidates'][1]['rejection_reason'], 'service suffix differs')

    def test_invalid_zero_score_prevents_ranking_alternatives(self):
        payload, diagnostics, _, scores = self.select([ValueError('missing final service'), 0., 0.])
        self.assertEqual(payload['selected_index'], 0)
        self.assertEqual(diagnostics['selection_reason'], 'zero_score_invalid')
        self.assertEqual(scores.call_count, 1)

    def test_invalid_zero_targets_prevents_any_scoring(self):
        event, _, scenario, map_doc = fixture()
        with patch('route_search_policy.service_targets', side_effect=ValueError('unknown completion')), \
                patch('route_search_policy.score_route_eta') as score:
            payload, diagnostics = RouteSearchPolicy().select(event, scenario, map_doc)
        self.assertEqual(payload['selected_index'], 0)
        self.assertEqual(diagnostics['selection_reason'], 'zero_target_invalid')
        score.assert_not_called()

    def test_completion_ids_are_passed_from_native_ledger_explicitly(self):
        _, diagnostics, targets, _ = self.select([10., 10., 10.], completed=[9, 2])
        self.assertEqual(targets.call_args.kwargs['completed_robot_ids'], [2, 9])
        self.assertEqual(diagnostics['completed_robot_ids'], [2, 9])

    def test_bad_preview_clock_cannot_win(self):
        event, *_ = fixture(); event['candidates'][1]['preview_snapshot']['clock'] -= 1
        payload, diagnostics, _, scores = self.select([10., 0., 10.], event=event)
        self.assertEqual(payload['selected_index'], 0)
        self.assertEqual(scores.call_count, 2)
        self.assertIn('clocks differ', diagnostics['candidates'][1]['rejection_reason'])

    def test_invalid_numeric_eta_cannot_win_and_remains_json_safe(self):
        for bad in (float('nan'), float('inf'), -1.):
            with self.subTest(bad=bad):
                payload, diagnostics, _, _ = self.select([10., bad, 10.])
                self.assertEqual(payload['selected_index'], 0)
                self.assertIsNone(diagnostics['candidates'][1]['mean_eta_s'])
                json.dumps(diagnostics, allow_nan=False)

    def test_unknown_method_bad_margin_and_wrong_boundary_rejected(self):
        for margin in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): RouteSearchPolicy(margin)
        with self.assertRaises(ValueError): RouteSearchPolicy().propose(*fixture(), method='typo')
        event, *_ = fixture(); event['boundary'] = 'post_planner_before_dispatch'
        with self.assertRaises(ValueError): self.select([10., 9., 8.], event=event)


if __name__ == '__main__':
    unittest.main()
