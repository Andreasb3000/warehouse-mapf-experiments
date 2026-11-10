"""Search must prefer no-op without a fleet-wide predicted progress gain."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from search_admission_policy import (POLICY, SearchAdmissionPolicy, candidate_interventions,
                                     eligible_entries, progress_score)


def move(node, start, goal):
    return {'node_id': node, 'type': 'M', 'start': list(start), 'goal': list(goal),
            'orientation': 0, 'dependencies': [], 'task_id': -1}


def fixture(clock=120):
    positions = [(30., 30.), (26., 30.), (30., 26.)]
    paths = [[move(1, positions[0], (34., 30.))],
             [move(1, positions[1], (26.5, 30.)), move(2, (26.5, 30.), (27., 30.)),
              move(3, (27., 30.), (28., 30.))],
             [move(1, positions[2], (30., 26.5)), move(2, (30., 26.5), (30., 27.)),
              move(3, (30., 27.), (30., 28.))]]
    event = {'clock': clock, 'observation_clock': clock + 1,
             'boundary': 'post_planner_before_dispatch',
             'pose_frame': {'clock': clock + 1, 'agents': [dict(id=i, x=-r, y=-c, rz=0.)
                                                         for i, (r, c) in enumerate(positions)]},
             'snapshot': {'clock': clock + 1, 'server_clock': clock,
                          'robots': [dict(id=i, finished_node_id=0, enqueued_node_id=0,
                                          committed_node_id=0, actions=path) for i, path in enumerate(paths)]},
             'controllers': [dict(id=i, clock=clock + 1, queue=[], poll_phase=0,
                                  nominal_speed_m_s=1., speed_m_s=0., previous_move_speed_m_s=0.,
                                  acceleration_m_s2=2.) for i in range(3)]}
    scenario = {'ticks_per_second': 10, 'cell_size_m': 1., 'safety': {'enabled': False},
                'agents': [dict(id=i, goals=[[40., 40.]]) for i in range(3)]}
    return event, None, scenario, {'layout': ['.' * 61 for _ in range(61)]}


def advance(event, clock):
    event = deepcopy(event)
    event['clock'] = clock
    event['observation_clock'] = event['pose_frame']['clock'] = event['snapshot']['clock'] = clock + 1
    event['snapshot']['server_clock'] = clock
    for controller in event['controllers']:
        controller['clock'] = clock + 1
    return event


def scored_rollout(scores):
    """Synthetic trajectories isolate selection logic, not simulator fidelity."""
    def predict(snapshot, pose, scenario, map_doc, **kwargs):
        held = tuple(kwargs['admission']['held_robot_ids']) if kwargs['admission'] else ()
        score = scores.get(held, scores.get((), 5.))
        offsets = np.arange(round(kwargs['horizon_s'] / kwargs['dt']) + 1) * kwargs['dt']
        scale = score / offsets[:-1].mean()
        ids = [r['id'] for r in snapshot['robots']]
        frames = [dict(offset_s=float(t), agents=[dict(id=rid, accepted_work_done_s=float(scale * t / len(ids)))
                                                   for rid in ids]) for t in offsets]
        diagnostics = {}
        if held:
            diagnostics['admission'] = dict(dispatch_block_events=len(held),
                empty_queue_wait_robot_seconds=float(len(held)), commitment_prerequisite_bypasses=0,
                events=[dict(robot_id=rid, node_id=1, offset_s=0., queued_action_count=0,
                             dispatchable_before_gate=0) for rid in held])
        return {'frames': frames, 'diagnostics': diagnostics}
    return predict


class SearchAdmissionTests(unittest.TestCase):
    def test_noop_wins_tied_predictions(self):
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5.})):
            payload, diagnostics = SearchAdmissionPolicy().decide(*fixture())
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(diagnostics['selected_candidate'], 'noop')
        self.assertEqual(diagnostics['noop_fraction'], 1.)

    def test_noop_wins_if_holding_every_candidate_reduces_fleet_progress(self):
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 0., (2,): 0.})):
            payload, diagnostics = SearchAdmissionPolicy().decide(*fixture())
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(diagnostics['best_gain_s'], 0.)

    def test_gain_must_exceed_absolute_margin(self):
        for score in (5.1, 5.5):
            with self.subTest(score=score), patch('search_admission_policy.rollout',
                    side_effect=scored_rollout({(): 5., (1,): score})):
                payload, _ = SearchAdmissionPolicy().decide(*fixture())
                self.assertEqual(payload['held_robot_ids'], [])

    def test_best_counterfactual_above_margin_is_selected_and_audited(self):
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6., (2,): 5.7})) as calls:
            payload, diagnostics = SearchAdmissionPolicy().decide(*fixture(), method='search_h5')
        self.assertEqual(payload['held_robot_ids'], [1])
        self.assertEqual(payload['until_clock'], 140)
        self.assertFalse(payload['protect_clearance'])
        self.assertAlmostEqual(diagnostics['selected_gain_s'], 1.)
        self.assertEqual(diagnostics['candidate_count'], 3)
        self.assertEqual(diagnostics['candidates'][1]['predicted_dispatch_block_events'], 1)
        self.assertEqual(diagnostics['choice_fraction'], 1.)
        self.assertIsNone(calls.call_args_list[0].kwargs['admission'])
        for call in calls.call_args_list:
            self.assertTrue(call.kwargs['record_progress'])
            self.assertTrue(call.kwargs['native_poll_before_completion'])
            self.assertEqual(call.kwargs['horizon_s'], 5.)
            if call.kwargs['admission']:
                self.assertEqual(call.kwargs['admission']['until_offset_s'], 2.)
        json.dumps(diagnostics, allow_nan=False)

    def test_default_absolute_margin_does_not_scale_with_distant_fleet_progress(self):
        scores = {(): 100., (1,): 100.8}
        with patch('search_admission_policy.rollout', side_effect=scored_rollout(scores)):
            default, _ = SearchAdmissionPolicy().decide(*fixture())
            relative, diagnostics = SearchAdmissionPolicy(relative_gain_fraction=.01).decide(*fixture())
        self.assertEqual(default['held_robot_ids'], [1])
        self.assertEqual(relative['held_robot_ids'], [])
        self.assertEqual(diagnostics['required_gain_s'], 1.)

    def test_zero_baseline_has_no_infinite_relative_gain(self):
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 0., (1,): .6})):
            payload, diagnostics = SearchAdmissionPolicy().decide(*fixture())
        self.assertEqual(payload['held_robot_ids'], [1])
        self.assertIsNone(diagnostics['candidates'][1]['relative_gain_fraction'])
        json.dumps(diagnostics, allow_nan=False)

    def test_candidate_budget_and_nearest_robot_per_approach(self):
        entries = {2 * side + j: dict(approach=name, remaining_move_distance_m=float(2 - j))
                   for side, name in enumerate(('north', 'east', 'south', 'west')) for j in range(2)}
        candidates = candidate_interventions(entries)
        self.assertEqual(len(candidates), 9)
        self.assertEqual(candidates[1]['held_robot_ids'], [1])
        self.assertEqual(candidates[5]['held_robot_ids'], [0, 1])
        self.assertEqual(len({tuple(c['held_robot_ids']) for c in candidates}), 9)

    def test_single_robot_approach_deduplicates_whole_approach(self):
        entries = {7: dict(approach='north', remaining_move_distance_m=1.)}
        self.assertEqual(len(candidate_interventions(entries)), 2)
        self.assertEqual(candidate_interventions(entries, {7}), [{'name': 'noop', 'held_robot_ids': []}])

    def test_inside_queued_and_committed_entries_are_excluded(self):
        for field in ('enqueued_node_id', 'committed_node_id'):
            event, *_ = fixture()
            event['snapshot']['robots'][1][field] = 2
            entries, exclusions = eligible_entries(event)
            self.assertEqual(set(entries), {2})
            self.assertEqual(exclusions['inside'], [0])
            self.assertEqual(exclusions['committed_or_queued_entry'], [1])

    def test_uncommitted_entry_with_already_queued_approach_is_eligible(self):
        event, *_ = fixture()
        event['snapshot']['robots'][1]['enqueued_node_id'] = 1
        entries, _ = eligible_entries(event)
        self.assertIn(1, entries)
        self.assertEqual(entries[1]['entry_node_id'], 2)

    def test_absent_or_distant_entry_is_not_candidate(self):
        event, *_ = fixture()
        event['snapshot']['robots'][1]['actions'] = []
        event['pose_frame']['agents'][2]['y'] = -20.
        event['snapshot']['robots'][2]['actions'][0]['start'] = [30., 20.]
        entries, exclusions = eligible_entries(event)
        self.assertEqual(entries, {})
        self.assertEqual(exclusions['no_nearby_accepted_entry'], [1, 2])

    def test_earlier_work_receives_higher_score_than_equal_terminal_work(self):
        def result(early):
            return {'frames': [dict(offset_s=k / 10, agents=[dict(id=4, accepted_work_done_s=
                min(k / 10, 2.) if early else max(0., k / 10 - 3.))]) for k in range(51)]}
        early, late = [progress_score(result(flag), [4], 5.) for flag in (True, False)]
        self.assertEqual(early['terminal_work_s'], late['terminal_work_s'])
        self.assertGreater(early['score_work_s'], late['score_work_s'])

    def test_exhausted_short_prefix_never_receives_completion_bonus(self):
        result = {'frames': [dict(offset_s=k / 10, agents=[dict(id=4,
                         accepted_work_done_s=min(k / 10, 1.), action_type='EXHAUSTED')]) for k in range(51)]}
        score = progress_score(result, [4], 5.)
        self.assertLessEqual(score['score_work_s'], 1.)
        self.assertEqual(score['terminal_work_s'], 1.)

    def test_score_includes_every_robot_even_when_other_robots_lose(self):
        frames = [dict(offset_s=k / 10, agents=[dict(id=0, accepted_work_done_s=k / 10),
                      dict(id=1, accepted_work_done_s=-k / 10)]) for k in range(51)]
        self.assertEqual(progress_score({'frames': frames}, [0, 1], 5.)['score_work_s'], 0.)
        with self.assertRaises(ValueError):
            progress_score({'frames': frames}, [0, 1, 2], 5.)

    def test_malformed_progress_or_time_grid_is_rejected(self):
        args = fixture()
        result = scored_rollout({(): 5.})(args[0]['snapshot'], args[0]['pose_frame'], *args[2:],
                                        horizon_s=5., dt=.1, admission=None)
        bad = deepcopy(result); bad['frames'][-1]['offset_s'] = 6.
        with self.assertRaises(ValueError):
            progress_score(bad, [0, 1, 2], 5.)
        bad = deepcopy(result); bad['frames'][3]['agents'][0]['accepted_work_done_s'] = float('nan')
        with self.assertRaises(ValueError):
            progress_score(bad, [0, 1, 2], 5.)

    def test_cap_prevents_repeatedly_postponing_release(self):
        event, past, scenario, map_doc = fixture()
        policy = SearchAdmissionPolicy()
        choices = []
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})):
            for clock in (120, 140, 160, 180, 200):
                payload, diagnostics = policy.decide(advance(event, clock), past, scenario, map_doc)
                choices.append(payload['held_robot_ids'])
                if clock == 180:
                    self.assertEqual(diagnostics['forced_release_robot_ids'], [1])
                    self.assertTrue(all(1 not in c['held_robot_ids'] for c in diagnostics['candidates']))
        self.assertEqual(choices, [[1], [1], [1], [], [1]])
        self.assertEqual(diagnostics['choice_fraction'], .8)

    def test_noop_resets_continuous_hold_history(self):
        event, past, scenario, map_doc = fixture()
        policy = SearchAdmissionPolicy()
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})):
            policy.decide(event, past, scenario, map_doc)
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5.})):
            policy.decide(advance(event, 140), past, scenario, map_doc)
        self.assertEqual(policy.held_since, {})

    def test_missed_refresh_breaks_continuous_holding(self):
        event, past, scenario, map_doc = fixture()
        policy = SearchAdmissionPolicy()
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})):
            policy.decide(event, past, scenario, map_doc)
            _, diagnostics = policy.decide(advance(event, 180), past, scenario, map_doc)
        self.assertEqual(diagnostics['forced_release_robot_ids'], [])
        self.assertEqual(diagnostics['applied_hold_since_clock'], {'1': 180})

    def test_horizon_variants_share_execution_duration(self):
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})) as calls:
            payload, diagnostics = SearchAdmissionPolicy(horizon_s=10).decide(*fixture(), method='search_h10')
        self.assertEqual(payload['until_clock'], 140)
        self.assertEqual(diagnostics['horizon_s'], 10.)
        self.assertTrue(all(call.kwargs['horizon_s'] == 10 for call in calls.call_args_list))
        with self.assertRaises(ValueError):
            SearchAdmissionPolicy(horizon_s=10).decide(*fixture(), method='search_h5')

    def test_fresh_postplanner_snapshot_is_required(self):
        event, past, scenario, map_doc = fixture()
        event['boundary'] = 'before_planner'
        with self.assertRaises(ValueError):
            SearchAdmissionPolicy().decide(event, past, scenario, map_doc)

    def test_optional_history_is_checked_but_not_required(self):
        event, _, scenario, map_doc = fixture()
        with self.assertRaises(ValueError):
            SearchAdmissionPolicy().decide(event, np.zeros((1, 3, 3)), scenario, map_doc)
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5.})):
            SearchAdmissionPolicy().decide(event, None, scenario, map_doc)

    def test_invalid_parameters_and_duplicate_clock_rejected(self):
        for kwargs in ({'horizon_s': 2}, {'absolute_gain_s': -.1}, {'relative_gain_fraction': float('nan')}):
            with self.assertRaises(ValueError):
                SearchAdmissionPolicy(**kwargs)
        policy = SearchAdmissionPolicy()
        with patch('search_admission_policy.rollout', side_effect=scored_rollout({(): 5.})):
            policy.decide(*fixture())
            with self.assertRaises(ValueError):
                policy.decide(*fixture())

    def test_actual_rollout_prefers_no_hold_without_robot_interactions(self):
        # A real model integration check: delaying free motion cannot increase
        # this all-fleet progress proxy on these simple independent routes.
        payload, diagnostics = SearchAdmissionPolicy().decide(*fixture())
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(diagnostics['candidate_count'], 3)
        self.assertTrue(any(c['predicted_dispatch_block_events'] > 0 for c in diagnostics['candidates'][1:]))
        self.assertFalse(POLICY['task_completion_inferred'])


if __name__ == '__main__':
    unittest.main()
