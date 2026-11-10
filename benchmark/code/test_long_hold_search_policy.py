"""Long-hold episodes must keep their modeled duration and mandatory release."""
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from long_hold_search_policy import LongHoldSearchPolicy
from test_search_admission_policy import advance, fixture, move, scored_rollout
from experiments.prediction.joint import rollout as actual_rollout


def crossing_fixture():
    """Prescribed surrogate paths, not a PBS-generated plan or native result."""
    event, past, scenario, map_doc = fixture()
    positions = [(26., 28.), (26., 29.), (26., 30.), (29.5, 26.)]
    robots, poses, controllers = [], [], []
    for rid, position in enumerate(positions):
        axis = 0 if rid < 3 else 1
        actions = []
        for step in range(20):
            start, goal = list(position), list(position)
            start[axis] += step * .5
            goal[axis] += (step + 1) * .5
            actions.append(move(step + 1, start, goal))
        robots.append(dict(id=rid, finished_node_id=0, enqueued_node_id=0,
                           committed_node_id=0, actions=actions))
        poses.append(dict(id=rid, x=-position[0], y=-position[1], rz=math.pi if axis == 0 else -math.pi / 2))
        controllers.append(dict(id=rid, clock=121, queue=[], poll_phase=0,
                                nominal_speed_m_s=1., speed_m_s=0., previous_move_speed_m_s=0.,
                                acceleration_m_s2=2.))
    event['snapshot']['robots'], event['pose_frame']['agents'], event['controllers'] = robots, poses, controllers
    scenario['agents'] = [dict(id=rid, goals=[[40., 40.]]) for rid in range(4)]
    scenario['safety'] = dict(enabled=True, robot_trigger='outer_overlap', inner_stop_enabled=False,
                              speed_scale=.2, robot_radius_m=.085)
    return event, past, scenario, map_doc


class LongHoldSearchTests(unittest.TestCase):
    def test_counterfactual_holds_six_seconds_but_native_request_expires_in_two(self):
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})) as calls:
            payload, detail = LongHoldSearchPolicy().decide(*fixture())
        self.assertEqual(payload['held_robot_ids'], [1])
        self.assertEqual(payload['until_clock'], 140)
        self.assertEqual(detail['active_until_clock'], 180)
        self.assertEqual(detail['forced_release_until_clock'], 200)
        self.assertTrue(detail['new_hold_started'])
        for call in calls.call_args_list:
            self.assertEqual(call.kwargs['horizon_s'], 10.)
            self.assertTrue(call.kwargs['native_poll_before_completion'])
            self.assertTrue(call.kwargs['record_progress'])
            if call.kwargs['admission']:
                self.assertEqual(call.kwargs['admission']['until_offset_s'], 6.)
                self.assertFalse(call.kwargs['admission']['protect_clearance'])
        json.dumps(detail, allow_nan=False)

    def test_carry_reviews_neither_search_nor_change_selected_ids(self):
        event, past, scenario, map_doc = fixture()
        policy = LongHoldSearchPolicy()
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})) as calls:
            policy.decide(event, past, scenario, map_doc)
            original_calls = calls.call_count
            for clock in (140, 160):
                payload, detail = policy.decide(advance(event, clock), past, scenario, map_doc)
                self.assertEqual(payload['held_robot_ids'], [1])
                self.assertEqual(payload['until_clock'], clock + 20)
                self.assertEqual(detail['candidate_count'], 0)
                self.assertEqual(detail['candidates'], [])
                self.assertTrue(detail['continuing_hold'])
                self.assertFalse(detail['new_hold_started'])
                self.assertIsNone(detail['best_gain_s'])
                self.assertAlmostEqual(detail['source_selected_gain_s'], 1.)
            self.assertEqual(calls.call_count, original_calls)

    def test_three_reviews_then_release_then_new_search(self):
        event, past, scenario, map_doc = fixture()
        policy = LongHoldSearchPolicy()
        held = []
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})) as calls:
            for clock in (120, 140, 160, 180, 200):
                payload, detail = policy.decide(advance(event, clock), past, scenario, map_doc)
                held.append(payload['held_robot_ids'])
                if clock == 180:
                    self.assertEqual(detail['selection_reason'], 'forced_release_after_hold')
                    self.assertEqual(detail['candidate_count'], 0)
                    self.assertEqual(detail['forced_release_robot_ids'], [1])
            self.assertEqual(calls.call_count, 6)  # Three branches per actual search.
        self.assertEqual(held, [[1], [1], [1], [], [1]])
        self.assertEqual(detail['search_call_count'], 2)
        self.assertEqual(detail['new_hold_count'], 2)
        self.assertEqual(detail['intervention_count'], 4)
        self.assertEqual(detail['choice_fraction'], .8)

    def test_new_commitments_are_logged_but_carry_does_not_override_native_guards(self):
        event, past, scenario, map_doc = fixture()
        policy = LongHoldSearchPolicy()
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})):
            policy.decide(event, past, scenario, map_doc)
            updated = advance(event, 140)
            updated['snapshot']['robots'][1]['committed_node_id'] = 2
            payload, detail = policy.decide(updated, past, scenario, map_doc)
        self.assertEqual(payload['held_robot_ids'], [1])
        self.assertFalse(payload['protect_clearance'])
        self.assertEqual(detail['selected_ids_now_inside_or_committed'], [1])
        self.assertIn('commitment promotions are not modeled', detail['warning'])

    def test_initial_search_excludes_committed_entry(self):
        event, past, scenario, map_doc = fixture()
        event['snapshot']['robots'][1]['committed_node_id'] = 2
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 9.})):
            payload, detail = LongHoldSearchPolicy().decide(event, past, scenario, map_doc)
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertTrue(all(1 not in candidate['held_robot_ids'] for candidate in detail['candidates']))

    def test_ties_and_gain_at_margin_stay_noop_without_starting_episode(self):
        for score in (5., 5.5):
            with self.subTest(score=score), patch('long_hold_search_policy.rollout',
                    side_effect=scored_rollout({(): 5., (1,): score})):
                payload, detail = LongHoldSearchPolicy().decide(*fixture())
                self.assertEqual(payload['held_robot_ids'], [])
                self.assertFalse(detail['new_hold_started'])
                self.assertIsNone(detail['active_until_clock'])

    def test_noop_search_is_repeated_at_next_review(self):
        event, past, scenario, map_doc = fixture()
        policy = LongHoldSearchPolicy()
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5.})) as calls:
            policy.decide(event, past, scenario, map_doc)
            _, detail = policy.decide(advance(event, 140), past, scenario, map_doc)
        self.assertEqual(calls.call_count, 6)
        self.assertEqual(detail['search_call_count'], 2)
        self.assertEqual(detail['new_hold_count'], 0)

    def test_missed_review_does_not_extend_six_second_episode(self):
        event, past, scenario, map_doc = fixture()
        policy = LongHoldSearchPolicy()
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})):
            policy.decide(event, past, scenario, map_doc)
            _, carry = policy.decide(advance(event, 160), past, scenario, map_doc)
            payload, release = policy.decide(advance(event, 180), past, scenario, map_doc)
        self.assertEqual(carry['active_until_clock'], 180)
        self.assertEqual(payload['held_robot_ids'], [])
        self.assertEqual(release['selection_reason'], 'forced_release_after_hold')

    def test_gap_past_release_window_allows_new_search(self):
        event, past, scenario, map_doc = fixture()
        policy = LongHoldSearchPolicy()
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5., (1,): 6.})):
            policy.decide(event, past, scenario, map_doc)
            _, detail = policy.decide(advance(event, 220), past, scenario, map_doc)
        self.assertTrue(detail['new_hold_started'])
        self.assertEqual(detail['active_until_clock'], 280)

    def test_invalid_horizon_method_or_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            LongHoldSearchPolicy(horizon_s=5.)
        with self.assertRaises(ValueError):
            LongHoldSearchPolicy().decide(*fixture(), method='search_h10')
        policy = LongHoldSearchPolicy()
        with patch('long_hold_search_policy.rollout', side_effect=scored_rollout({(): 5.})):
            policy.decide(*fixture())
            with self.assertRaises(ValueError):
                policy.decide(*fixture())

    def test_actual_safety_rollout_can_select_beneficial_crossing_delay(self):
        predictions = {}

        def record(*args, **kwargs):
            result = actual_rollout(*args, **kwargs)
            selected = tuple(kwargs['admission']['held_robot_ids']) if kwargs['admission'] else ()
            predictions[selected] = result
            return result

        with patch('long_hold_search_policy.rollout', side_effect=record):
            payload, detail = LongHoldSearchPolicy().decide(*crossing_fixture())
        self.assertEqual(payload['held_robot_ids'], [3])
        self.assertGreater(detail['selected_gain_s'], .5)
        self.assertGreater(predictions[()]['diagnostics']['slow_move_robot_seconds'], 0.)
        self.assertGreater(predictions[(3,)]['diagnostics']['admission']['dispatch_block_events'], 0)
        # The gain should not rely on overlapping robot bodies in this sampled
        # kinematic fixture. Native contact physics is still outside this test.
        for key in ((), (3,)):
            states = np.array([[[a['row'], a['col']] for a in frame['agents']]
                               for frame in predictions[key]['frames']])
            clearance = min(np.linalg.norm(states[:, i] - states[:, j], axis=1).min()
                            for i in range(4) for j in range(i + 1, 4))
            self.assertGreater(clearance, 2 * .085)


if __name__ == '__main__':
    unittest.main()
