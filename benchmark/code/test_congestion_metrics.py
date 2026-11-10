import unittest
import numpy as np
from congestion_metrics import SCREEN, episodes, graph_distance, bypass_info, stationary_flags, safety_exposure, difficulty_screen, central_arrivals
from features import rasterize


def tiny_scenario():
    centers = {'A': [10, 10], 'C': [10, 20], 'B': [10, 30], 'D': [0, 20]}
    return {'zones': [{'id': key, 'center': p, 'conflict_zone': {'rows': [p[0]-1, p[0]+1], 'cols': [p[1]-1, p[1]+1]}} for key, p in centers.items()],
            'network': {'edges': [['A', 'C'], ['C', 'B'], ['A', 'D'], ['D', 'B']]},
            'agents': [{'id': 0, 'zone_id': 'A', 'destination_zone_id': 'B', 'goals': [[40, 40], [41, 41]]}]}


class CongestionMetricsTests(unittest.TestCase):
    def test_episode_clips_warmup_and_keeps_sample_duration(self):
        result = episodes(np.ones(50, bool), np.arange(50)*.1, .1, start_s=2)
        self.assertEqual(len(result), 1)
        for key, value in [('duration_s', 3), ('start_s', 2), ('end_s', 5)]: self.assertAlmostEqual(result[0][key], value)

    def test_creeping_is_slow_but_not_stationary(self):
        states = np.zeros((80, 1, 3)); states[:, 0, 0] = 20-np.arange(80)*.02; states[:, 0, 1] = 20
        scenario = tiny_scenario(); active = np.ones((80, 1), bool)
        _, slow = rasterize(states, scenario, (50, 50), .1)
        self.assertTrue(slow[20:].all()); self.assertFalse(stationary_flags(states, scenario, active, .1).any())

    def test_turning_and_service_are_not_stationary_congestion(self):
        scenario = tiny_scenario(); states = np.zeros((40, 1, 3)); states[:, 0, :2] = [20, 20]; states[:, 0, 2] = np.arange(40)*.02
        self.assertFalse(stationary_flags(states, scenario, np.ones((40, 1), bool), .1).any())
        states[:, 0, :2] = [40, 40]; states[:, 0, 2] = 0
        self.assertFalse(stationary_flags(states, scenario, np.ones((40, 1), bool), .1).any())

    def test_unreleased_or_completed_robot_cannot_count_stationary(self):
        states = np.zeros((40, 1, 3)); states[:, 0, :2] = [20, 20]
        active = np.zeros((40, 1), bool); active[20:25] = True
        self.assertEqual(stationary_flags(states, tiny_scenario(), active, .1).sum(), 5)

    def test_bypass_requires_connectivity_not_just_multiple_neighbors(self):
        scenario = tiny_scenario(); info = bypass_info(scenario, 'A', 'B', 'C')
        self.assertTrue(info['geometric_bypass_available']); self.assertTrue(info['central_is_on_a_shortest_zone_path']); self.assertGreater(info['bypass_extra_vs_shortest_m'], 0)
        scenario['network']['edges'].remove(['D', 'B'])
        self.assertIsNone(graph_distance(scenario, 'A', 'B', forbidden='C'))

    def test_selected_safety_blocker_durations_not_transition_counts(self):
        events = [dict(type='safety_state', agent_id=0, clock=t, mode=m, blocker_kind=k) for t,m,k in [(10, 'slow', 'robot'), (20, 'slow', 'box'), (40, 'clear', '')]]
        active = np.ones((50, 1), bool); result, _ = safety_exposure(events, [0], active, active.copy(), .1)
        self.assertAlmostEqual(result['by_selected_blocker']['robot']['active_robot_seconds'], 1)
        self.assertAlmostEqual(result['by_selected_blocker']['obstacle']['coincident_slow_robot_seconds'], 2)
        self.assertAlmostEqual(result['active_slow_mode_robot_seconds'], 3); self.assertIsNone(result['native_whole_recording_counters'])

    def test_unfinished_and_unobserved_diversion_fail_feasible_screen(self):
        summary = {'task_completion_fraction': 1, 'native_plan_failures': 0, 'central_arrivals': {'post_warmup_observed_geometric_bypass_opportunities': 1}}
        for key in ('stationary_queue', 'slow_queue', 'moving_slow_queue'): summary[key] = {'central_longest_threshold_episode_after_warmup_s': 4}
        self.assertTrue(difficulty_screen(summary, SCREEN)['candidate_for_slow_congestion_study'])
        summary['task_completion_fraction'] = .9
        self.assertFalse(difficulty_screen(summary, SCREEN)['candidate_for_slow_congestion_study'])
        summary['task_completion_fraction'] = 1; summary['central_arrivals']['post_warmup_observed_geometric_bypass_opportunities'] = 0
        self.assertFalse(difficulty_screen(summary, SCREEN)['candidate_for_slow_congestion_study'])

    def test_central_arrival_checks_observed_branch_after_warmup(self):
        scenario = tiny_scenario(); states = np.zeros((30, 1, 3)); states[:15, 0, :2] = [10, 10]; states[15:20, 0, :2] = [10, 15]; states[20:, 0, :2] = [10, 20]
        tasks = [{'robot_id': 0, 'source_zone_id': 'A', 'destination_zone_id': 'B'}]; screen = {**SCREEN, 'central_zone_id': 'C', 'warmup_s': 12}
        result, _ = central_arrivals(states, np.ones((30, 1), bool), [0], scenario, tasks, np.arange(30), screen)
        self.assertEqual(result['first_entry_s'], 20); self.assertEqual(result['post_warmup_observed_geometric_bypass_opportunities'], 1)
        screen['warmup_s'] = 16
        result, _ = central_arrivals(states, np.ones((30, 1), bool), [0], scenario, tasks, np.arange(30), screen)
        self.assertEqual(result['post_warmup_observed_geometric_bypass_opportunities'], 0)


if __name__ == '__main__': unittest.main()
