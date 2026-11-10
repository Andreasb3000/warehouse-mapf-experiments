"""Causality, native-clock boundaries and fleet-count regression checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_fleet_predictors import (load_recording, prepare_query, target_for,
                                       validate_prediction, origins_for)
from fleet_execution import write_json


def event(clock, count=2):
    agents = [{'id': i, 'x': -(2 + i), 'y': -2., 'rz': 0.} for i in range(count)]
    return {'type': 'prediction_snapshot', 'clock': clock,
            'pose_frame': {'clock': clock, 'agents': agents},
            'snapshot': {'clock': clock, 'robots': [{'id': i} for i in range(count)]},
            'controllers': [{'id': i, 'clock': clock} for i in range(count)]}


def recording():
    shape = (8, 8)
    scenario = {'agents': [{'id': i, 'goals': [[6, 6], [7, 7]]} for i in range(2)]}
    states = np.array([[2., 2., 0.], [3., 2., 0.]])
    return {'directory': Path('synthetic'), 'frames': {t: states.copy() for t in range(400)},
            'snapshots': {120: event(120), 160: event(160)}, 'num_agents': 2,
            'scenario': scenario, 'map_doc': {'layout': ['.' * 8] * 8},
            'shape': shape, 'static': np.zeros((3, *shape))}


class FleetPredictorTests(unittest.TestCase):
    def test_origin_twelve_has_real_zero_frame_and_exact_current_pose(self):
        r = recording()
        r['frames'][120][:, :2] = 7  # Rounded archived frame is not the exact origin pose.
        q = prepare_query(r, event(120))
        self.assertEqual(q['past'].shape, (121, 2, 3))
        np.testing.assert_array_equal(q['past'][0], r['frames'][0])
        np.testing.assert_array_equal(q['past'][-1, :, :2], [[2, 2], [3, 2]])
        del r['frames'][0]
        with self.assertRaisesRegex(ValueError, 'Missing twelve-second'):
            prepare_query(r, event(120))

    def test_query_is_causal_and_owns_its_history(self):
        r = recording()
        q = prepare_query(r, event(120))
        saved = q['past'].copy()
        for t in r['frames']:
            r['frames'][t][:] = 0
        np.testing.assert_array_equal(q['past'], saved)
        self.assertNotIn('frames', q)
        self.assertNotIn('snapshots', q)
        self.assertNotIn('target', q)

    def test_future_mutations_change_labels_only(self):
        r = recording()
        q = prepare_query(r, event(120))
        original = target_for(r, q)
        history = q['past'].copy()
        for t in range(121, 321):
            r['frames'][t][:, 1] = 4
        changed = target_for(r, q)
        self.assertFalse(np.array_equal(original, changed))
        np.testing.assert_array_equal(q['past'], history)
        np.testing.assert_allclose(changed[:, 0].sum(axis=(1, 2)), 2)

    def test_target_uses_exact_origin_then_physical_future(self):
        r = recording()
        q = prepare_query(r, event(120))
        for t in range(120, 321):
            r['frames'][t][:, 1] = 4
        target = target_for(r, q)
        # [0,5) contains one exact origin frame at old column then49new ones.
        self.assertAlmostEqual(float(target[0, 0, 2, 2]), 1 / 50, places=6)
        self.assertAlmostEqual(float(target[0, 0, 2, 4]), 49 / 50, places=6)
        self.assertAlmostEqual(float(target[1, 0, 2, 4]), 1., places=6)
        self.assertAlmostEqual(float(target[2, 0, 2, 4]), 1., places=6)

    def test_missing_or_incoherent_physical_clock_is_rejected(self):
        r = recording()
        broken = event(120)
        broken['snapshot']['clock'] = 119
        with self.assertRaisesRegex(ValueError, 'clocks differ'):
            prepare_query(r, broken)
        del r['frames'][320]
        with self.assertRaisesRegex(ValueError, 'Incomplete future'):
            target_for(r, prepare_query(r, event(120)))

    def test_robot_mass_uses_actual_eighty_one_not_twenty_seven(self):
        q = {'shape': (10, 10), 'num_agents': 81}
        prediction = np.zeros((3, 2, 10, 10), dtype=np.float32)
        prediction[:, 0, :9, :9] = 1
        validate_prediction(prediction, q, 'joint')
        prediction[:, 0] /= 3
        with self.assertRaisesRegex(ValueError, 'actual N'):
            validate_prediction(prediction, q, 'joint')

    def test_loader_requires_native_zero_based_complete_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            write_json(p / 'config.json', {'parameters': {'num_agents': 2, 'ticks_per_second': 10,
                                                         'sim_duration': 3}})
            write_json(p / 'scenario.json', {'agents': [{'id': 0}, {'id': 1}],
                                              'safety': {'robot_trigger': 'outer_overlap'}})
            write_json(p / 'map.json', {'layout': ['..', '..']})
            frames = [{'type': 'tick', 'clock': t, 'agents': event(t)['pose_frame']['agents']}
                      for t in range(3)]
            fake_masks = (np.ones((2, 2), bool), np.ones((2, 2), bool), np.zeros((2, 2)))
            with patch('evaluate_fleet_predictors.objects', return_value=iter([*frames, event(2)])), \
                    patch('evaluate_fleet_predictors.masks', return_value=fake_masks):
                r = load_recording(p)
            self.assertEqual(sorted(r['frames']), [0, 1, 2])
            frames[0]['clock'] = 3
            with patch('evaluate_fleet_predictors.objects', return_value=iter([*frames, event(2)])), \
                    patch('evaluate_fleet_predictors.masks', return_value=fake_masks):
                with self.assertRaisesRegex(ValueError, 'frame clocks'):
                    load_recording(p)

    def test_origins_require_future_inside_native_recording(self):
        r = recording()
        self.assertEqual(origins_for(r, {'first_origin_s': 12, 'last_origin_s': 16,
                                         'origin_stride_s': 4}), [120, 160])
        r['snapshots'][240] = event(240)
        with self.assertRaisesRegex(ValueError, 'beyond recorded truth'):
            origins_for(r, {'first_origin_s': 24, 'last_origin_s': 24, 'origin_stride_s': 4})


if __name__ == '__main__':
    unittest.main()
