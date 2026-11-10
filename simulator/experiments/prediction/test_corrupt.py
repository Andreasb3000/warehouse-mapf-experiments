"""Causal corruption, dependency propagation, and missing-output score checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.prediction.corrupt import PROFILES, analyze, forecast_at_origin
from experiments.prediction.predict import predict_snapshot
from experiments.prediction.test_predict import action, frame, robot, snapshot


class DurationScaleTests(unittest.TestCase):
    def test_slow_robot_delays_dependent_robot(self):
        state = snapshot([
            robot(0, [action(0, [0, 0], [0, 1])]),
            robot(1, [action(0, [1, 0], [1, 1], dependencies=[[0, 0]])]),
        ])
        predicted, _ = predict_snapshot(state, frame(0, [(0, 0), (1, 0)]),
                                        [1.5, 2.5], duration_scales={0: 2})
        self.assertEqual(predicted[1.5][0], (0, .75))
        self.assertEqual(predicted[1.5][1], (1, 0))
        self.assertEqual(predicted[2.5][1], (1, .5))

    def test_turn_station_and_movement_all_scale(self):
        state = snapshot([robot(0, [action(0, [0, 0], [0, 0], kind="T"),
                                    action(1, [0, 0], [0, 0], kind="S"),
                                    action(2, [0, 0], [0, 2])])])
        prediction, _ = predict_snapshot(state, frame(0, [(0, 0)]), [5, 7],
                                         duration_scales={0: 2}, station_dwell_s=2)
        self.assertEqual(prediction[5][0], (0, 0))
        self.assertEqual(prediction[7][0], (0, .5))

    def test_invalid_scales_rejected(self):
        for scale in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                predict_snapshot(snapshot([robot(0, [])]), frame(0, [(0, 0)]),
                                 [1], duration_scales={0: scale})


class CausalCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.first = snapshot([robot(0, [action(0, [0, 0], [0, 10])])], clock=0)
        self.later = snapshot([robot(0, [action(0, [0, 3], [10, 3])])], clock=3, version=2)
        self.frames = {i: frame(i, [(0, i)]) for i in range(7)}

    def test_clean_matches_existing_predictor(self):
        result = forecast_at_origin([self.first], self.frames, 0, [1, 2],
                                    profile=PROFILES["clean"], ticks_per_second=1)
        expected, diagnostics = predict_snapshot(self.first, self.frames[0], [1, 2])
        self.assertEqual(result["predictions"], expected)
        self.assertEqual(result["diagnostics"], diagnostics)
        self.assertTrue(result["delivered"][0])

    def test_delayed_plan_rolls_forward_age_plus_horizon_and_excludes_new_inputs(self):
        profile = {"duration_bias": 0, "observation_delay_s": 2, "delivery_dropout": 0}
        result = forecast_at_origin([self.first, self.later], self.frames, 3, [1],
                                    profile=profile, ticks_per_second=1)
        self.assertEqual(result["snapshot_clock"], 0)
        self.assertEqual(result["pose_clock"], 0)
        self.assertEqual(result["predictions"][1][0], (0, 4))
        altered = deepcopy(self.frames)
        for clock in range(1, 7):
            altered[clock] = frame(clock, [(999, 999)])
        future = deepcopy(self.later)
        future["robots"][0]["actions"] = [action(0, [999, 999], [-999, -999])]
        repeated = forecast_at_origin([self.first, future], altered, 3, [1],
                                      profile=profile, ticks_per_second=1)
        self.assertEqual(result, repeated)

    def test_duration_bias_persists_and_seed_is_reproducible(self):
        profile = {"duration_bias": .4, "observation_delay_s": 0, "delivery_dropout": .2}
        a = forecast_at_origin([self.first], self.frames, 0, [1, 2], profile=profile,
                               ticks_per_second=1, seed=123)
        b = forecast_at_origin([self.first, self.later], self.frames, 3, [1], profile=profile,
                               ticks_per_second=1, seed=123)
        self.assertEqual(a["duration_scales"], b["duration_scales"])
        self.assertTrue(.6 <= a["duration_scales"][0] <= 1.4)
        self.assertEqual(a, forecast_at_origin([self.first], self.frames, 0, [1, 2],
                                               profile=profile, ticks_per_second=1, seed=123))
        c = forecast_at_origin([self.first], self.frames, 0, [1, 2], profile=profile,
                               ticks_per_second=1, seed=124)
        self.assertNotEqual(a["duration_scales"], c["duration_scales"])

    def test_dropout_never_removes_robots_from_dependency_rollout(self):
        profile = {"duration_bias": 0, "observation_delay_s": 0, "delivery_dropout": 1}
        result = forecast_at_origin([self.first], self.frames, 0, [1], profile=profile,
                                    ticks_per_second=1)
        self.assertFalse(result["delivered"][0])
        self.assertEqual(result["predictions"][1][0], (0, 1))


class CorruptionEvaluationTests(unittest.TestCase):
    def recording(self, root, events):
        run = root / "run-01"
        run.mkdir(exist_ok=True)
        (root / "config.json").write_text(json.dumps({
            "scenario": "scenario.json", "parameters": {"ticks_per_second": 1, "velocity": 100}}))
        (root / "scenario.json").write_text(json.dumps({
            "name": "test", "zones": [{"id": "Z0", "bounds": {"rows": [0, 2], "cols": [1, 1]}}]}))
        (run / "simulator.log").write_text("\n".join(json.dumps(e) for e in events) + "\n")
        return run

    def events(self):
        return [snapshot([robot(0, [action(0, [0, 0], [0, 2])]), robot(1, [])]),
                frame(0, [(0, 0), (2, 0)]), frame(1, [(0, 1), (2, 0)]), frame(2, [(0, 2), (2, 0)])]

    def test_total_dropout_has_zero_coverage_null_errors_and_nonempty_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.recording(Path(directory), self.events())
            result = analyze(run, profiles=["missing"], horizons=[1], profile_settings={
                "missing": {"duration_bias": 0, "observation_delay_s": 0, "delivery_dropout": 1}})
            score = result["profiles"]["missing"]["metrics"][0]
            self.assertEqual(score["coverage"], 0)
            self.assertIsNone(score["endpoint_mean_error_m"])
            self.assertIsNone(score["hold_endpoint_mean_error_m"])
            self.assertIsNone(score["zones"]["Z0"]["occupancy_mae_robots"])
            output = json.loads((run / "corrupted_forecasts.jsonl").read_text())
            self.assertEqual(output["zones"]["Z0"]["count_range"], [0, 2])
            self.assertTrue(all(p["row"] is None and p["col"] is None for p in output["robots"]))

    def test_clean_and_hold_errors_use_known_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.recording(Path(directory), self.events())
            result = analyze(run, profiles=["clean"], horizons=[1])
            score = result["profiles"]["clean"]["metrics"][0]
            self.assertEqual(score["endpoint_mean_error_m"], 0)
            self.assertEqual(score["hold_endpoint_mean_error_m"], .5)
            self.assertEqual(score["moving_robot_endpoints"], 1)
            self.assertEqual(score["hold_moving_endpoint_mean_error_m"], 1)
            self.assertEqual(score["zones"]["Z0"]["occupancy_mae_robots"], 0)

    def test_hold_and_clean_comparisons_share_delivered_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.recording(Path(directory), self.events())
            result = analyze(run, profiles=["masked"], horizons=[1], profile_settings={
                "masked": {"duration_bias": .2, "observation_delay_s": 0, "delivery_dropout": .5}})
            output = json.loads((run / "corrupted_forecasts.jsonl").read_text())
            delivered = [p for p in output["robots"] if p["available"]]
            score = result["profiles"]["masked"]["metrics"][0]
            expected = sum(1 if p["id"] == 0 else 0 for p in delivered) / len(delivered)
            self.assertEqual(score["hold_endpoint_mean_error_m"], expected)
            self.assertEqual(score["matched_clean_endpoint_mean_error_m"], 0)
            self.assertEqual(score["delivered_robot_endpoints"], len(delivered))

    def test_future_truth_changes_scores_but_not_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = self.events()
            run = self.recording(root, events)
            a = analyze(run, profiles=["clean"], horizons=[1])
            forecasts = (run / "corrupted_forecasts.jsonl").read_text()
            events[2] = frame(1, [(8, 8), (8, 9)])
            self.recording(root, events)
            b = analyze(run, profiles=["clean"], horizons=[1])
            self.assertEqual(forecasts, (run / "corrupted_forecasts.jsonl").read_text())
            self.assertNotEqual(a["profiles"]["clean"]["metrics"], b["profiles"]["clean"]["metrics"])


if __name__ == "__main__":
    unittest.main()
