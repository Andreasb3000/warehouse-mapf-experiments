"""Meaningful scoring, censoring, and future-data isolation regressions."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.prediction import joint_evaluate as evaluate
from experiments.prediction.flow_metrics import queue_samples


def agent(ident, row, col):
    return {"id": ident, "row": row, "col": col, "yaw": 0}


def frame(*agents, offset=0):
    return {"offset_s": offset, "agents": list(agents)}


def visit(entry, exit_time, inside=False):
    return {"entry_s": entry, "exit_s": exit_time, "already_inside_at_origin": inside}


def queue(approach_id="a", count=0, tail=0, spillback=False):
    return {"approaches": [{"approach_id": approach_id, "queued_robots": count,
                            "queue_tail_extent_m": tail, "spillback_warning": spillback}]}


class EndpointScoreTests(unittest.TestCase):
    def test_robot_ids_match_not_list_order_and_moving_cohort_uses_truth(self):
        scores = evaluate.empty_scores()
        initial = frame(agent(0, 0, 0), agent(1, 3, 3), agent(2, 6, 6))
        truth = frame(agent(0, 1, 0), agent(1, 3, 3), agent(2, 6.25, 6))
        predicted = frame(agent(2, 6.75, 6), agent(1, 7, 3), agent(0, 1.2, 0))
        evaluate.score_endpoint(scores, predicted, truth, initial, [])
        result = evaluate.finalize_scores(scores)
        self.assertEqual(result["robot_endpoints"], 3)
        # A bad moving prediction for actually-stationary robot 1 must not add it
        # to the moving cohort. Exactly 0.25 m displacement is also excluded.
        self.assertEqual(result["moving_robot_endpoints"], 1)
        self.assertAlmostEqual(result["moving_position_mae_m"], .2)
        self.assertAlmostEqual(result["position_mae_m"], (.5 + 4 + .2)/3)
        self.assertAlmostEqual(result["position_p95_m"], 4)
        with self.assertRaisesRegex(ValueError, "IDs"):
            evaluate.score_endpoint(evaluate.empty_scores(), frame(agent(9, 1, 0)), truth, initial, [])

    def test_occupancy_and_queue_classification_are_per_approach(self):
        scores = evaluate.empty_scores()
        zones = [{"id": "j", "conflict_zone": {"rows": [4, 4], "cols": [4, 4]}}]
        actual = frame(agent(0, 4, 4))
        predicted = frame(agent(0, 5, 4))
        evaluate.score_endpoint(scores, predicted, actual, actual, zones,
                                queue(count=1, tail=1), queue(count=3, tail=4, spillback=True))
        result = evaluate.finalize_scores(scores)
        self.assertEqual(result["junction_occupancy_mae_robots"], 1)
        self.assertEqual(result["approach_queue_count_mae_robots"], 2)
        self.assertEqual(result["approach_queue_tail_mae_m"], 3)
        self.assertEqual(result["multi_robot_queue"]["fn"], 1)
        self.assertEqual(result["multi_robot_queue"]["recall"], 0)
        self.assertIsNone(result["multi_robot_queue"]["precision"])
        self.assertEqual(result["spillback_warning"]["fn"], 1)
        with self.assertRaisesRegex(ValueError, "approach"):
            evaluate.score_endpoint(evaluate.empty_scores(), predicted, actual, actual, [],
                                    queue("a"), queue("b"))

    def test_duplicate_robot_ids_and_missing_approaches_reject_before_mutating_scores(self):
        actual = frame(agent(0, 4, 4))
        for predicted, p_queue, t_queue in (
            (frame(agent(0, 4, 4), agent(0, 4, 4)), None, None),
            (actual, {"approaches": []}, queue()),
            (actual, {"approaches": queue()["approaches"]*2}, queue()),
        ):
            scores = evaluate.empty_scores()
            pristine = deepcopy(scores)
            with self.assertRaisesRegex(ValueError, "IDs"):
                evaluate.score_endpoint(scores, predicted, actual, actual, [], p_queue, t_queue)
            self.assertEqual(scores, pristine)

    def test_queue_approaches_match_by_id_when_order_changes(self):
        actual = frame(agent(0, 4, 4))
        truth = {"approaches": queue("a", count=2)["approaches"] + queue("b", count=0)["approaches"]}
        prediction = {"approaches": list(reversed(truth["approaches"]))}
        scores = evaluate.empty_scores()
        evaluate.score_endpoint(scores, actual, actual, actual, [], prediction, truth)
        self.assertEqual(scores["queue_errors"], [0, 0])
        self.assertEqual(scores["queue_tp"], 1)
        self.assertEqual(scores["queue_tn"], 1)

    def test_first_two_seconds_of_queue_history_are_unknown_not_negative(self):
        scenario = {"agents": [{"id": 0, "goals": []}]}
        frames = [frame(agent(0, 1, 1), offset=k*.1) for k in range(21)]
        measured = queue_samples(frames, [], scenario)
        self.assertTrue(all(value is None for value in measured[:20]))
        self.assertIsNotNone(measured[20])
        scores = evaluate.empty_scores()
        for i in (0, 10, 19):
            evaluate.score_endpoint(scores, frames[i], frames[i], frames[0], [],
                                    measured[i], measured[i])
        result = evaluate.finalize_scores(scores)
        self.assertEqual(result["robot_endpoints"], 3)
        self.assertEqual(sum(result["multi_robot_queue"][key] for key in ("tp", "tn", "fp", "fn")), 0)
        self.assertIsNone(result["approach_queue_count_mae_robots"])
        self.assertIsNone(result["multi_robot_queue"]["recall"])


class PlanCoverageTests(unittest.TestCase):
    def test_horizon_boundary_and_no_exhaustion_use_only_origin_rollout_diagnostics(self):
        diagnostics = {"plan_exhausted_at_s": {"0": None, "1": 0, "2": 3, "3": 4}}
        self.assertEqual(evaluate.unexhausted_plan_ids(diagnostics, [0, 1, 2, 3], 3), {0, 2, 3})
        self.assertEqual(evaluate.unexhausted_plan_ids(diagnostics, [0, 1, 2, 3], 5), {0})
        with self.assertRaisesRegex(ValueError, "every robot"):
            evaluate.unexhausted_plan_ids(diagnostics, [0, 1], 3)
        diagnostics["plan_exhausted_at_s"]["1"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            evaluate.unexhausted_plan_ids(diagnostics, [0, 1, 2, 3], 3)

    def test_same_coverage_cohort_keeps_all_robot_error_and_scores_every_method(self):
        initial = frame(agent(0, 0, 0), agent(1, 0, 1))
        truth = frame(agent(0, 2, 0), agent(1, 2, 1))
        covered = evaluate.unexhausted_plan_ids({"plan_exhausted_at_s": {"0": None, "1": .5}}, [0, 1], 2)
        predictions = [frame(agent(0, 2.1, 0), agent(1, 0, 1)),
                       frame(agent(0, 2.4, 0), agent(1, 1, 1))]
        for prediction, covered_error, exhausted_error in zip(predictions, [.1, .4], [2, 1]):
            scores = evaluate.empty_scores()
            evaluate.score_endpoint(scores, prediction, truth, initial, [], unexhausted_ids=covered)
            result = evaluate.finalize_scores(scores)
            self.assertEqual(result["robot_endpoints"], 2)
            self.assertEqual(result["unexhausted_plan_robot_endpoints"], 1)
            self.assertEqual(result["exhausted_plan_robot_endpoints"], 1)
            self.assertAlmostEqual(result["unexhausted_plan_position_mae_m"], covered_error)
            self.assertAlmostEqual(result["exhausted_plan_position_mae_m"], exhausted_error)
            self.assertAlmostEqual(result["position_mae_m"], (covered_error+exhausted_error)/2)


class CrossingScoreTests(unittest.TestCase):
    def test_missing_unfinished_and_false_clearances_are_counted(self):
        truth = {(0, "j"): visit(2, 4), (1, "j"): visit(3, 8),
                 (2, "j"): visit(5, None), (3, "j"): visit(1, 9)}
        predicted = {(0, "j"): visit(3, 6), (1, "j"): visit(2, None),
                     (2, "j"): visit(4, 7), (4, "j"): visit(3, 5)}
        scored = evaluate.score_crossings(predicted, truth)
        self.assertEqual(scored["actual_clearances"], 3)
        self.assertEqual(scored["predicted_clearances"], 3)
        self.assertEqual(scored["matched_clearances"], 1)
        # One unfinished forecast and one wholly absent forecast are both misses.
        self.assertEqual(scored["missed_clearances"], 2)
        # One unfinished truth visit and one wholly absent truth visit are both false exits.
        self.assertEqual(scored["false_clearances"], 2)
        self.assertEqual(scored["actual_censored_visits"], 1)
        self.assertEqual(scored["predicted_censored_visits"], 1)
        self.assertEqual(scored["clearance_errors_s"], [2])
        self.assertEqual(scored["arrival_errors_s"], [1])

    def test_initial_occupants_keep_clearance_error_but_not_zero_arrival_error(self):
        truth = {(0, "j"): visit(0, 3, True), (1, "j"): visit(2, 4)}
        prediction = {(0, "j"): visit(0, 5, True), (1, "j"): visit(3, 6)}
        scored = evaluate.score_crossings(prediction, truth)
        self.assertEqual(scored["clearance_errors_s"], [2, 2])
        self.assertEqual(scored["arrival_errors_s"], [1])
        self.assertEqual(scored["matched_clearances"], 2)
        self.assertEqual(scored["matched_arrivals"], 1)
        self.assertEqual(scored["initially_inside_matched_clearances"], 1)

    def test_no_paired_clearances_has_no_fabricated_error(self):
        scored = evaluate.score_crossings({(0, "j"): visit(0, None, True)},
                                          {(0, "j"): visit(0, 3, True)})
        self.assertEqual(scored["missed_clearances"], 1)
        self.assertEqual(scored["matched_clearances"], 0)
        self.assertEqual(scored["clearance_errors_s"], [])
        self.assertIsNone(evaluate.average(scored["clearance_errors_s"]))


class EvaluationIsolationTests(unittest.TestCase):
    def test_unannotated_or_unmeasurable_maps_fail_before_forecasting_or_writing(self):
        config = {"parameters": {"ticks_per_second": 10, "velocity": 100, "seed": 42}}
        cases = [
            ({"name": "unannotated", "agents": []}, {"layout": ["....."]*5},
             "requires junction metadata"),
            ({"name": "missing-links", "agents": [], "zones": [
                {"id": "a", "center": [1, 1], "conflict_zone": {"rows": [1, 1], "cols": [1, 1]}},
                {"id": "b", "center": [3, 1], "conflict_zone": {"rows": [3, 3], "cols": [1, 1]}},
             ]}, {"layout": ["....."]*5}, "add scenario.network.edges"),
            ({"name": "enclosed", "agents": [],
              "conflict_zone": {"rows": [1, 1], "cols": [1, 1]}},
             {"layout": ["@@@", "@.@", "@@@"]}, "traversable cardinal approaches"),
        ]
        for scenario, map_doc, message in cases:
            with self.subTest(scenario=scenario["name"]), tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)
                with patch.object(evaluate, "read_run", return_value=(config, scenario, map_doc, [], {}, [])), \
                     patch.object(evaluate, "rollout") as rollout:
                    with self.assertRaisesRegex(ValueError, message):
                        evaluate.evaluate_run(out, out)
                    rollout.assert_not_called()
                self.assertEqual(list(out.iterdir()), [])

    def test_forecasts_finish_before_any_future_pose_is_read(self):
        completed_predictors = []
        clock = 10
        pose = {"clock": clock, "agents": [{"id": 0, "x": -1, "y": -1, "rz": 0}]}
        snapshot = {"clock": clock, "plan_version": 3, "robots": [{"id": 0, "actions": []}]}
        controllers = [{"id": 0, "clock": clock, "queue": []}]
        event = {"type": "prediction_snapshot", "clock": clock, "pose_frame": pose,
                 "snapshot": snapshot, "controllers": controllers}
        pristine = deepcopy(event)
        predicted_frames = [frame(agent(0, 1, 1+k*.05), offset=k*.1) for k in range(21)]

        class GuardedTruth(dict):
            def __getitem__(self, key):
                if key > clock and len(completed_predictors) != 4:
                    raise AssertionError("Future pose accessed before every forecast finished")
                return super().__getitem__(key)

        future = GuardedTruth({clock+k: {"clock": clock+k, "agents": [
            {"id": 0, "x": -1, "y": -(1+k*.1), "rz": 0}]} for k in range(21)})
        scenario = {"name": "isolation-test", "agents": [{"id": 0, "goals": []}],
                    "conflict_zone": {"rows": [2, 2], "cols": [2, 2]},
                    "safety": {"robot_trigger": "body_in_outer"}}
        config = {"parameters": {"ticks_per_second": 10, "velocity": 100, "seed": 42}}
        map_doc = {"layout": ["....."]*5}

        def fake_rollout(state, observed, settings, layout, **kwargs):
            self.assertEqual(state, snapshot)
            self.assertEqual(observed, pose)
            self.assertEqual(kwargs["controllers"], controllers)
            self.assertEqual(observed["clock"], clock)
            self.assertNotIn("future", kwargs)
            completed_predictors.append("rollout")
            return {"frames": deepcopy(predicted_frames), "diagnostics": {"plan_exhausted_at_s": {"0": None}}}

        def fake_legacy(state, observed, horizons, **kwargs):
            self.assertEqual(state, snapshot)
            self.assertEqual(observed, pose)
            completed_predictors.append("legacy")
            return {2.0: {0: [1, 2]}}, {}

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with patch.object(evaluate, "read_run", return_value=(config, scenario, map_doc, [event], future, [clock, clock+5])), \
                 patch.object(evaluate, "rollout", side_effect=fake_rollout), \
                 patch.object(evaluate, "predict_snapshot", side_effect=fake_legacy):
                result, _, _ = evaluate.evaluate_run(out, out, horizons=(2,), origin_stride_s=1)
            saved = json.loads((out/"forecasts.jsonl").read_text())
        self.assertEqual(event, pristine)
        self.assertEqual(completed_predictors, ["rollout", "rollout", "rollout", "legacy"])
        self.assertEqual(result["forecast_origins"], 1)
        self.assertEqual(result["origins_with_later_plan"], 1)
        self.assertTrue(all(value["unexhausted_plan_robot_endpoints"] == 1 for value in result["metrics"]["2.0"].values()))
        self.assertAlmostEqual(result["metrics"]["2.0"]["joint"]["position_mae_m"], 1)
        self.assertEqual(saved["horizons"]["2.0"]["forecasts"]["joint"][0]["col"], 2)
        self.assertEqual(saved["horizons"]["2.0"]["actual"][0]["col"], 3)


if __name__ == "__main__":
    unittest.main()
