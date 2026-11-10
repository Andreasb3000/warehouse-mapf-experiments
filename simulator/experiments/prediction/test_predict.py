"""Behavioral checks for the approximate forecast and causal evaluation path."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .predict import analyze, predict_snapshot
except ImportError:
    from predict import analyze, predict_snapshot


def action(node_id, start, goal, *, kind="M", orientation=1, dependencies=()):
    return {"node_id": node_id, "type": kind, "orientation": orientation,
            "start": start, "goal": goal, "task_id": -1,
            "dependencies": list(dependencies)}


def robot(robot_id, actions, *, finished=-1, enqueued=-1, committed=-1):
    return {"id": robot_id, "finished_node_id": finished,
            "enqueued_node_id": enqueued, "committed_node_id": committed,
            "actions": actions}


def snapshot(robots, *, clock=0, version=1):
    return {"type": "plan_snapshot", "schema_version": 1, "clock": clock,
            "plan_version": version, "planner_source": "primary", "robots": robots}


def frame(clock, positions):
    return {"type": "tick", "clock": clock,
            "agents": [{"id": robot_id, "x": -row, "y": -col, "rz": 0}
                       for robot_id, (row, col) in enumerate(positions)]}


class FrozenActionTests(unittest.TestCase):
    def test_dependencies_delay_other_robot(self):
        state = snapshot([
            robot(0, [action(0, [0, 0], [0, 1])]),
            robot(1, [action(0, [1, 0], [1, 1], dependencies=[[0, 0]])]),
        ])
        forecast, _ = predict_snapshot(state, frame(0, [(0, 0), (1, 0)]), [.5, 1.5])
        self.assertEqual(forecast[.5][0], (0, .5))
        self.assertEqual(forecast[.5][1], (1, 0))
        self.assertEqual(forecast[1.5][1], (1, .5))

    def test_turns_consume_time_without_translation(self):
        state = snapshot([robot(0, [
            action(0, [0, 0], [0, 0], kind="T", orientation=1),
            action(1, [0, 0], [0, 1]),
        ])])
        forecast, _ = predict_snapshot(state, frame(0, [(0, 0)]), [.5, 1.5, 10])
        self.assertEqual(forecast[.5][0], (0, 0))
        self.assertEqual(forecast[1.5][0], (0, .5))
        self.assertEqual(forecast[10][0], (0, 1))

    def test_empty_plan_stays_at_actual_pose(self):
        forecast, _ = predict_snapshot(snapshot([robot(0, [])]),
                                       frame(0, [(3.2, 4.6)]), [1, 10])
        self.assertEqual(forecast[1][0], (3.2, 4.6))
        self.assertEqual(forecast[10][0], (3.2, 4.6))

    def test_dispatch_releases_dependencies_but_commitment_does_not(self):
        blocked = action(0, [0, 0], [0, 1], dependencies=[[7, 99]])
        state = snapshot([robot(0, [blocked], committed=0)])
        forecast, diagnostics = predict_snapshot(state, frame(0, [(0, 0)]), [1])
        self.assertEqual(forecast[1][0], (0, 0))
        self.assertEqual(diagnostics["unresolved_action_count"], 1)
        state["robots"][0]["enqueued_node_id"] = 0
        forecast, diagnostics = predict_snapshot(state, frame(0, [(0, 0)]), [1])
        self.assertEqual(forecast[1][0], (0, 1))
        self.assertEqual(diagnostics["unresolved_action_count"], 0)

    def test_already_passed_dispatched_half_move_does_not_reverse(self):
        state = snapshot([robot(0, [action(0, [0, 0], [0, .5]),
                                    action(1, [0, .5], [0, 1])], enqueued=1)])
        forecast, _ = predict_snapshot(state, frame(0, [(0, .7)]), [.1])
        self.assertAlmostEqual(forecast[.1][0][1], .8)

    def test_station_wait_holds_pose_and_delays_dependents(self):
        state = snapshot([
            robot(0, [action(0, [0, .5], [0, 1], kind="S")]),
            robot(1, [action(0, [1, 0], [1, 1], dependencies=[[0, 0]])]),
        ])
        forecast, _ = predict_snapshot(state, frame(0, [(0, 1), (1, 0)]), [1, 2.5])
        self.assertEqual(forecast[1][0], (0, 1))
        self.assertEqual(forecast[1][1], (1, 0))
        self.assertEqual(forecast[2.5][1], (1, .5))

    def test_future_pose_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "after its origin"):
            predict_snapshot(snapshot([robot(0, [])]), frame(1, [(0, 0)]), [1])

    def test_nonfinite_rates_are_rejected(self):
        for keyword in ("speed_m_s", "turn_rate_deg_s", "station_dwell_s"):
            for value in (float("nan"), float("inf")):
                with self.subTest(parameter=keyword, value=value):
                    with self.assertRaises(ValueError):
                        predict_snapshot(snapshot([robot(0, [])]), frame(0, [(0, 0)]),
                                         [1], **{keyword: value})


class RecordingEvaluationTests(unittest.TestCase):
    def write_recording(self, root, events):
        run_dir = root / "run-01"
        run_dir.mkdir(exist_ok=True)
        (root / "config.json").write_text(json.dumps({
            "scenario": "scenario.json",
            "parameters": {"ticks_per_second": 1, "velocity": 100},
        }))
        (root / "scenario.json").write_text(json.dumps({
            "name": "test", "conflict_zone": {"rows": [0, 0], "cols": [1, 1]},
        }))
        (run_dir / "simulator.log").write_text(
            "\n".join("[log] " + json.dumps(event) for event in events) + "\n")
        return run_dir

    def test_future_plans_and_future_truth_do_not_change_origin_forecast(self):
        initial = snapshot([robot(0, [action(0, [0, 0], [0, 2])])])
        later = snapshot([robot(0, [])], clock=1, version=2)
        # Deliberately put tick0 after its snapshot in stdout. Clock order,
        # rather than interleaved process line order, determines the pose.
        events = [initial, frame(0, [(0, 0)]), later,
                  frame(1, [(0, 1)]), frame(2, [(0, 2)])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self.write_recording(root, events)
            first_metrics = analyze(run_dir, [1, 2])
            first_forecasts = [json.loads(line) for line in
                               (run_dir / "forecasts.jsonl").read_text().splitlines()]
            altered = deepcopy(events)
            altered[2]["robots"][0]["actions"] = [action(0, [0, 1], [5, 1])]
            altered[3] = frame(1, [(4, 1)])
            altered[4] = frame(2, [(5, 1)])
            self.write_recording(root, altered)
            second_metrics = analyze(run_dir, [1, 2])
            second_forecasts = [json.loads(line) for line in
                                (run_dir / "forecasts.jsonl").read_text().splitlines()]
            self.assertEqual([item for item in first_forecasts if item["origin_clock"] == 0],
                             [item for item in second_forecasts if item["origin_clock"] == 0])
            self.assertNotEqual(first_metrics["metrics"], second_metrics["metrics"])
            self.assertEqual(first_metrics["metrics"][0]["forecast_origins_with_later_plan"], 1)
            self.assertEqual(first_metrics["metrics"][0]["fraction_origins_with_later_plan"], .5)

    def test_hold_baseline_and_occupancy_metrics_on_known_trajectory(self):
        events = [frame(0, [(0, 0)]),
                  snapshot([robot(0, [action(0, [0, 0], [0, 2])])]),
                  frame(1, [(0, 1)]), frame(2, [(0, 2)])]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.write_recording(Path(temporary), events)
            result = analyze(run_dir, [1, 2])
            one = result["metrics"][0]
            self.assertEqual(one["moving_robot_endpoints"], 1)
            self.assertEqual(one["frozen_action"]["endpoint_mean_error_m"], 0)
            self.assertEqual(one["hold_position"]["endpoint_mean_error_m"], 1)
            self.assertEqual(one["frozen_action"]["junction_occupancy_mae_robots"], 0)
            self.assertEqual(one["hold_position"]["junction_occupancy_mae_robots"], 1)

    def test_stable_positions_do_not_create_a_fake_moving_cohort(self):
        events = [snapshot([robot(0, [])]), frame(0, [(0, 0)]), frame(1, [(0, 0)])]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.write_recording(Path(temporary), events)
            result = analyze(run_dir, [1])
            one = result["metrics"][0]
            self.assertEqual(one["moving_robot_endpoints"], 0)
            self.assertEqual(one["frozen_action"]["endpoint_mean_error_m"], 0)
            self.assertIsNone(one["frozen_action"]["moving_endpoint_mean_error_m"])

    def test_no_observed_targets_is_an_error(self):
        events = [snapshot([robot(0, [])]), frame(0, [(0, 0)])]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self.write_recording(Path(temporary), events)
            with self.assertRaisesRegex(ValueError, "No forecast endpoints"):
                analyze(run_dir, [1])


if __name__ == "__main__":
    unittest.main()
