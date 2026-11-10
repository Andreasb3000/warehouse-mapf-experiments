import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.zones.analyze import analyze, in_bounds, payload_events, validate_safety_configuration


class ZoneAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.task = {"id": "t0", "robot_id": 0, "payload_id": "p0", "stages": [
            {"operation": "pickup", "zone_id": "Z00", "location": [10, 12]},
            {"operation": "dropoff", "zone_id": "Z11", "location": [30, 32]},
        ]}

    def test_zone_partition_has_no_boundary_overlap_or_gap(self):
        first = {"rows": [1, 19], "cols": [1, 19]}
        second = {"rows": [20, 39], "cols": [1, 19]}
        for row in (19.49, 19.5, 19.51):
            self.assertEqual(in_bounds(row, 10, first) + in_bounds(row, 10, second), 1)

    def test_acknowledged_service_controls_payload_state(self):
        events = [{"type": "plan_snapshot", "clock": 0},
                  {"type": "goal_reached", "agent": 0, "clock": 100, "row": 10, "col": 12},
                  {"type": "goal_reached", "agent": 0, "clock": 400, "row": 30, "col": 32}]
        states, errors = payload_events(events, [self.task])
        self.assertEqual(errors, [])
        self.assertEqual([s["loaded"] for s in states], [True, False])
        self.assertEqual([s["clock"] for s in states], [100, 400])

    def test_dropoff_before_pickup_and_extra_acknowledgement_rejected(self):
        premature = {"type": "goal_reached", "agent": 0, "clock": 10, "row": 30, "col": 32}
        states, errors = payload_events([premature], [self.task])
        self.assertEqual(states, [])
        self.assertEqual(len(errors), 1)

        pickup = {**premature, "clock": 20, "row": 10, "col": 12}
        dropoff = {**premature, "clock": 30}
        states, errors = payload_events([pickup, dropoff, dropoff], [self.task])
        self.assertEqual(len(states), 2)
        self.assertEqual(len(errors), 1)

    def test_native_configuration_must_match_trigger_and_inner_stop(self):
        scenario = {"safety": {"enabled": True, "robot_trigger": "outer_overlap", "inner_stop_enabled": False},
                    "agents": [{"id": 0}, {"id": 1, "safety": {"inner_stop_enabled": True}}]}
        robots = [{"agent_id": 0, **scenario["safety"]},
                  {"agent_id": 1, **scenario["safety"], "inner_stop_enabled": True}]
        events = [{"type": "safety_config", "robots": robots}]
        self.assertEqual(validate_safety_configuration(events, scenario)[1], [])
        robots[0]["robot_trigger"] = "body_in_outer"
        robots[1]["inner_stop_enabled"] = False
        errors = validate_safety_configuration(events, scenario)[1]
        self.assertEqual(len(errors), 2)
        self.assertIn("robot_trigger", errors[0]); self.assertIn("inner_stop_enabled", errors[1])

    def test_legacy_recording_preserves_inner_stop_enabled_default(self):
        scenario = {"safety": {"enabled": True}, "agents": [{"id": 0}]}
        events = [{"type": "safety_config", "robots": [{"agent_id": 0, "enabled": True}]}]
        self.assertEqual(validate_safety_configuration(events, scenario)[1], [])
        scenario["safety"]["inner_stop_enabled"] = False
        self.assertTrue(validate_safety_configuration(events, scenario)[1])

    def test_measured_metrics_use_run_local_settings_and_keep_unfinished_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / "config.json").write_text("Invalid parent config: run-local inputs must win")
            run = parent / "run-01"; run.mkdir()
            safety = {"enabled": True, "robot_trigger": "outer_overlap", "inner_stop_enabled": False, "speed_scale": .2}
            scenario = {"name": "test", "tasks_file": "tasks.json", "safety": safety,
                        "agents": [{"id": i, "start": [1, 1+2*i]} for i in range(2)],
                        "zones": [{"id": "Z", "kind": "corner", "bounds": {"rows": [0, 10], "cols": [0, 10]},
                                   "conflict_zone": {"rows": [0, 3], "cols": [0, 3]}}]}
            tasks = [{"id": f"t{i}", "payload_id": f"p{i}", "robot_id": i, "source_zone_id": "Z", "destination_zone_id": "Z",
                      "stages": [{"operation": "pickup", "zone_id": "Z", "location": [4, 4+i]},
                                 {"operation": "dropoff", "zone_id": "Z", "location": [5, 4+i]}]} for i in range(2)]
            config = {"scenario": "scenario.json", "parameters": {"seed": 43, "ticks_per_second": 1, "num_agents": 2, "velocity": 100}}
            for name, data in (("config.json", config), ("scenario.json", scenario), ("tasks.json", {"tasks": tasks})):
                (run / name).write_text(json.dumps(data))
            frames = [{"clock": tick, "agents": [{"id": i, "x": -1, "y": -1-2*i, "rz": 0} for i in range(2)]} for tick in range(4)]
            (run / "trajectories.jsonl").write_text("\n".join(json.dumps(frame) for frame in frames))
            events = [{"type": "safety_config", "robots": [{"agent_id": i, **safety} for i in range(2)]},
                      {"type": "goal_reached", "agent": 0, "clock": 1, "row": 4, "col": 4},
                      {"type": "goal_reached", "agent": 0, "clock": 3, "row": 5, "col": 4},
                      {"type": "safety_summary", "clock": 4, "robots": [
                          {"agent_id": i, "max_slow_wheel_command_cm_s": 20, "slowed_move_ticks": 2-i,
                           "stopped_command_ticks": 0, "polygon_only_trigger_ticks": 2-i,
                           "robot_outer_trigger_ticks": 3-i, "obstacle_outer_trigger_ticks": i} for i in range(2)]}]
            (run / "simulator.log").write_text("\n".join(json.dumps(event) for event in events))
            with patch("experiments.zones.analyze.validate_recording", return_value={"errors": [], "warnings": [], "summary": {}}):
                metrics = analyze(run)
            self.assertTrue(metrics["valid"], metrics["errors"])
            self.assertEqual(metrics["seed"], 43)
            self.assertEqual(metrics["deliveries_completed"], 1)
            self.assertEqual(metrics["deliveries_unfinished"], 1)
            self.assertEqual(metrics["mean_completed_delivery_time_s"], 3)
            self.assertIsNone(metrics["all_deliveries_completed_s"])
            self.assertEqual(metrics["polygon_only_trigger_robot_seconds"], 3)
            self.assertEqual(metrics["stationary_active_robot_seconds"], 3)
            self.assertEqual(metrics["peak_stationary_active_robots"], 2)
            self.assertAlmostEqual(metrics["p95_stationary_active_robots"], 1.85)
            # A stationary robot at its service bay is not waiting in a traffic
            # queue, even before its pickup acknowledgement arrives.
            scenario["agents"][1]["start"] = [4, 5]
            (run / "scenario.json").write_text(json.dumps(scenario))
            for frame in frames:
                frame["agents"][1].update(x=-4, y=-5)
            (run / "trajectories.jsonl").write_text("\n".join(json.dumps(frame) for frame in frames))
            with patch("experiments.zones.analyze.validate_recording", return_value={"errors": [], "warnings": [], "summary": {}}):
                service_metrics = analyze(run)
            self.assertTrue(service_metrics["valid"], service_metrics["errors"])
            self.assertEqual(service_metrics["stationary_active_robot_seconds"], 1)


if __name__ == "__main__":
    unittest.main()
