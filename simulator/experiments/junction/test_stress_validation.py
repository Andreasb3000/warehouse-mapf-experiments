"""Stress runs may stall, but their recordings must remain complete and valid."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from analyze import analyze


class StressValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name)
        self.config = {"num_agents": 2, "sim_duration": 3, "ticks_per_second": 1}
        self.frames = [
            {"type": "tick", "clock": tick, "agents": [
                {"id": agent, "x": float(agent), "y": 0.0, "z": 0.0,
                 "rx": 0.0, "ry": 0.0, "rz": 0.0}
                for agent in range(2)
            ]}
            for tick in range(3)
        ]
        self.stats = {
            "success": True, "congested": True,
            "total_finished_tasks": 0, "total_finished_backup_tasks": 0,
            "throughput": 0.0, "planner_invoke_ticks": [0], "n_planner_invokes": 1,
            "stats_per_tick": {
                "n_unfinished_nodes": [2, 2, 2],
                "min_unfinished_nodes": [1, 1, 1],
                "portion_idle_robots": [1.0, 1.0, 1.0],
            },
        }

    def analyze_recording(self, *, stress=False):
        (self.run_dir / "stats.json").write_text(json.dumps(self.stats))
        (self.run_dir / "simulator.log").write_text(
            "\n".join(json.dumps(frame) for frame in self.frames)
            + "\nSimulation count: 3\nServer closed successfully.\n"
        )
        return analyze(self.run_dir, self.config, stress=stress)

    def test_complete_stationary_congested_run_is_valid_stress_recording(self):
        result = self.analyze_recording(stress=True)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["validation_mode"], "stress_recording")
        self.assertEqual(result["summary"]["moving_agents"], 0)
        self.assertEqual(result["summary"]["total_finished_tasks"], 0)
        self.assertTrue(result["summary"]["congested"])

    def test_same_recording_still_fails_default_baseline_requirements(self):
        result = self.analyze_recording()
        self.assertFalse(result["valid"])
        errors = "\n".join(result["errors"])
        self.assertIn("never moved", errors)
        self.assertIn("congestion", errors)
        self.assertIn("No completed tasks", errors)

    def test_stress_rejects_missing_frame(self):
        del self.frames[1]
        result = self.analyze_recording(stress=True)
        self.assertFalse(result["valid"])
        self.assertTrue(any("consecutive trajectory clocks" in error
                            for error in result["errors"]))

    def test_stress_rejects_duplicate_unknown_and_noninteger_robot_ids(self):
        for invalid_id in (0, 99, True):
            with self.subTest(invalid_id=invalid_id):
                self.frames[1]["agents"][1]["id"] = invalid_id
                result = self.analyze_recording(stress=True)
                self.assertFalse(result["valid"])
                self.assertTrue(any("Invalid agent identities" in error
                                    for error in result["errors"]))

    def test_stress_accepts_consistently_short_adg_trace_with_coverage_warning(self):
        for key in self.stats["stats_per_tick"]:
            self.stats["stats_per_tick"][key].pop(0)
        result = self.analyze_recording(stress=True)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["summary"]["frames"], 3)
        self.assertEqual(result["summary"]["adg_stat_samples"],
                         {key: 2 for key in self.stats["stats_per_tick"]})
        self.assertTrue(result["warnings"], "Incomplete ADG coverage must be visible.")

    def test_stress_rejects_mismatched_adg_trace_lengths(self):
        self.stats["stats_per_tick"]["n_unfinished_nodes"].pop()
        result = self.analyze_recording(stress=True)
        self.assertFalse(result["valid"])
        self.assertEqual(result["summary"]["adg_stat_samples"], {
            "n_unfinished_nodes": 2,
            "min_unfinished_nodes": 3,
            "portion_idle_robots": 3,
        })

    def test_baseline_still_requires_adg_trace_for_every_tick(self):
        self.stats.update(congested=False, total_finished_tasks=1, throughput=1 / 3)
        for agent in self.frames[-1]["agents"]:
            agent["x"] += 1
        for key in self.stats["stats_per_tick"]:
            self.stats["stats_per_tick"][key].pop()
        result = self.analyze_recording()
        self.assertFalse(result["valid"])
        self.assertTrue(all("stats_per_tick" in error for error in result["errors"]),
                        result["errors"])

    def test_stress_rejects_empty_or_overlong_adg_trace(self):
        for length in (0, 4):
            with self.subTest(samples=length):
                for key in self.stats["stats_per_tick"]:
                    self.stats["stats_per_tick"][key] = [1] * length
                result = self.analyze_recording(stress=True)
                self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
