"""Paired reports must reject confounded experiments and preserve censoring."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.zones.comparison import summarize


class SafetyComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        for variant in ("body_in_outer", "outer_overlap"):
            run = self.output / variant / "run-01"; run.mkdir(parents=True)
            scenario = {"name": "test", "map_file": "map.json", "tasks_file": "tasks.json",
                        "safety": {"enabled": True, "inner_stop_enabled": False, "robot_trigger": variant},
                        "agents": [{"id": 0, "start": [0, 0]}]}
            metrics = {"valid": True, "robot_trigger": variant, "inner_stop_enabled": False, "safety_enabled": True,
                       "seed": 42, "deliveries_completed": 1, "deliveries_unfinished": 0,
                       "all_deliveries_completed_s": 10 if variant == "body_in_outer" else 12,
                       "mean_completed_delivery_time_s": 10 if variant == "body_in_outer" else 12}
            for name, value in (("config.json", {"scenario": "scenario.json", "parameters": {"seed": 42}}),
                                ("scenario.json", scenario), ("map.json", {"layout": ["."]}),
                                ("tasks.json", {"tasks": [{"id": "t0"}]}), ("zone_metrics.json", metrics)):
                (run / name).write_text(json.dumps(value))
            (run / "replay.html").write_text("<html>measured replay</html>")

    def change(self, variant, file, update):
        path = self.output / variant / "run-01" / file
        value = json.loads(path.read_text()); update(value); path.write_text(json.dumps(value))

    def test_paired_differences_and_offline_links(self):
        report = summarize(self.output)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["pairs"][0]["outer_minus_body"]["all_deliveries_completed_s"], 2)
        self.assertEqual(report["paired_difference_summary"]["all_deliveries_completed_s"]["pairs_with_both_values"], 1)
        page = (self.output / "index.html").read_text()
        self.assertIn('src="body_in_outer/run-01/replay.html"', page)
        self.assertIn("finite", page.lower())
        self.assertIn("not the proposed 16-junction", page)
        self.assertIn("spillback", page)
        self.assertNotIn('src="http', page)

    def test_changed_task_assignment_invalidates_pair(self):
        self.change("outer_overlap", "tasks.json", lambda value: value["tasks"].append({"id": "t1"}))
        report = summarize(self.output)
        self.assertFalse(report["valid"])
        self.assertTrue(any("inputs differ" in error for error in report["errors"]))

    def test_seed_and_actual_native_mode_mismatches_are_rejected(self):
        self.change("outer_overlap", "config.json", lambda value: value["parameters"].update(seed=43))
        self.change("outer_overlap", "zone_metrics.json", lambda value: value.update(robot_trigger="body_in_outer"))
        report = summarize(self.output)
        self.assertFalse(report["valid"])
        self.assertTrue(any("same explicit seed" in error for error in report["errors"]))
        self.assertTrue(any("recorded trigger" in error for error in report["errors"]))

    def test_unfinished_tasks_do_not_get_fictitious_completion_times(self):
        self.change("outer_overlap", "zone_metrics.json", lambda value: value.update(
            deliveries_completed=0, deliveries_unfinished=1, all_deliveries_completed_s=None,
            mean_completed_delivery_time_s=None))
        report = summarize(self.output)
        self.assertTrue(report["valid"], report["errors"])
        self.assertIsNone(report["pairs"][0]["outer_minus_body"]["all_deliveries_completed_s"])
        self.assertEqual(report["pairs"][0]["outer_minus_body"]["deliveries_unfinished"], 1)
        self.assertEqual(report["paired_difference_summary"]["all_deliveries_completed_s"]["pairs_with_both_values"], 0)


if __name__ == "__main__":
    unittest.main()
