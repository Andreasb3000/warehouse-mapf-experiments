"""Guard matched safety experiments and the saved inputs used for replay."""

from copy import deepcopy
import json
import hashlib
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.zones.generate import build_case, write_case
from experiments.zones.run import (
    check_pbs_sources, condition_metadata, config_for_run, copy_inputs, execute,
    experiment_cases, scenario_with_policy,
)


class MatchedRunTests(unittest.TestCase):
    def setUp(self):
        _, self.scenario, self.config, _ = build_case()

    def test_pbs_source_check_rejects_stale_or_missing_planner_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "planner/PBS/src/solver.cpp"
            header = root / "planner/PBS/inc/solver.h"
            source.parent.mkdir(parents=True)
            header.parent.mkdir(parents=True)
            source.write_text("original planner")
            header.write_text("planner interface")
            manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (source, header)}
            with patch("experiments.zones.run.ROOT", root), \
                 patch("experiments.zones.run.capture", return_value=json.dumps(manifest)) as capture:
                self.assertEqual(check_pbs_sources(["docker"], "image"), manifest)
                self.assertEqual(capture.call_args.args[0][-1], "/usr/project/pbs-source-sha256.json")
                source.write_text("parking-corrected planner")
                with self.assertRaisesRegex(SystemExit, "PBS sources differ"):
                    check_pbs_sources(["docker"], "image")
                capture.side_effect = subprocess.CalledProcessError(1, ["cat"])
                with self.assertRaisesRegex(SystemExit, "metadata is missing"):
                    check_pbs_sources(["docker"], "image")

    def test_trigger_comparison_preserves_geometry_and_enforces_all_robot_policies(self):
        self.scenario["safety"]["enabled"] = False
        self.scenario["safety"]["inner_stop_enabled"] = True
        self.scenario["agents"][0]["safety"] = {
            "enabled": False, "robot_trigger": "outer_overlap",
            "inner_stop_enabled": True, "slow_front_m": 1.8, "speed_scale": .4,
        }
        original = deepcopy(self.scenario)
        cases = experiment_cases(self.scenario, compare_triggers=True)
        self.assertEqual([name for name, _ in cases], ["body_in_outer", "outer_overlap"])
        for name, scenario in cases:
            for agent in scenario["agents"]:
                policy = {**scenario["safety"], **agent.get("safety", {})}
                self.assertTrue(policy["enabled"])
                self.assertFalse(policy["inner_stop_enabled"])
                self.assertEqual(policy["robot_trigger"], name)
            self.assertEqual(scenario["agents"][0]["safety"]["slow_front_m"], 1.8)
            self.assertEqual(scenario["agents"][0]["safety"]["speed_scale"], .4)
        body, overlap = [deepcopy(scenario) for _, scenario in cases]
        for scenario in (body, overlap):
            scenario["safety"].pop("robot_trigger")
            for agent in scenario["agents"]:
                agent.get("safety", {}).pop("robot_trigger", None)
        self.assertEqual(body, overlap)
        self.assertEqual(self.scenario, original)

    def test_legacy_defaults_and_explicit_overrides(self):
        self.scenario["safety"].pop("robot_trigger")
        self.scenario["safety"].pop("inner_stop_enabled")
        legacy = scenario_with_policy(self.scenario)
        self.assertEqual(legacy["safety"]["robot_trigger"], "body_in_outer")
        self.assertTrue(legacy["safety"]["inner_stop_enabled"])
        self.scenario["agents"][0]["safety"] = {"inner_stop_enabled": True, "enabled": True}
        override = scenario_with_policy(self.scenario, safety=False,
                                        robot_trigger="outer_overlap", inner_stop=False)
        self.assertFalse(override["agents"][0]["safety"]["enabled"])
        self.assertFalse(override["agents"][0]["safety"]["inner_stop_enabled"])
        self.assertEqual(override["agents"][0]["safety"]["robot_trigger"], "outer_overlap")

    def test_safety_on_off_comparison_overrides_agent_enabled_flags_only(self):
        self.scenario["agents"][0]["safety"] = {"enabled": True, "inner_stop_enabled": True}
        cases = experiment_cases(self.scenario, compare_safety=True)
        self.assertEqual([name for name, _ in cases], ["safety_on", "safety_off"])
        for name, scenario in cases:
            enabled = name == "safety_on"
            self.assertEqual(scenario["safety"]["enabled"], enabled)
            self.assertEqual(scenario["agents"][0]["safety"]["enabled"], enabled)
            self.assertTrue(scenario["agents"][0]["safety"]["inner_stop_enabled"])

    def test_repetitions_use_distinct_but_paired_seeds_without_mutating_base(self):
        seeds = [[config_for_run(self.config, index)["parameters"]["seed"]
                  for index in (1, 2, 3)] for _ in range(2)]
        self.assertEqual(seeds, [[42, 43, 44], [42, 43, 44]])
        self.assertEqual(self.config["parameters"]["seed"], 42)
        for index in (0, -1, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                config_for_run(self.config, index)

    def test_execute_mounts_and_saves_exact_run_inputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = write_case(directory / "fixture")
            case = directory / "outer_overlap"
            scenario = experiment_cases(self.scenario, compare_triggers=True)[1][1]
            copy_inputs(source, case, self.config, scenario)
            (case / "provenance.json").write_text(json.dumps({"image_id": "test-image"}))
            config = deepcopy(self.config)
            config["record_plans"] = False
            command = []

            def simulate(args, **kwargs):
                command.extend(args)
                return type("Result", (), {"returncode": 0})()

            def analyze_saved(run_dir):
                saved = json.loads((run_dir / "config.json").read_text())
                self.assertEqual(saved["parameters"]["seed"], 44)
                self.assertEqual(json.loads((run_dir / "scenario.json").read_text()), scenario)
                for name in ("map.json", "tasks.json", "condition.json", "provenance.json"):
                    self.assertTrue((run_dir / name).is_file())
                return {"valid": True, "errors": []}

            with patch("experiments.zones.run.subprocess.run", side_effect=simulate), \
                    patch("experiments.zones.run.analyze", side_effect=analyze_saved), \
                    patch("experiments.zones.viewer.write_viewer"):
                result = execute(["docker"], "test-image", case, config, scenario, 3)
            run_dir = case / "run-03"
            self.assertIn(f"{run_dir}:/inputs:ro", command)
            self.assertIn("--seed=44", command)
            self.assertEqual(result["condition"], "outer_overlap")
            self.assertEqual(result["robot_trigger"], "outer_overlap")
            self.assertFalse(result["inner_stop_enabled"])
            self.assertEqual(result["seed"], 44)
            self.assertEqual(result["run_index"], 3)
            recorded = json.loads((run_dir / "condition.json").read_text())
            self.assertEqual(recorded["input_sha256"], result["input_sha256"])
            self.assertEqual(json.loads((case / "config.json").read_text())["parameters"]["seed"], 42)

    def test_metadata_does_not_mislabel_heterogeneous_agent_policies(self):
        self.scenario["agents"][0]["safety"] = {"inner_stop_enabled": True}
        with tempfile.TemporaryDirectory() as temporary:
            source = write_case(Path(temporary) / "fixture")
            run_dir = Path(temporary) / "run"
            copy_inputs(source, run_dir, self.config, self.scenario)
            metadata = condition_metadata("custom", self.config, self.scenario, 1, run_dir)
            self.assertEqual(metadata["inner_stop_enabled"], "mixed")
            self.assertEqual(metadata["robot_trigger"], "body_in_outer")


if __name__ == "__main__":
    unittest.main()
