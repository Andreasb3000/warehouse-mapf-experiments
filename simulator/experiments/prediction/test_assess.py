import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import hashlib
import tempfile

from experiments.prediction.assess import assess_documents, recording_identity, render


PROFILE = json.loads(Path(__file__).with_name("readiness_profile.json").read_text())
PROFILE["bootstrap_samples"] = 100


def scores(tp=100, fp=0, fn=0, tn=100):
    return {"multi_robot_queue": dict(tp=tp, fp=fp, fn=fn, tn=tn),
            "spillback_warning": dict(tp=0, fp=0, fn=0, tn=200)}


def recording(seed, **overrides):
    result = {"scenario": "test_map", "trigger": "body_in_outer", "seed": seed,
              "run_directory": f"/test/seed-{seed}", "forecast_origins": 100,
              "runtime_s": {"p95": .1}, "parameters": {"dt_s": .1},
              "metrics": {"10.0": {"joint": scores(), "plan_only": scores(70, 30, 30, 70),
                                    "hold": scores(50, 50, 50, 50)}}}
    result.update(overrides)
    return result


def report(runs, profile=None):
    with patch("experiments.prediction.assess.recording_identity", return_value=("verified-test-condition", None)):
        return assess_documents([{"runs": runs, "source_sha256": {"model.py": "test-source"}}], profile or PROFILE)


class AssessTest(unittest.TestCase):
    def test_perfect_supported_queue_screen_and_absent_spillback(self):
        result = report([recording(s) for s in range(5)])
        c = result["conditions"][0]
        self.assertEqual(c["status"], "PASS")
        q = c["events"]["multi_robot_queue"]
        self.assertEqual(q["methods"]["joint"]["tn"], 500)
        self.assertEqual(q["seed_bootstrap_95_intervals"]["recall"], [1, 1])
        self.assertEqual(c["events"]["spillback_warning"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["research_readiness"], "NOT_ESTABLISHED")

    def test_one_seed_many_windows_is_insufficient(self):
        run = recording(0, forecast_origins=1000000)
        result = report([run])["conditions"][0]
        self.assertEqual(result["status"], "INSUFFICIENT_EVIDENCE")
        self.assertNotIn("seed_bootstrap_95_intervals", result["events"]["multi_robot_queue"])

    def test_repeated_seed_is_one_cluster(self):
        runs = [recording(1, run_directory=f"/test/repeat-{i}") for i in range(5)]
        self.assertEqual(report(runs)["conditions"][0]["distinct_seeds"], 1)

    def test_no_events_never_pass(self):
        runs = [recording(s) for s in range(5)]
        for r in runs:
            for m in r["metrics"]["10.0"].values():
                m["multi_robot_queue"] = dict(tp=0, fp=0, fn=0, tn=200)
        event = report(runs)["conditions"][0]["events"]["multi_robot_queue"]
        self.assertEqual(event["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(event["methods"]["joint"]["recall"])

    def test_supported_bad_recall_fails(self):
        runs = [recording(s) for s in range(5)]
        for r in runs:
            r["metrics"]["10.0"]["joint"] = scores(50, 0, 50, 100)
        self.assertEqual(report(runs)["conditions"][0]["status"], "FAIL")

    def test_matching_baseline_is_not_improvement(self):
        runs = [recording(s) for s in range(5)]
        for r in runs:
            r["metrics"]["10.0"]["hold"] = scores()
        event = report(runs)["conditions"][0]["events"]["multi_robot_queue"]
        self.assertEqual(event["status"], "FAIL")
        self.assertEqual(event["paired_f1_gain_95_intervals"]["hold"], [0, 0])

    def test_truth_mismatch_rejected(self):
        run = recording(0)
        run["metrics"]["10.0"]["hold"] = scores(0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "different truth cohorts"):
            report([run])

    def test_opposite_per_run_truth_mismatches_cannot_cancel(self):
        a, b = recording(0), recording(0, run_directory="/test/repeat")
        a["metrics"]["10.0"]["hold"] = scores(100, 0, 100, 0)
        b["metrics"]["10.0"]["hold"] = scores(0, 100, 0, 100)
        with self.assertRaisesRegex(ValueError, "different truth cohorts"):
            report([a, b])

    def test_unverifiable_inputs_cannot_pool_or_pass(self):
        result = assess_documents([{"runs": [recording(s) for s in range(5)]}], PROFILE)
        self.assertEqual(len(result["conditions"]), 5)
        for c in result["conditions"]:
            self.assertEqual(c["status"], "INSUFFICIENT_EVIDENCE")
            self.assertTrue(c["provenance_issues"])

    def test_fingerprint_verifies_inputs_and_excludes_only_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contents = {"config.json": {"parameters": {"seed": 1, "velocity": 100}},
                        "scenario.json": {"name": "test_map"}, "map.json": {"layout": ["..."]}}
            def save():
                hashes = {}
                for name, obj in contents.items():
                    data = json.dumps(obj).encode()
                    (root/name).write_bytes(data)
                    hashes[name] = hashlib.sha256(data).hexdigest()
                (root/"provenance.json").write_text(json.dumps({"image_id": "native-version",
                    "compiled_native_source_sha256": {"c.cpp": "source"}, "input_sha256": hashes}))
            save()
            first, issue = recording_identity(recording(1, run_directory=temp))
            self.assertIsNone(issue)
            contents["config.json"]["parameters"]["seed"] = 2
            save()
            second, issue = recording_identity(recording(2, run_directory=temp))
            self.assertEqual(first, second)
            contents["config.json"]["parameters"]["velocity"] = 50
            save()
            third, _ = recording_identity(recording(2, run_directory=temp))
            self.assertNotEqual(first, third)
            (root/"map.json").write_text("tampered")
            self.assertIsNone(recording_identity(recording(2, run_directory=temp))[0])

    def test_duplicate_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate recording"):
            report([recording(0), recording(0)])

    def test_conditions_not_pooled_and_source_versions_separated(self):
        a, b = recording(0), recording(1, trigger="outer_overlap")
        self.assertEqual(len(report([a, b])["conditions"]), 2)
        docs = [{"runs": [recording(s)], "source_sha256": {"model.py": f"hash{s}"}} for s in range(2)]
        self.assertEqual(len(assess_documents(docs, PROFILE)["conditions"]), 2)

    def test_missing_horizon_or_method_stays_insufficient(self):
        run = recording(0, metrics={"5.0": {}})
        self.assertEqual(report([run])["conditions"][0]["status"], "INSUFFICIENT_EVIDENCE")
        run = recording(0)
        del run["metrics"]["10.0"]["joint"]
        self.assertEqual(report([run])["conditions"][0]["status"], "INSUFFICIENT_EVIDENCE")

    def test_bad_runtime_and_invalid_profile(self):
        runs = [recording(s, runtime_s={"p95": 1}) for s in range(5)]
        self.assertEqual(report(runs)["conditions"][0]["status"], "FAIL")
        runs[0]["runtime_s"] = {"p95": float("nan")}
        self.assertEqual(report(runs)["conditions"][0]["core_runtime"]["status"], "INSUFFICIENT_EVIDENCE")
        for key, bad in [("min_precision", 1.1), ("min_seeds", 1), ("horizon_s", float("nan"))]:
            p = {**PROFILE, key: bad}
            with self.assertRaises(ValueError): report([recording(0)], p)

    def test_runtime_must_measure_the_selected_predictor(self):
        runs = [recording(s, runtime_method="different_model") for s in range(5)]
        self.assertEqual(report(runs)["conditions"][0]["core_runtime"]["status"], "INSUFFICIENT_EVIDENCE")

    def test_determinism_input_immutability_and_html_escaping(self):
        runs = [recording(s, scenario="<script>unsafe</script>") for s in range(5)]
        saved = copy.deepcopy(runs)
        a, b = report(runs), report(runs)
        self.assertEqual(a, b)
        self.assertEqual(runs, saved)
        self.assertNotIn("<script>unsafe", render(a))
        self.assertIn("&lt;script&gt;unsafe", render(a))


if __name__ == "__main__":
    unittest.main()
