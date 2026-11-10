"""Resume must neither silently reuse changed experiments nor discard old ones."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import shutil

from decision_resume import ARCHIVE, RESUME_INSERTION, validate_resume


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ResumeValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "recording"
        self.run.mkdir()
        self.spec = {"image_id": "sha256:test", "workload_seed": 8001}
        self.policy = {"penalty_weight": 8, "window_weights": [.5, .3, .2]}
        self.original = b"def run_case(spec):\n    if True:\n        return None\n"
        self.guarded = self.original.replace(b"        return None\n", RESUME_INSERTION + b"        return None\n")
        files = {"code/run_decisions.py": self.guarded, "code/decision_policy.py": b"policy",
                 "code/features.py": b"features", "code/local_model.py": b"local",
                 "code/joint.py": b"joint", "results/local/model.joblib": b"weights",
                 str(ARCHIVE / "run_decisions.py"): self.original}
        for name, value in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        for name in ("map.json", "scenario.json", "tasks.json", "config.json"):
            (self.run / name).write_bytes(b"{}")
        self.provenance = {"image_id": self.spec["image_id"], "policy": self.policy,
                           "model_sha256": sha(b"weights"),
                           "host_code_sha256": {name: sha(value) for name, value in files.items()
                                                if name in ("code/run_decisions.py", "code/decision_policy.py", "code/features.py", "code/local_model.py")},
                           "input_sha256": {name: sha(b"{}") for name in ("map.json", "scenario.json", "tasks.json", "config.json")}}
        self.provenance["host_code_sha256"]["code/run_decisions.py"] = sha(self.original)
        (self.run / "run_spec.json").write_text(json.dumps(self.spec))
        (self.run / "provenance.json").write_text(json.dumps(self.provenance))
        (self.root / ARCHIVE / "dependencies.json").write_text(json.dumps({"runner_sha256": sha(self.original),
                                                                         "files_sha256": {"code/joint.py": sha(b"joint")}}))

    def validate(self):
        return validate_resume(self.run, self.spec, self.policy, report_root=self.root)

    def test_existing_recording_accepts_only_the_two_validation_lines(self):
        self.validate()
        self.assertEqual(json.loads((self.run / "provenance.json").read_text()), self.provenance)

    def test_new_guarded_recording_accepts_exact_runner_hash(self):
        self.provenance["host_code_sha256"]["code/run_decisions.py"] = sha(self.guarded)
        (self.run / "provenance.json").write_text(json.dumps(self.provenance))
        self.validate()

    def test_changed_weight_rejected(self):
        self.policy = {**self.policy, "penalty_weight": 9}
        with self.assertRaisesRegex(ValueError, "policy changed"):
            self.validate()

    def test_changed_model_rejected(self):
        (self.root / "results/local/model.joblib").write_bytes(b"new weights")
        with self.assertRaisesRegex(ValueError, "model changed"):
            self.validate()

    def test_changed_features_rejected(self):
        (self.root / "code/features.py").write_bytes(b"new features")
        with self.assertRaisesRegex(ValueError, "Execution dependency changed"):
            self.validate()

    def test_arbitrary_runner_execution_edit_rejected(self):
        (self.root / "code/run_decisions.py").write_bytes(self.guarded.replace(b"return None", b"return 7"))
        with self.assertRaisesRegex(ValueError, "Execution dependency changed"):
            self.validate()

    def test_duplicate_guard_insertion_is_not_a_compatibility_exception(self):
        (self.root / "code/run_decisions.py").write_bytes(self.guarded.replace(RESUME_INSERTION, RESUME_INSERTION * 2))
        with self.assertRaisesRegex(ValueError, "Execution dependency changed"):
            self.validate()

    def test_changed_imported_rollout_rejected(self):
        (self.root / "code/joint.py").write_bytes(b"different dynamics")
        with self.assertRaisesRegex(ValueError, "Frozen imported dependency changed"):
            self.validate()

    def test_modified_recorded_input_rejected(self):
        (self.run / "tasks.json").write_bytes(b'{"different":"tasks"}')
        with self.assertRaisesRegex(ValueError, "Recorded input changed"):
            self.validate()

    def test_missing_provenance_rejected(self):
        (self.run / "provenance.json").unlink()
        with self.assertRaisesRegex(ValueError, "without recorded provenance"):
            self.validate()

    def fresh_provenance(self):
        (self.root / 'code/runtime.py').write_bytes(b'portable runtime')
        self.provenance.update(dependency_scope='fresh-source-v1',
            execution_source_sha256={str(p.relative_to(self.root)): sha(p.read_bytes())
                                     for p in (self.root / 'code').glob('*.py')})
        (self.run / 'provenance.json').write_text(json.dumps(self.provenance))
        shutil.rmtree(self.root / ARCHIVE)

    def test_fresh_export_run_resumes_without_historical_archive(self):
        self.fresh_provenance()
        self.validate()

    def test_fresh_export_rejects_changed_imported_rollout(self):
        self.fresh_provenance()
        (self.root / 'code/joint.py').write_bytes(b'changed dynamics')
        with self.assertRaisesRegex(ValueError, 'Execution dependency changed'):
            self.validate()


if __name__ == "__main__":
    unittest.main()
