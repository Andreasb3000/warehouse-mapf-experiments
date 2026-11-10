"""Reject cached native runs when their execution inputs have changed.

The sole legacy-runner compatibility exception is a byte-exact insertion of the
two validation lines below. Removing them must reproduce the recorded runner
hash, so this exception cannot conceal edits to forecast or execution logic.
"""
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("archive/2026-10-31-routing-execution")
RESUME_INSERTION = (b"        from decision_resume import validate_resume\n"
                    b"        validate_resume(directory, spec, POLICY)\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_resume(directory, spec, policy, *, report_root=REPORT):
    """Validate existing artifacts; never rewrite their historical provenance."""
    directory, report_root = Path(directory), Path(report_root)
    provenance_path = directory / "provenance.json"
    if not provenance_path.is_file():
        raise ValueError(f"Cannot resume without recorded provenance: {directory}")
    provenance = json.loads(provenance_path.read_text())
    if json.loads((directory / "run_spec.json").read_text()) != spec:
        raise ValueError(f"Run specification changed; do not reuse {directory}")
    if provenance.get("image_id") != spec["image_id"]:
        raise ValueError(f"Native image changed; do not reuse {directory}")
    if provenance.get("policy") != policy:
        raise ValueError(f"Routing policy changed; do not reuse {directory}")
    if (provenance.get("model_sha256") is not None and
            provenance["model_sha256"] != digest(report_root / "results/local/model.joblib")):
        raise ValueError(f"Predictor model changed; do not reuse {directory}")

    if provenance.get('dependency_scope') == 'fresh-source-v1':
        dependencies = provenance.get('execution_source_sha256', {})
        required = {'code/run_decisions.py', 'code/decision_policy.py', 'code/features.py',
                    'code/local_model.py', 'code/runtime.py'}
        if not required.issubset(dependencies):
            raise ValueError(f'Incomplete recorded execution dependencies: {directory}')
        for relative, expected in dependencies.items():
            if digest(report_root / relative) != expected:
                raise ValueError(f'Execution dependency changed ({relative}); do not reuse {directory}')
        inputs = provenance.get('input_sha256', {})
        if not {'map.json', 'scenario.json', 'tasks.json', 'config.json'}.issubset(inputs):
            raise ValueError(f'Incomplete recorded input hashes: {directory}')
        for relative, expected in inputs.items():
            if digest(directory / relative) != expected:
                raise ValueError(f'Recorded input changed ({relative}); do not reuse {directory}')
        return

    archived = report_root / ARCHIVE
    frozen = json.loads((archived / "dependencies.json").read_text())
    if digest(archived / "run_decisions.py") != frozen["runner_sha256"]:
        raise ValueError("Archived execution runner does not match its frozen fingerprint")
    dependencies = provenance.get("host_code_sha256", {})
    required = {"code/run_decisions.py", "code/decision_policy.py", "code/features.py", "code/local_model.py"}
    if not required.issubset(dependencies):
        raise ValueError(f"Incomplete recorded execution dependencies: {directory}")
    for relative, expected in dependencies.items():
        path = report_root / relative
        if digest(path) == expected:
            continue
        if relative == "code/run_decisions.py" and expected == frozen["runner_sha256"]:
            source = path.read_bytes()
            if source.count(RESUME_INSERTION) == 1:
                original = source.replace(RESUME_INSERTION, b"", 1)
                if hashlib.sha256(original).hexdigest() == expected:
                    continue
        raise ValueError(f"Execution dependency changed ({relative}); do not reuse {directory}")
    for relative, expected in frozen["files_sha256"].items():
        if digest(report_root / relative) != expected:
            raise ValueError(f"Frozen imported dependency changed ({relative}); do not reuse {directory}")
    inputs = provenance.get("input_sha256", {})
    if not {"map.json", "scenario.json", "tasks.json", "config.json"}.issubset(inputs):
        raise ValueError(f"Incomplete recorded input hashes: {directory}")
    for relative, expected in inputs.items():
        if digest(directory / relative) != expected:
            raise ValueError(f"Recorded input changed ({relative}); do not reuse {directory}")
