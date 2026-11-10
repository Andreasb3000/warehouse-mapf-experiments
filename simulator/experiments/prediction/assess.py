#!/usr/bin/env python3
"""Screen saved forecasts by condition; this does not certify routing usefulness.

Confidence intervals resample whole seed clusters, never robot/time samples.
Thresholds in readiness_profile.json are provisional research choices.
"""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import random

COUNTS = ("tp", "fp", "fn", "tn")
EVENTS = ("multi_robot_queue", "spillback_warning")
UNMEASURED = [
    "Matched congestion episodes and warning before the last viable rerouting decision",
    "Independent native free-flow traversal times for excess delay",
    "End-to-end forecast latency, including observation age and communication",
    "Paired closed-loop routing benefit and congestion created elsewhere",
    "An untouched map/workload test set and declared scope of generalization",
]


def validate_profile(p):
    for key in ("horizon_s", "max_core_p95_s"):
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("min_precision", "min_recall"):
        if not isinstance(p[key], (int, float)) or isinstance(p[key], bool) or not 0 < p[key] <= 1:
            raise ValueError(f"{key} must lie in (0, 1]")
    for key, minimum in (("min_seeds", 2), ("min_positive_seeds", 1), ("bootstrap_samples", 100)):
        if type(p[key]) is not int or p[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if p["min_positive_seeds"] > p["min_seeds"]:
        raise ValueError("min_positive_seeds cannot exceed min_seeds")
    if not isinstance(p["method"], str) or not p["method"]:
        raise ValueError("method must be a nonempty string")
    if (not isinstance(p["baselines"], list) or not p["baselines"]
            or any(not isinstance(b, str) or not b for b in p["baselines"])
            or len(set(p["baselines"])) != len(p["baselines"])
            or p["method"] in p["baselines"]):
        raise ValueError("baselines must be distinct names excluding method")
    if type(p["bootstrap_seed"]) is not int:
        raise ValueError("bootstrap_seed must be an integer")


def add_counts(counts):
    counts = list(counts)
    return {key: sum(c[key] for c in counts) for key in COUNTS}


def event_counts(method, event):
    c = method[event]
    if any(type(c[k]) is not int or c[k] < 0 for k in COUNTS):
        raise ValueError("Confusion counts must be nonnegative integers")
    return {k: c[k] for k in COUNTS}


def rates(c):
    def ratio(n, d):
        return n/d if d else None
    return {"precision": ratio(c["tp"], c["tp"]+c["fp"]),
            "recall": ratio(c["tp"], c["tp"]+c["fn"]),
            "f1": ratio(2*c["tp"], 2*c["tp"]+c["fp"]+c["fn"])}


def quantile(values, q):
    values = sorted(values)
    at = (len(values)-1)*q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def interval(values):
    return [quantile(values, .025), quantile(values, .975)]


def recording_identity(run):
    """Verify archived inputs, grouping only identical conditions except seed.

    Conservatively keep different realized task streams separate. A future
    workload-generator adapter may explicitly identify their common distribution.
    """
    directory = Path(run["run_directory"])
    try:
        provenance = json.loads((directory/"provenance.json").read_text())
        hashes = provenance["input_sha256"]
        if not {"config.json", "scenario.json", "map.json"}.issubset(hashes):
            raise ValueError("Missing required archived input hashes")
        for name, expected in hashes.items():
            path = directory/name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Archived input must be inside its run directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Archived input hash differs: {name}")
        config = json.loads((directory/"config.json").read_text())
        scenario = json.loads((directory/"scenario.json").read_text())
        if config["parameters"].pop("seed") != run["seed"] or scenario["name"] != run["scenario"]:
            raise ValueError("Metrics and archived seed/scenario differ")
        if scenario.get("safety", {}).get("robot_trigger", "body_in_outer") != run["trigger"]:
            raise ValueError("Metrics and archived safety trigger differ")
        if not provenance["image_id"] or not provenance["compiled_native_source_sha256"]:
            raise ValueError("Missing native image or compiled-source identity")
        condition = {"config_without_seed": config,
                     "input_sha256": {k: v for k, v in hashes.items() if k != "config.json"},
                     "image_id": provenance["image_id"],
                     "compiled_native_source_sha256": provenance["compiled_native_source_sha256"]}
        return hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest(), None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"Cannot verify recording condition: {exc}"


def assess_event(runs, horizon_key, event, profile):
    methods = [profile["method"], *profile["baselines"]]
    clusters = defaultdict(lambda: defaultdict(list))
    for run in runs:
        scores = run["metrics"][horizon_key]
        per_run = {}
        for method in methods:
            if method not in scores or event not in scores[method]:
                return {"status": "INSUFFICIENT_EVIDENCE", "reason": f"Missing {method}/{event} scores"}
            per_run[method] = event_counts(scores[method], event)
        truth = {(c["tp"]+c["fn"], c["fp"]+c["tn"]) for c in per_run.values()}
        if len(truth) != 1:
            raise ValueError("Methods have different truth cohorts in a recording; paired comparison is invalid")
        for method, counts in per_run.items():
            clusters[run["seed"]][method].append(counts)
    clusters = [{m: add_counts(c[m]) for m in methods} for c in clusters.values()]
    for cluster in clusters:
        truth = {(c["tp"]+c["fn"], c["fp"]+c["tn"]) for c in cluster.values()}
        if len(truth) != 1:
            raise ValueError("Methods have different truth cohorts; paired comparison is invalid")
    pooled = {m: add_counts(c[m] for c in clusters) for m in methods}
    scores = {m: {**pooled[m], **rates(pooled[m])} for m in methods}
    positive_seeds = sum(c[profile["method"]]["tp"]+c[profile["method"]]["fn"] > 0 for c in clusters)
    result = {"status": "INSUFFICIENT_EVIDENCE", "seed_clusters": len(clusters),
              "positive_seed_clusters": positive_seeds, "methods": scores}
    point = scores[profile["method"]]
    result["point_targets_met"] = all(point[k] is not None and point[k] >= profile[f"min_{k}"]
                                       for k in ("precision", "recall"))
    if len(clusters) < profile["min_seeds"] or positive_seeds < profile["min_positive_seeds"]:
        result["reason"] = "Too few distinct seeds or seeds containing positive cases; windows are not independent trials"
        return result
    rng = random.Random(profile["bootstrap_seed"])
    sampled_rates = {k: [] for k in ("precision", "recall")}
    differences = {m: [] for m in profile["baselines"]}
    for _ in range(profile["bootstrap_samples"]):
        sample = [rng.choice(clusters) for _ in clusters]
        r = {m: rates(add_counts(c[m] for c in sample)) for m in methods}
        for k in sampled_rates:
            # An event-free or alarm-free resample cannot establish skill.
            sampled_rates[k].append(r[profile["method"]][k] or 0.0)
        for m in differences:
            differences[m].append((r[profile["method"]]["f1"] or 0.0)-(r[m]["f1"] or 0.0))
    result["seed_bootstrap_95_intervals"] = {k: interval(v) for k, v in sampled_rates.items()}
    result["paired_f1_gain_95_intervals"] = {m: interval(v) for m, v in differences.items()}
    passed = all(result["seed_bootstrap_95_intervals"][k][0] >= profile[f"min_{k}"] for k in sampled_rates)
    passed &= all(bounds[0] > 0 for bounds in result["paired_f1_gain_95_intervals"].values())
    result["status"] = "PASS" if passed else "FAIL"
    result["reason"] = "Requires lower confidence bounds above targets and positive paired F1 gain over every baseline"
    return result


def assess_documents(documents, profile):
    validate_profile(profile)
    groups, seen, provenance_issues = defaultdict(list), set(), {}
    for document in documents:
        source = document.get("source_sha256", {})
        for run in document.get("runs", [document]):
            identity = str(Path(run["run_directory"]).resolve())
            if identity in seen:
                raise ValueError(f"Duplicate recording: {identity}")
            seen.add(identity)
            if type(run["seed"]) is not int:
                raise ValueError("Each recording must identify an integer random seed")
            fingerprint, issue = recording_identity(run)
            if not source:
                issue = "Missing evaluator/predictor source hashes; use combined evaluation metrics"
            if issue:
                provenance_issues[identity] = issue
            # Do not pool different named workloads, safety rules, settings or source versions.
            key = json.dumps([run["scenario"], run["trigger"], run.get("parameters", {}), source,
                              fingerprint if not issue else identity], sort_keys=True)
            groups[key].append(run)
    if not groups:
        raise ValueError("No evaluated runs supplied")
    conditions = []
    for key, runs in sorted(groups.items()):
        scenario, trigger, parameters, source, fingerprint = json.loads(key)
        issues = [provenance_issues[str(Path(r["run_directory"]).resolve())]
                  for r in runs if str(Path(r["run_directory"]).resolve()) in provenance_issues]
        item = {"scenario": scenario, "trigger": trigger, "parameters": parameters,
                "source_sha256": source, "run_directories": [r["run_directory"] for r in runs],
                "condition_sha256": None if issues else fingerprint, "provenance_issues": issues,
                "runs": len(runs), "distinct_seeds": len({r["seed"] for r in runs}),
                "forecast_origins": sum(r.get("forecast_origins", 0) for r in runs)}
        horizons = [{float(h): h for h in r["metrics"]} for r in runs]
        if any(profile["horizon_s"] not in h for h in horizons):
            item.update(status="INSUFFICIENT_EVIDENCE", reason="Requested horizon absent")
            conditions.append(item)
            continue
        # Normalize numeric JSON horizon spellings (10 and 10.0).
        horizon = str(float(profile["horizon_s"]))
        normalized = [{**r, "metrics": {horizon: r["metrics"][h[profile["horizon_s"]]]}} for r, h in zip(runs, horizons)]
        item["events"] = {e: assess_event(normalized, horizon, e, profile) for e in EVENTS}
        runtimes = [r.get("runtime_s", {}).get("p95")
                    if r.get("runtime_method", "joint") == profile["method"] else None for r in runs]
        valid_runtime = all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in runtimes)
        value = max(runtimes) if valid_runtime else None
        item["core_runtime"] = {"worst_run_p95_s": value,
            "status": "INSUFFICIENT_EVIDENCE" if value is None else "PASS" if value <= profile["max_core_p95_s"] else "FAIL",
            "note": "Worst run p95, not pooled p95; core rollout only, not end-to-end latency"}
        states = [item["events"]["multi_robot_queue"]["status"], item["core_runtime"]["status"]]
        item["status"] = "INSUFFICIENT_EVIDENCE" if "INSUFFICIENT_EVIDENCE" in states else "FAIL" if "FAIL" in states else "PASS"
        if issues:
            item["status"] = "INSUFFICIENT_EVIDENCE"
        conditions.append(item)
    return {"schema_version": 1, "profile": profile, "conditions": conditions,
            "research_readiness": "NOT_ESTABLISHED", "unmeasured_requirements": UNMEASURED,
            "interpretation": "PASS applies only to the queue/core-runtime screen for the named condition. Spillback is separate. No overall readiness or arbitrary-map guarantee. Current data were seen before these pilot thresholds were chosen. Confirm on new held-out runs. Seed bootstrap assumes independent seed clusters; five seeds is a pilot minimum, not a power calculation. Undefined resampled precision/recall/F1 is conservatively zero."}


def render(report):
    esc = html.escape
    def percent(x): return "untested" if x is None else f"{100*x:.1f}%"
    rows = []
    for c in report["conditions"]:
        events = c.get("events", {})
        queue = events.get("multi_robot_queue", {})
        scores = queue.get("methods", {}).get(report["profile"]["method"], {})
        values = [c["scenario"], c["trigger"], str(c["distinct_seeds"]), percent(scores.get("precision")),
                  percent(scores.get("recall")), c["status"], events.get("spillback_warning", {}).get("status", "INSUFFICIENT_EVIDENCE")]
        rows.append("<tr>"+"".join(f"<td>{esc(v)}</td>" for v in values)+"</tr>")
    missing = "".join(f"<li>{esc(v)}</li>" for v in report["unmeasured_requirements"])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prediction research readiness</title><style>body{{font:16px/1.55 system-ui;margin:40px auto;padding:0 24px;max-width:1200px;color:#152c41}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ccd6df}}.table{{overflow:auto}}small{{color:#486077}}</style>
<h1>Prediction research readiness</h1><p>The current evidence does not establish readiness for routing decisions. Exploratory research can continue while these gaps are measured.</p>
<p>This pilot screen uses {report['profile']['horizon_s']:g}-second forecasts. Its provisional precision and recall targets are {percent(report['profile']['min_precision'])} and {percent(report['profile']['min_recall'])}. Passing requires seed-bootstrap lower confidence bounds meeting those targets and a positive paired F1 improvement over each baseline.</p>
<div class="table"><table><thead><tr><th>Workload</th><th>Safety rule</th><th>Distinct seeds</th><th>Queue precision</th><th>Queue recall</th><th>Queue and runtime screen</th><th>Spillback screen</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Percentages above are descriptive point estimates. A lack of positive examples never counts as passing. Conditions with insufficient seeds do not receive confidence intervals.</p>
<h2>Evidence still required</h2><ul>{missing}</ul><p><small>{esc(report['interpretation'])}</small></p>
<a href="report.json">Detailed measurements and thresholds</a> · <a href="READINESS.md">Evaluation protocol</a></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("readiness_profile.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payloads = [p.read_bytes() for p in args.metrics]
        report = assess_documents([json.loads(v) for v in payloads], json.loads(args.profile.read_text()))
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    report["inputs"] = [{"path": str(p.resolve()), "sha256": hashlib.sha256(v).hexdigest()} for p, v in zip(args.metrics, payloads)]
    report["assessment_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["profile_sha256"] = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.output/"index.html").write_text(render(report))
    guide = Path(__file__).with_name("READINESS.md")
    if guide.exists(): (args.output/guide.name).write_text(guide.read_text())
    print(json.dumps({"output": str(args.output), "research_readiness": report["research_readiness"],
                      "conditions": [{"scenario": c["scenario"], "trigger": c["trigger"], "status": c["status"]} for c in report["conditions"]]}, indent=2))


if __name__ == "__main__":
    main()
