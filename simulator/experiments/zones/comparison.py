#!/usr/bin/env python3
"""Make a paired safety-trigger report from measured nine-zone recordings."""

import argparse
import copy
import csv
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.zones.viewer import _input_directory


VARIANTS = ("body_in_outer", "outer_overlap")
METRICS = {
    "deliveries_completed": "Delivered tasks",
    "deliveries_unfinished": "Unfinished tasks",
    "all_deliveries_completed_s": "All deliveries complete (s)",
    "mean_completed_delivery_time_s": "Mean completed task time (s)",
    "slowed_move_robot_seconds": "Slowed movement (robot·s)",
    "stopped_command_robot_seconds": "Stopped commands (robot·s)",
    "polygon_only_trigger_robot_seconds": "Polygon-only triggers (robot·s)",
    "stationary_active_robot_seconds": "Stationary active proxy (robot·s)",
    "peak_stationary_active_robots": "Peak stationary active proxy (robots)",
    "p95_stationary_active_robots": "P95 stationary active proxy (robots)",
    "queued_robot_seconds": "Approach queue proxy (robot·s)",
    "max_queue_robots": "Longest approach queue proxy (robots)",
    "max_queue_tail_extent_m": "Farthest queue tail from junction entry (m)",
    "spillback_warning_approach_seconds": "Spillback warning proxy (approach·s)",
    "max_simultaneous_spillback_warning_approaches": "Peak simultaneous spillback warnings (approaches)",
    "sampled_min_robot_separation_m": "Minimum sampled center separation (m)",
}
NOTES = [
    "This compares recorded execution on the existing nine-zone junction fixture, not the proposed 16-junction warehouse blueprint.",
    "The workload is finite: each robot performs one pickup–delivery task and then parks. Completion times begin at initialization, not at pickup.",
    "Each pair shares its seed, floor geometry, initial robot states, tasks and safety dimensions; the robot trigger rule is the only intended policy difference. Native effective settings are checked by the run analyzer.",
    "Both policies use outer polygons only, with identical slowdown factors and identical physical-obstacle detection. Geometric contact counts as overlap. These are experimental semantics, not confirmed Amazon specifications.",
    "Polygon-only trigger time includes parked robots and counts enabled observer ticks with another outer polygon overlapping but no robot body or obstacle in that observer outer polygon. It does not measure added movement delay.",
    "Stationary active counts exclude completed deliveries, turning and service bays over a two-second window. Approach queue proxies additionally require headings within 45 degrees toward the entry, a leading center within 2 m of entry and consecutive longitudinal center gaps at most 2 m. The group aggregates approach width; no lanes are inferred.",
    "Queue tail extent is measured upstream from junction entry to the rear body edge, including any empty space ahead of the leader. A spillback warning means this tail is within 0.5 m of the upstream conflict-zone exit. It does not prove upstream obstruction. These provisional operational thresholds are fixed across conditions; they are not Amazon specifications.",
    "Approach rectangles and thresholds are saved as queue_approaches.json; per-approach measurements are in approach_queues.csv. Isolated-travel delay and causal spillback are not measured.",
    "Mean completed task time excludes unfinished tasks. Compare completion counts first; equal sample means do not imply equal outcomes when unfinished work differs.",
    "Paired differences are outer-overlap minus body-in-outer. A few runs provide a demonstration, not evidence of statistical significance or large-fleet scalability.",
    "Minimum separation is sampled between robot centers; it is not a continuous collision-safety certificate. Radio loss, sensor occlusion and load-dependent physics are not modeled.",
]


def read_json(path):
    return json.loads(path.read_text())


def case_inputs(run):
    directory, config, scenario_path = _input_directory(run)
    scenario = read_json(scenario_path)
    return {"config": config, "scenario": scenario,
            "map": read_json(scenario_path.parent / scenario["map_file"]),
            "tasks": read_json(scenario_path.parent / scenario["tasks_file"])}


def without_trigger(inputs):
    normalized = copy.deepcopy(inputs)
    normalized["scenario"].get("safety", {}).pop("robot_trigger", None)
    for agent in normalized["scenario"]["agents"]:
        agent.get("safety", {}).pop("robot_trigger", None)
    return normalized


def summarize(output_dir):
    output_dir = Path(output_dir).resolve()
    errors, pairs, rows = [], [], []
    runs = {variant: {path.name: path for path in (output_dir / variant).glob("run-*") if path.is_dir()}
            for variant in VARIANTS}
    names = sorted(set(runs[VARIANTS[0]]) | set(runs[VARIANTS[1]]))
    if not names:
        errors.append("No trigger comparison runs found")
    for name in names:
        if any(name not in runs[variant] for variant in VARIANTS):
            errors.append(f"{name}: missing paired variant")
            continue
        inputs, measured, seeds = {}, {}, {}
        for variant in VARIANTS:
            run = runs[variant][name]
            try:
                measured[variant] = read_json(run / "zone_metrics.json")
                inputs[variant] = case_inputs(run)
            except (OSError, KeyError, ValueError) as error:
                errors.append(f"{variant}/{name}: {error}")
                continue
            metric = measured[variant]
            seeds[variant] = inputs[variant]["config"]["parameters"].get("seed")
            if not metric.get("valid", False):
                errors.append(f"{variant}/{name}: run validation failed")
            if metric.get("robot_trigger") != variant:
                errors.append(f"{variant}/{name}: recorded trigger does not match condition")
            if metric.get("inner_stop_enabled") is not False or not metric.get("safety_enabled"):
                errors.append(f"{variant}/{name}: expected enabled outer-only safety")
            if metric.get("seed") != seeds[variant]:
                errors.append(f"{variant}/{name}: metric seed differs from actual run configuration")
            rows.append({"run": name, "seed": seeds[variant], "variant": variant,
                         **{key: metric.get(key) for key in METRICS}})
            if not (run / "replay.html").is_file():
                errors.append(f"{variant}/{name}: replay.html is missing")
        if len(inputs) != 2:
            continue
        if seeds[VARIANTS[0]] is None or seeds[VARIANTS[0]] != seeds[VARIANTS[1]]:
            errors.append(f"{name}: matched pairs must have the same explicit seed")
        if without_trigger(inputs[VARIANTS[0]]) != without_trigger(inputs[VARIANTS[1]]):
            errors.append(f"{name}: inputs differ beyond the selected robot trigger")
        pair = {"run": name, "seed": seeds[VARIANTS[0]],
                "replays": {variant: f"{variant}/{name}/replay.html" for variant in VARIANTS},
                "values": {variant: {key: measured[variant].get(key) for key in METRICS} for variant in VARIANTS},
                "outer_minus_body": {}}
        for key in METRICS:
            body, overlap = (measured[variant].get(key) for variant in VARIANTS)
            pair["outer_minus_body"][key] = overlap - body if body is not None and overlap is not None else None
        pairs.append(pair)
    aggregate = {key: {"mean_difference": statistics.mean(values) if values else None,
                       "pairs_with_both_values": len(values)}
                 for key in METRICS
                 for values in [[pair["outer_minus_body"][key] for pair in pairs if pair["outer_minus_body"][key] is not None]]}
    report = {"valid": not errors, "errors": errors, "variants": list(VARIANTS),
              "paired_runs": len(pairs), "metric_labels": METRICS, "pairs": pairs,
              "paired_difference_summary": aggregate, "notes": NOTES,
              "unavailable_metrics": ["causal_upstream_obstruction", "delay_vs_isolated_travel"]}
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "seed", "variant", *METRICS])
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "index.html").write_text(render_report(report))
    return report


def display(value):
    return "unavailable" if value is None else f"{value:.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else html.escape(str(value))


def render_report(report):
    errors = "" if report["valid"] else '<div class="error"><strong>Comparison did not validate.</strong><ul>' + "".join(f"<li>{html.escape(error)}</li>" for error in report["errors"]) + "</ul></div>"
    sections = []
    for pair in report["pairs"]:
        metrics = "".join(f"<tr><th>{html.escape(label)}</th><td>{display(pair['values']['body_in_outer'][key])}</td><td>{display(pair['values']['outer_overlap'][key])}</td><td>{display(pair['outer_minus_body'][key])}</td></tr>" for key, label in METRICS.items())
        replays = "".join(f'<div><h3>{html.escape(variant)}</h3><a href="{html.escape(pair["replays"][variant], quote=True)}" target="_blank" rel="noopener">Open full replay</a><iframe loading="lazy" title="{html.escape(variant)} {html.escape(pair["run"])}" src="{html.escape(pair["replays"][variant], quote=True)}"></iframe></div>' for variant in VARIANTS)
        sections.append(f'<section id="{html.escape(pair["run"], quote=True)}"><h2>{html.escape(pair["run"])} · seed {display(pair["seed"])}</h2><p class="replay-help">Replays have independent controls. Scrub both to the same simulation time to compare a paired encounter.</p><div class="replays">{replays}</div><h3 class="metrics-title">Measured results · outer-overlap minus body-in-outer</h3><div class="table-scroll"><table><thead><tr><th>Metric</th><th>Body enters outer</th><th>Outer overlaps outer</th><th>Outer − body</th></tr></thead><tbody>{metrics}</tbody></table></div></section>')
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in report["notes"])
    navigation = "".join(f'<a href="#{html.escape(pair["run"], quote=True)}">{html.escape(pair["run"])} · seed {display(pair["seed"])}</a>' for pair in report["pairs"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Safety polygon comparison</title><style>
body{{font:15px system-ui,sans-serif;background:#f1f5f9;color:#172337;margin:0;padding:24px;line-height:1.55}}main{{max-width:1500px;margin:auto}}section,.intro{{background:white;padding:18px;margin:14px 0;border:1px solid #d8e0eb;border-radius:10px}}section{{scroll-margin-top:14px}}.intro{{padding:12px 18px}}.intro p{{margin:0 0 8px}}h1{{margin:0;font-size:28px}}h2{{font-size:22px;margin:0}}h3{{font-size:15px;margin:5px 0}}.subtitle{{margin:4px 0 12px;color:#526076}}.replay-help{{margin:5px 0 12px;font-size:13px;color:#526076}}nav{{display:flex;gap:8px;flex-wrap:wrap}}nav a{{padding:5px 12px;background:#dbeafe;border-radius:6px;text-decoration:none;font-size:13px}}summary{{cursor:pointer;color:#334155;font-weight:600}}details[open] summary{{margin-bottom:12px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #d8e0eb;text-align:right}}th:first-child{{text-align:left}}thead{{background:#eaf0f9}}.table-scroll{{overflow:auto}}.metrics-title{{margin:22px 0 8px}}.replays{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}iframe{{width:100%;height:700px;border:1px solid #d8e0eb;margin-top:10px;border-radius:8px}}.error{{background:#fee2e2;color:#991b1b;padding:20px}}li{{margin:8px 0}}a{{color:#1d4ed8}}@media(max-width:900px){{.replays{{grid-template-columns:1fr}}body{{padding:10px}}table{{font-size:12px}}h1{{font-size:24px}}}}
</style></head><body><main><h1>Safety polygon trigger comparison</h1><p class="subtitle">{report['paired_runs']} recorded pairs · Outer polygons only · Finite transport workload</p>{errors}<nav aria-label="Recorded pairs">{navigation}</nav><div class="intro"><p><a href="comparison.json">Full measured report (JSON)</a> · <a href="comparison.csv">Run metrics (CSV)</a></p><details><summary>Methods, metric definitions and limitations</summary><ul>{notes}</ul></details></div>{''.join(sections)}</main></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = summarize(args.output_dir)
    print(json.dumps({"valid": report["valid"], "errors": report["errors"], "paired_runs": report["paired_runs"],
                      "report": str(args.output_dir / "index.html")}, indent=2))
    sys.exit(0 if report["valid"] else 1)
