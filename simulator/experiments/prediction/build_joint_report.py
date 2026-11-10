#!/usr/bin/env python3
"""Package measured forecast reports into a local interactive demo."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.prediction.joint_viewer import write_viewer


def weighted(rows, key, count):
    selected = [(r[key], r[count]) for r in rows if r.get(key) is not None and r.get(count, 0)]
    return sum(value*n for value, n in selected)/sum(n for _, n in selected) if selected else None


def pool(document):
    runs = document["runs"]
    pooled = {}
    for horizon in runs[0]["metrics"]:
        pooled[horizon] = {}
        for method in ("joint", "plan_only", "legacy", "hold"):
            rows = [r["metrics"][horizon][method] for r in runs]
            result = {"robot_endpoints": sum(r["robot_endpoints"] for r in rows),
                      "moving_robot_endpoints": sum(r["moving_robot_endpoints"] for r in rows),
                      "unexhausted_plan_robot_endpoints": sum(r["unexhausted_plan_robot_endpoints"] for r in rows)}
            for key, count in (("position_mae_m", "robot_endpoints"),
                               ("moving_position_mae_m", "moving_robot_endpoints"),
                               ("unexhausted_plan_position_mae_m", "unexhausted_plan_robot_endpoints")):
                result[key] = weighted(rows, key, count)
            for event in ("multi_robot_queue", "spillback_warning"):
                c = {k: sum(r[event][k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
                c["precision"] = c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else None
                c["recall"] = c["tp"]/(c["tp"]+c["fn"]) if c["tp"]+c["fn"] else None
                result[event] = c
            pooled[horizon][method] = result
    clearance = {}
    for method in ("joint", "plan_only", "hold"):
        rows = [r["clearance"][method] for r in runs]
        clearance[method] = {k: sum(r[k] for r in rows) for k in rows[0] if not k.endswith("mae_s")}
        clearance[method]["matched_clearance_mae_s"] = weighted(rows, "matched_clearance_mae_s", "matched_clearances")
        clearance[method]["matched_arrival_mae_s"] = weighted(rows, "matched_arrival_mae_s", "matched_arrivals")
    return {"recordings": len(runs), "forecast_origins": sum(r["forecast_origins"] for r in runs),
            "origins_with_later_plan": sum(r["origins_with_later_plan"] for r in runs),
            "horizons": pooled, "clearance": clearance,
            "max_actual_queue_proxy_robots": max(r["max_actual_queue_proxy_robots"] for r in runs),
            "mean_rollout_runtime_s": weighted([{**r["runtime_s"], "n": r["forecast_origins"]} for r in runs], "mean", "n"),
            "run_sources": [r["run_directory"] for r in runs]}


def number(value, digits=2):
    return "Unavailable" if value is None else f"{value:.{digits}f}"


def build(fleet, stress, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "conditions": {}, "source_reports": {},
               "interpretation": "Position errors are pooled by robot endpoints; moving cohort uses actual displacement >0.25m for evaluation only. Queue precision/recall pool approach-time decisions. Origin windows overlap and are not independent trials. Clearance errors are conditional on matched exits; inspect misses and false exits. The isolated delay reference remains a model, not separately measured ground truth."}
    cards, rows = [], []
    for name, source, title in (("fleet", fleet, "27 robots across nine zones"),
                                ("stress", stress, "24 robots at a narrow T-junction")):
        report_path = source/"metrics.json"
        report = json.loads(report_path.read_text())
        measured = pool(report)
        summary["conditions"][name] = measured
        summary["source_reports"][name] = {"path": str(report_path), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}
        target = output/name; target.mkdir()
        shutil.copyfile(report_path, target/"metrics.json")
        data = json.loads((source/"demo_data.json").read_text())
        # Demo navigation samples are illustrative; every origin enters metrics.
        if name == "fleet":
            data["samples"] = [s for s in data["samples"] if "seed 43" in s.get("label", "")]
        ten = measured["horizons"]["10.0"]
        data["title"] = title+" — forecast versus execution"
        data["metrics"] = {"headline": [
            {"label": "10 s moving-robot error", "value": number(ten["joint"]["moving_position_mae_m"])+" m"},
            {"label": "Previous predictor", "value": number(ten["legacy"]["moving_position_mae_m"])+" m"},
            {"label": "Evaluated forecasts", "value": str(measured["forecast_origins"])}]}
        data["links"] = [{"label": "Overview and results", "href": "../index.html"},
                         {"label": "Measured results JSON", "href": "metrics.json"},
                         {"label": "Other floor", "href": "../"+("stress" if name == "fleet" else "fleet")+"/index.html"}]
        (target/"demo_data.json").write_text(json.dumps(data, separators=(",", ":"))+"\n")
        write_viewer(target)
        cards.append(f'<a class="card" href="{name}/index.html"><span class="tag">{measured["recordings"]} recorded runs</span><h2>{escape(title)}</h2><p>Compare frozen forecasts with actual execution. Change the origin, method, and robot; play the next 20 seconds.</p><strong>Open interactive comparison →</strong></a>')
        for horizon in ("5.0", "10.0", "20.0"):
            h = measured["horizons"][horizon]
            rows.append(f'<tr><td>{escape(title)}</td><td>{int(float(horizon))} s</td><td>{number(h["legacy"]["moving_position_mae_m"])} m</td><td>{number(h["joint"]["moving_position_mae_m"])} m</td><td>{number(h["plan_only"]["moving_position_mae_m"])} m</td></tr>')
    stress_ten = summary["conditions"]["stress"]["horizons"]["10.0"]["joint"]
    queue = stress_ten["multi_robot_queue"]
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Warehouse prediction results</title>
<style>:root{{font:16px/1.6 system-ui,sans-serif;color:#172b42;background:#f2f6fa}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1150px;margin:auto;padding:40px 24px}}h1{{font-size:36px;line-height:1.2;margin:8px 0 16px}}h2{{font-size:22px;line-height:1.3}}p{{color:#526478}}.eyebrow{{font-size:12px;font-weight:750;color:#225bb1;letter-spacing:1.5px;text-transform:uppercase}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:26px 0}}.card{{background:white;border:1px solid #d4dfeb;border-radius:14px;padding:24px;text-decoration:none;color:inherit}}.card:hover{{border-color:#4076c5}}.card strong{{color:#225bb1}}.tag{{font-size:12px;color:#526478}}section{{background:white;border:1px solid #d4dfeb;border-radius:14px;padding:24px;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:12px 9px;border-bottom:1px solid #e0e7ef}}th{{background:#f3f6fa}}.table{{overflow:auto}}a{{color:#225bb1}}.fine{{font-size:13px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}h1{{font-size:28px}}main{{padding:24px 14px}}}}</style>
<main><div class="eyebrow">Measured simulator forecasts</div><h1>Predicting how warehouse traffic unfolds</h1><p>The joint predictor advances all robots together, checking safety fields and action dependencies every 0.1 seconds. Each forecast starts with a recorded execution snapshot and holds the accepted routes fixed for 20 seconds.</p>
<div class="grid">{''.join(cards)}</div>
<section><h2>How much did the forecasts improve?</h2><p>Mean position error for robots that actually moved more than 0.25 m during the forecast. This avoids letting parked robots dominate the comparison.</p><div class="table"><table><thead><tr><th>Floor</th><th>Horizon</th><th>Previous predictor</th><th>Joint predictor</th><th>Same model, safety off</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="fine">Previous predictor: fixed action durations. Safety-off comparison: the new controller-aware model with safety responses disabled, keeping its state and dependencies. These are distinct baselines.</p></section>
<section><h2>Does it detect queues?</h2><p>At 10 seconds on the narrow T-junction, queue precision was <b>{number(100*queue['precision'],1)}%</b> and recall was <b>{number(100*queue['recall'],1)}%</b>. That means {queue['tp']} correct queue predictions, {queue['fp']} false alarms, and {queue['fn']} missed queues across sampled approaches and forecast origins.</p><p>A queue here means at least two nearby robots moving slowly or waiting near a junction entry. It is a geometric and speed-based measure. No positive spillback-warning examples occurred in these runs, so spillback detection remains unvalidated.</p></section>
<section><h2>What still limits the prediction?</h2><p>Known action lists sometimes end before the forecast horizon. Robots then hold their last known position in the forecast, while actual execution can receive new tasks or routes. The replay marks this condition. Angular control, contact physics, and future replanning are also approximate or omitted.</p><p>Clearance timing and queue measurements are saved alongside position errors. An isolated model supplies a provisional excess-delay reference; it has not been validated as a separate native free-flow baseline.</p><p class="fine">Six nine-zone runs and two T-junction runs; no parameters fitted to these recordings. Samples from overlapping forecast windows are correlated. Restricted communication and prediction-driven rerouting are future work.</p><a href="summary.json">Download pooled measurements</a> · <a href="JOINT_PREDICTION.md">Implementation and reproduction</a></section></main></html>'''
    (output/"index.html").write_text(text)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    documentation = Path(__file__).parent/"JOINT_PREDICTION.md"
    if documentation.exists(): shutil.copyfile(documentation, output/documentation.name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.fleet, args.stress, args.output)
    print(args.output/"index.html")
