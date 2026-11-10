#!/usr/bin/env python3
"""Export portable native-motion replays and a small pilot index."""
import argparse
from html import escape
import json
from pathlib import Path

from features import REPORT
from experiments.baseline.analyze import _objects
from experiments.zones.viewer import write_viewer


def build(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete' or not manifest.get('pilot'):
        raise ValueError('Use a completed difficulty-pilot manifest')
    rows = []
    for spec in sorted(manifest['runs'], key=lambda s: (s['demand_regime'], s['trigger'], s['workload_seed'])):
        folder = Path(spec['run_directory'])
        with (folder / 'trajectories.jsonl').open('w') as output:
            for event in _objects((folder / 'simulator.log').read_text()):
                if event.get('type') == 'tick':
                    output.write(json.dumps(event, separators=(',', ':')) + '\n')
        replay = write_viewer(folder)
        text = replay.read_text()
        text = text.replace("Initially ${data.network.robots_per_zone||'?'} robots per zone.",
                            "Starting counts by zone: ${(data.network.initial_robot_matrix||[]).flat().join(', ')}.")
        replay.write_text(text)
        metrics = json.loads((folder / 'decision_metrics.json').read_text())
        label = escape(spec['demand_regime'].replace('_', ' '))
        link = escape(str(replay.relative_to(manifest_path.parent)), quote=True)
        layout = escape(str((folder / 'layout.svg').relative_to(manifest_path.parent)), quote=True)
        rows.append(f'<tr><td>{label}</td><td>{spec["workload_seed"]}</td><td>{escape(spec["trigger"])}</td>'
                    f'<td>{metrics["deliveries_completed"]}/27</td><td>{metrics["horizon_s"]:.0f}</td>'
                    f'<td><a href="{link}">Replay</a> · '
                    f'<a href="{layout}">Layout</a></td></tr>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Congestion development pilots</title><style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#17263b;background:#f5f7fa}
h1{font-size:28px}p{line-height:1.6;max-width:950px}table{background:white;border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;border-bottom:1px solid #dce3eb;padding:12px}th{background:#eaf0f7}a{color:#174fb4}
</style><h1>Congestion development pilots</h1>
<p>These are recorded native simulations on the same floor, with 27 robots and fixed safety fields.
Concentrated cases alter task assignments and starting locations; “compact” describes geometric staging,
not guaranteed simultaneous arrival. All cases use the same planner with zero prediction penalties.</p>
<p>Open a replay, press Play, or scrub the timeline. Safety polygons and reported blockers are available for inspection.
Each run retains scenario.json, tasks.json, trajectories.jsonl, and simulator.log for later congestion models.
The 12 and 24 labels count tasks aimed through the center; every scenario contains 27 robots.
These are scenario-development recordings, with no adopted congestion definition or predictor-usefulness claim.</p>
<table><thead><tr><th>Demand</th><th>Seed</th><th>Safety rule</th><th>Delivered</th><th>Duration (s)</th>
<th>View</th></tr></thead><tbody>'''
    html += '\n'.join(rows) + '</tbody></table></html>\n'
    output = manifest_path.parent / 'index.html'
    output.write_text(html)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    print(build(parser.parse_args().manifest))
