#!/usr/bin/env python3
"""Export fleet-size recordings to portable replays, without congestion metrics."""
from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import quote

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parent / 'simulator'
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.baseline.analyze import _objects
from experiments.zones.viewer import HTML, build_replay_data


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_run(manifest_path, spec):
    """Allow a downloaded remote manifest to keep its original absolute paths."""
    recorded = Path(spec['run_directory'])
    candidates = [recorded if recorded.is_absolute() else manifest_path.parent / recorded,
                  manifest_path.parent / spec.get('run_id', recorded.name),
                  manifest_path.parent / recorded.name]
    for folder in candidates:
        if (folder / 'config.json').is_file() and (folder / 'execution_summary.json').is_file():
            return folder.resolve()
    raise FileNotFoundError(f"No downloaded/recorded input directory for {spec.get('run_id', recorded.name)}")


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return value


def export_run(folder, spec):
    """Validate actual N in every recorded frame before writing a replay."""
    config = json.loads((folder / 'config.json').read_text())
    scenario = json.loads((folder / config.get('scenario', 'scenario.json')).read_text())
    tasks = json.loads((folder / scenario.get('tasks_file', 'tasks.json')).read_text())['tasks']
    summary = json.loads((folder / 'execution_summary.json').read_text())
    ids = sorted(a['id'] for a in scenario['agents'])
    count = len(ids)
    if count == 0 or len(set(ids)) != count:
        raise ValueError('A nonempty fleet must have unique robot IDs')
    if integer(config['parameters']['num_agents'], 'Configured fleet') != count:
        raise ValueError('Configured fleet differs from scenario robots')
    if 'num_agents' in spec and integer(spec['num_agents'], 'Manifest fleet') != count:
        raise ValueError('Manifest fleet differs from actual scenario')
    if integer(summary['num_agents'], 'Summarized fleet') != count:
        raise ValueError('Execution summary fleet differs from actual scenario')
    if sorted(t['robot_id'] for t in tasks) != ids:
        raise ValueError('Expected one assigned finite task for every robot')
    if integer(summary['task_count'], 'Summarized tasks') != len(tasks):
        raise ValueError('Execution summary task count differs from assigned tasks')
    delivered = integer(summary['deliveries_completed'], 'Completed deliveries')
    pickups = integer(summary['pickups_completed'], 'Completed pickups')
    planner_calls = integer(summary['native_plan_count'], 'Native planner calls')
    planner_failures = integer(summary['native_plan_failures'], 'Unsuccessful native planner calls')
    if planner_failures > planner_calls:
        raise ValueError('Unsuccessful planner calls exceed total native planner calls')
    if not 0 <= delivered <= pickups <= len(tasks):
        raise ValueError('Acknowledged pickup/delivery counts are inconsistent')
    tps = config['parameters']['ticks_per_second']
    expected_frames = integer(config['parameters']['sim_duration'], 'Configured ticks')
    if not isinstance(tps, (int, float)) or not math.isfinite(tps) or tps <= 0:
        raise ValueError('Tick rate must be positive and finite')
    duration = expected_frames / tps
    if not math.isclose(summary['native_duration_s'], duration, abs_tol=1e-8):
        raise ValueError('Summary duration differs from configured native horizon')
    frames = []
    for event in _objects((folder / 'simulator.log').read_text(errors='replace')):
        if event.get('type') != 'tick':
            continue
        if integer(event['clock'], 'Pose clock') != len(frames):
            raise ValueError('Native pose clocks must cover the entire configured run in order')
        if sorted(a['id'] for a in event['agents']) != ids:
            raise ValueError(f'Frame {event["clock"]} does not contain exactly {count} distinct robots')
        if any(not math.isfinite(a[k]) for a in event['agents'] for k in ('x', 'y', 'rz')):
            raise ValueError('Native replay poses must be finite')
        frames.append(event)
    if len(frames) != expected_frames or integer(summary['native_frames'], 'Summarized frames') != len(frames):
        raise ValueError('Incomplete native recording; do not make it look like a complete replay')
    trajectory_path = folder / 'trajectories.jsonl'
    with trajectory_path.open('w') as output:
        for frame in frames:
            output.write(json.dumps(frame, separators=(',', ':'), allow_nan=False) + '\n')
    data = build_replay_data(folder)
    raw_pickups = sum(any(event[1] == 'pickup' for event in history) for history in data['payload_history'].values())
    raw_deliveries = sum(any(event[1] == 'dropoff' for event in history) for history in data['payload_history'].values())
    if (raw_pickups, raw_deliveries) != (pickups, delivered):
        raise ValueError('Replay task acknowledgements disagree with execution_summary.json')
    # The imported renderer supports arbitrary IDs/fleet sizes. All changes
    # below affect this new export only; the original viewer stays untouched.
    if data['payload_source'] == 'native goal_reached acknowledgements':
        for history in data['payload_history'].values():
            for event in history:
                event[0] += 1
        data['payload_clock_shift_ticks'] = 1
        data['payload_source'] += ' (shown one physical tick after server label)'
    data['execution_summary'] = summary
    data['manifest_spec'] = {key: spec[key] for key in ('run_id', 'num_agents', 'workload_seed', 'trigger', 'demand_regime') if key in spec}
    encoded = json.dumps(data, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    encoded = encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    text = HTML.replace('__DATA__', encoded)
    text = text.replace('<title>Warehouse simulation replay</title>', f'<title>{count}-robot warehouse replay</title>')
    text = text.replace('<h1>Nine-zone warehouse replay</h1>', f'<h1>{count}-robot warehouse replay</h1>')
    text = text.replace("Initially ${data.network.robots_per_zone||'?'} robots per zone.",
                        "${data.agents.length} robots total. Starting counts by zone: ${data.zones.map(z=>`${z.id}: ${data.agents.filter(a=>a.zone_id===z.id).length}`).join('; ')}.")
    replay = folder / 'replay.html'
    replay.write_text(text)
    exported = {
        'num_agents': count, 'task_count': len(tasks), 'deliveries_completed': delivered,
        'native_duration_s': duration, 'native_frames': len(frames),
        'native_plan_failures': planner_failures, 'native_plan_count': planner_calls,
        'payload_display_clock_offset_ticks': data.get('payload_clock_shift_ticks', 0),
        'source_hashes': {name: sha256(folder / name) for name in ('config.json', 'scenario.json', 'tasks.json', 'simulator.log', 'execution_summary.json')},
        'exporter_sha256': sha256(__file__),
        'scope': 'Actual recorded poses, safety reports and task acknowledgements. No congestion model or acceptance criteria.',
    }
    (folder / 'replay_export.json').write_text(json.dumps(exported, indent=2, allow_nan=False) + '\n')
    return replay, exported


def link(path, parent, label):
    href = quote(Path(os.path.relpath(path, parent)).as_posix(), safe='/')
    return f'<a href="{escape(href, quote=True)}">{escape(label)}</a>'


def build(manifest_path):
    """Build all completed fleet replays and return the portable index path."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete' or not manifest.get('pilot'):
        raise ValueError('Use a completed fleet-pilot manifest')
    specs = manifest['runs']
    if not specs or manifest.get('expected_runs', len(specs)) != len(specs):
        raise ValueError('Manifest run count is incomplete')
    prepared = []
    map_hashes = set()
    for spec in specs:
        folder = resolve_run(manifest_path, spec)
        replay, summary = export_run(folder, spec)
        config = json.loads((folder / 'config.json').read_text())
        scenario = json.loads((folder / config.get('scenario', 'scenario.json')).read_text())
        map_hashes.add(sha256(folder / scenario['map_file']))
        prepared.append((spec, folder, replay, summary))
    if len(map_hashes) != 1:
        raise ValueError('The matched fleet-size index requires exactly the same map for every run')
    rows = []
    for spec, folder, replay, summary in sorted(prepared, key=lambda row: (row[3]['num_agents'], row[0]['workload_seed'], row[0]['trigger'])):
        links = [link(replay, manifest_path.parent, 'Replay')]
        if (folder / 'layout.svg').exists():
            links.append(link(folder / 'layout.svg', manifest_path.parent, 'Map'))
        links.extend([link(folder / 'scenario.json', manifest_path.parent, 'Robots'),
                      link(folder / 'tasks.json', manifest_path.parent, 'Tasks'),
                      link(folder / 'trajectories.jsonl', manifest_path.parent, 'Trajectory'),
                      link(folder / 'execution_summary.json', manifest_path.parent, 'Execution')])
        rows.append(f'<tr><td>{summary["num_agents"]}</td><td>{escape(str(spec["workload_seed"]))}</td>'
                    f'<td>{escape(spec["trigger"])}</td><td>{summary["deliveries_completed"]}</td>'
                    f'<td>{summary["task_count"]}</td><td>{summary["native_duration_s"]:g}</td>'
                    f'<td>{summary["native_plan_failures"]}/{summary["native_plan_count"]}</td>'
                    f'<td>{" · ".join(links)}</td></tr>')
    fleets = sorted({row[3]['num_agents'] for row in prepared})
    fleet_label = ', '.join(str(value) for value in fleets)
    text = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Warehouse fleet-size recordings</title><style>
body{font:16px system-ui,sans-serif;max-width:1240px;margin:32px auto;padding:0 20px;color:#17263b;background:#f5f7fa}
h1{font-size:29px}p{line-height:1.6;max-width:1050px}.table-scroll{overflow:auto}table{background:white;border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;border-bottom:1px solid #dce3eb;padding:12px}th{background:#eaf0f7}a{color:#174fb4}.note{font-size:13px;color:#52627a}
</style></head><body><h1>Warehouse recordings with different fleet sizes</h1>
'''
    text += f'<p>These recordings contain <strong>{escape(fleet_label)} actual robots</strong>, respectively, on the same floor. Every robot has an assigned pickup-and-delivery task. The fleet count is checked against every native trajectory frame; these numbers do not count only selected crossing tasks.</p>\n'
    text += '''<p>Open a replay and press Play, or scrub the timeline. Select a robot to inspect its path, payload and recorded safety state. The safety polygons can be shown for all robots, only the selected robot, or hidden.</p>
<p>All runs use one common map with additional private bays, the same aisle network and safety geometry, and zero prediction penalties. The common map is also used for the smaller fleet, so a fleet-size comparison does not change the floor underneath it. These are configurable scenario recordings; no congestion model or acceptance rule is imposed.</p>
<div class="table-scroll"><table><thead><tr><th>Fleet count</th><th>Seed</th><th>Safety rule</th><th>Delivered</th><th>Total tasks</th><th>Duration (s)</th><th>Unsuccessful PBS calls / total</th><th>Open</th></tr></thead><tbody>
'''
    text += '\n'.join(rows) + '''</tbody></table></div>
<p class="note">Delivered counts come from native task acknowledgements through the configured horizon. Remaining tasks stay visible; they are not dropped from the table. Unsuccessful planner calls are retained as raw execution status, so delayed deliveries should not be attributed solely to safety interactions. Duration is simulated time, not recording wall time. Replays are self-contained HTML files and need no simulator or server to open. Keep the run folders beside this index for its links to work.</p></body></html>\n'''
    index = manifest_path.parent / 'index.html'
    index.write_text(text)
    (manifest_path.parent / 'replay_index.json').write_text(json.dumps({
        'fleets': fleets, 'runs': len(prepared), 'common_map_sha256': next(iter(map_hashes)),
        'manifest_sha256': sha256(manifest_path), 'exporter_sha256': sha256(__file__),
        'scope': 'Recorded fleet-size scenarios, without congestion-model selection or acceptance gates.'}, indent=2) + '\n')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    print(build(parser.parse_args().manifest))
