"""An explicitly selected candidate-route illustration, never realized future truth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _route(robot, pose):
    """Polyline from current pose along unfinished MOVE goals in its preview.

Only initially already-passed collinear goals are omitted from the drawing;
their acknowledgements may still be pending. No route is simulated or repaired.
"""
    current = np.array([-pose['x'], -pose['y']], dtype=float)
    points = [current]
    at_origin = True
    for action in sorted(robot['actions'], key=lambda a: a['node_id']):
        if action['node_id'] <= robot['finished_node_id'] or action['type'] != 'M':
            continue
        start, goal = np.asarray(action['start']), np.asarray(action['goal'])
        segment = goal - start
        length = np.linalg.norm(segment)
        if at_origin and length > 1e-9:
            along = np.dot(current - start, segment) / length
            delta = current - start
            cross = abs(segment[0] * delta[1] - segment[1] * delta[0]) / length
            if along >= length - 1e-9 and cross <= .15:
                continue
        at_origin = False
        if np.linalg.norm(goal - points[-1]) > 1e-9:
            points.append(goal.astype(float))
    return np.asarray(points)


def _spatial_tasks(robot):
    result = []
    for action in sorted(robot['actions'], key=lambda a: a['node_id']):
        if action['node_id'] <= robot['finished_node_id'] or action['type'] not in ('M', 'S'):
            continue
        item = action['type'], tuple(action['goal']), action.get('task_id', -1)
        if not result or result[-1] != item:
            result.append(item)
    return result


def choose_example(studies):
    choices = []
    for study in studies:
        for row in study['rows']:
            if row['method'] != 'rollout' or not row['changed_route_geometry_reviews']:
                continue
            folder = Path(row.get('_study_path', study['path'])) / row['run_id']
            for line in (folder / 'host_queries.jsonl').read_text().splitlines():
                query = json.loads(line)
                selected = query['selected_index']
                if not selected:
                    continue
                request = folder / 'plan_search' / f"request_{query['clock']}.json"
                batch = _read(request)
                zero = {r['id']: r for r in batch['candidates'][0]['preview_snapshot']['robots']}
                modified = {r['id']: r for r in batch['candidates'][selected]['preview_snapshot']['robots']}
                changed = [rid for rid in sorted(zero) if _spatial_tasks(zero[rid]) != _spatial_tasks(modified[rid])]
                if changed:
                    choices.append(dict(study=study, row=row, folder=folder, query=query, batch=batch,
                                        changed_ids=changed, request=request,
                                        gain=query['selection']['selected_mean_eta_gain_s']))
    if not choices:
        return None
    return min(choices, key=lambda item: (-item['gain'], item['study']['name'],
                                         item['row']['num_agents'], item['row']['workload_seed'],
                                         item['query']['clock']))


def make_preview_figure(studies, output):
    """Return provenance for a deterministic example, or None if none exists."""
    example = choose_example(studies)
    if example is None:
        return None
    query, batch, folder = example['query'], example['batch'], example['folder']
    selected = query['selected_index']
    zero_eta = query['selection']['candidates'][0]['eta']['per_robot_eta_s']
    selected_eta = query['selection']['candidates'][selected]['eta']['per_robot_eta_s']
    gains = {rid: zero_eta[str(rid)] - selected_eta[str(rid)] for rid in example['changed_ids']}
    shown = sorted(example['changed_ids'], key=lambda rid: (-gains[rid], rid))[:5]
    map_doc = _read(folder / 'map.json')
    road = np.array([[cell not in '@T' for cell in row] for row in map_doc['layout']])
    poses = {pose['id']: pose for pose in batch['pose_frame']['agents']}
    zero = batch['candidates'][0]
    winner = batch['candidates'][selected]
    targets = query['selection']['candidates'][0]['eta']['diagnostics']['targets']
    colors = ['#2563eb', '#ea580c', '#16a34a', '#9333ea', '#dc2626']
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.8), constrained_layout=True)
    for ax, candidate, title in zip(axes, (zero, winner), ('Zero-cost PBS preview', f"Selected {winner['name']}-cost PBS preview")):
        ax.imshow(np.where(road, 1., .77), vmin=0, vmax=1, cmap='gray', origin='upper',
                  extent=(-.5, road.shape[1] - .5, road.shape[0] - .5, -.5), interpolation='nearest')
        ax.scatter([-p['y'] for p in poses.values()], [-p['x'] for p in poses.values()],
                   s=9, color='#94a3b8', alpha=.65, zorder=2)
        robots = {robot['id']: robot for robot in candidate['preview_snapshot']['robots']}
        for rid, color in zip(shown, colors):
            points = _route(robots[rid], poses[rid])
            ax.plot(points[:, 1], points[:, 0], color=color, lw=1.7, alpha=.9, zorder=3)
            ax.scatter(points[0, 1], points[0, 0], s=32, facecolors='white', edgecolors=color,
                       linewidths=1.5, zorder=5)
            goal = targets[str(rid)]['services'][-1]['goal']
            ax.scatter(goal[1], goal[0], s=72, color=color, marker='*', edgecolors='white',
                       linewidths=.35, zorder=5)
            if len(points) > 2:
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
                midpoint = min(int(np.searchsorted(np.cumsum(lengths), lengths.sum() / 2)), len(points) - 2)
                start, end = points[midpoint], points[midpoint + 1]
                direction = end - start
                if np.linalg.norm(direction) > 1e-9:
                    end = start + direction / np.linalg.norm(direction) * min(1., np.linalg.norm(direction))
                    ax.annotate('', xy=(end[1], end[0]), xytext=(start[1], start[0]),
                                arrowprops=dict(arrowstyle='-|>', color=color, lw=1., mutation_scale=9), zorder=4)
        ax.set(title=title, xlabel='Map column (m)', ylabel='Map row (m)', aspect='equal',
               xlim=(-.5, road.shape[1] - .5), ylim=(road.shape[0] - .5, -.5))
        ax.set_xticks([0, 10, 30, 50, 60]); ax.set_yticks([0, 10, 30, 50, 60])
    row = example['row']
    fig.suptitle(f"Candidate previews at {query['clock'] / 10:g} s · {row['num_agents']} robots · workload {row['workload_seed']}\n"
                 f"Predicted fleet-mean ETA improvement: {example['gain']:.2f} s", fontsize=10)
    legend = [Line2D([], [], color=color, lw=2, label=f"Robot {rid} (ETA gain {gains[rid]:+.1f} s)")
              for rid, color in zip(shown, colors)]
    legend += [Line2D([], [], color='#475569', marker='o', markerfacecolor='white', ls='', label='Observed start'),
               Line2D([], [], color='#475569', marker='*', ls='', markersize=9, label='Common terminal service')]
    fig.legend(handles=legend, loc='outside lower center', ncol=3, fontsize=8, frameon=False)
    for suffix in ('pdf', 'png'):
        fig.savefig(Path(output).with_suffix('.' + suffix), dpi=210)
    plt.close(fig)
    files = [folder / 'map.json', folder / 'scenario.json', folder / 'tasks.json', folder / 'run_spec.json',
             folder / 'host_queries.jsonl', example['request'],
             folder / 'plan_search' / f"response_{query['clock']}.json",
             folder / 'bridge' / f"request_{query['clock']}.json",
             folder / 'bridge' / f"response_{query['clock']}.json"]
    return dict(study=row.get('_study_name', example['study']['name']), run_id=row['run_id'], clock=query['clock'],
                num_agents=row['num_agents'], workload_seed=row['workload_seed'],
                selected_index=selected, selected_name=winner['name'],
                predicted_mean_eta_gain_s=example['gain'], changed_robot_ids=example['changed_ids'],
                displayed_robot_ids=shown, displayed_robot_eta_gains_s={str(rid): gains[rid] for rid in shown},
                selection_rule='Among selected rollout branches with changed spatial paths, largest predicted fleet-mean ETA gain; up to five changed robots ranked by individual ETA gain; stable ID/time tie-breaks.',
                scope='Candidate preview polylines at a shared frozen origin; not realized future trajectories.',
                source_sha256={str(path.relative_to(folder)): _digest(path) for path in files},
                manifest_sha256=_digest(folder.parent / 'manifest.json'),
                builder_sha256=_digest(__file__))
