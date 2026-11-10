#!/usr/bin/env python3
"""Plot fresh native fleet recordings and predictor scores without report archives.

Workload ranges are descriptive. This script does not select a predictor or
claim significance from the plotted differences.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'zero': ('Zero', '#a3a3a3'), 'persistence': ('Persistence', '#64748b'),
    'kinematic': ('Plan projection', '#d97706'), 'joint': ('Joint rollout', '#2563eb'),
    'local_ridge': ('Local ridge', '#15803d'), 'convlstm': ('ConvLSTM', '#7c3aed'),
    'residual_convlstm': ('Residual ConvLSTM', '#be185d')}
RULES = {'body_in_outer': 'Safety rule 1', 'outer_overlap': 'Safety rule 2'}
COLORS = {27: '#2563eb', 54: '#d97706', 81: '#7c3aed'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def read(path):
    return json.loads(Path(path).read_text())


def save(figure, output, name):
    for suffix in ('pdf', 'png'):
        figure.savefig(output / f'{name}.{suffix}', bbox_inches='tight', dpi=180)
    plt.close(figure)


def native_runs(paths):
    runs, seen = [], set()
    for path in paths:
        manifest = read(path)
        if manifest['status'] != 'complete':
            raise ValueError(f'Incomplete native study: {path}')
        for entry in manifest['runs']:
            folder = path.parent / entry['relative_directory']
            summary, scenario = read(folder/'execution_summary.json'), read(folder/'scenario.json')
            if entry['run_id'] in seen:
                raise ValueError('Duplicate native run IDs')
            seen.add(entry['run_id'])
            if summary['num_agents'] != len(scenario['agents']) or summary['num_agents'] != entry['num_agents']:
                raise ValueError('Recording fleet size differs from manifest')
            if summary['native_frames'] != round(summary['native_duration_s'] * 10):
                raise ValueError('Recording frame count differs from duration')
            runs.append({**entry, 'summary': summary, 'scenario': scenario,
                         'map': read(folder/'map.json'), 'directory': folder})
    if not runs or len({r['summary']['native_duration_s'] for r in runs}) != 1:
        raise ValueError('Need completed recordings with one shared horizon')
    if len({json.dumps(r['map'], sort_keys=True) for r in runs}) != 1:
        raise ValueError('Fleet comparison requires the same floor')
    return runs


def map_figure(runs, output):
    counts = sorted({r['num_agents'] for r in runs})
    seed = min(r['workload_seed'] for r in runs)
    selected = [next(r for r in runs if r['num_agents'] == n and r['workload_seed'] == seed) for n in counts]
    figure, axes = plt.subplots(1, len(counts), figsize=(3.7*len(counts), 3.8), squeeze=False,
                                constrained_layout=True)
    for axis, run in zip(axes.ravel(), selected):
        floor = np.array([[char not in '@T' for char in row] for row in run['map']['layout']])
        axis.imshow(np.where(floor, 1., .8), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        for zone in run['scenario']['zones']:
            row, col = zone['center']
            axis.add_patch(Rectangle((col-1.5, row-1.5), 3, 3, facecolor='#fde68a', edgecolor='none'))
        positions = np.array([a['start'] for a in run['scenario']['agents']])
        axis.scatter(positions[:, 1], positions[:, 0], s=12, c=COLORS.get(run['num_agents'], '#2563eb'),
                     edgecolors='white', linewidths=.3)
        axis.set(title=f"{run['num_agents']} robots", xticks=[], yticks=[], aspect='equal')
    save(figure, output, 'fleet_maps')


def native_figure(runs, output):
    horizon = runs[0]['summary']['native_duration_s']
    groups = defaultdict(list)
    for run in runs:
        groups[run['trigger'], run['num_agents']].append(run)
    rules = [rule for rule in RULES if any(key[0] == rule for key in groups)]
    counts = sorted({r['num_agents'] for r in runs})
    figure, axes = plt.subplots(1, len(rules), figsize=(5.2*len(rules), 3.8), squeeze=False,
                                constrained_layout=True)
    grid = np.arange(round(horizon*10)+1)/10
    for axis, rule in zip(axes.ravel(), rules):
        for count in counts:
            curves = []
            for run in groups[rule, count]:
                times = sorted(t['delivery_completed_s'] + .1 for t in run['summary']['task_outcomes']
                               if t.get('delivery_completed_s') is not None)
                curve = np.searchsorted(times, grid, side='right')
                curves.append(curve)
                axis.step(grid, curve, where='post', color=COLORS.get(count), alpha=.2, lw=.7)
            if curves:
                axis.step(grid, np.mean(curves, axis=0), where='post', color=COLORS.get(count),
                          lw=2, label=f'{count} robots')
        axis.set(title=RULES[rule], xlabel='Simulated time (s)', ylabel='Completed deliveries',
                 xlim=(0, horizon), ylim=(0, None))
        axis.grid(alpha=.15); axis.legend(frameon=False)
    save(figure, output, 'fleet_execution')
    values = [{key: r[key] for key in ('run_id', 'workload_seed', 'num_agents', 'trigger')} |
              {key: r['summary'][key] for key in ('deliveries_completed', 'native_plan_failures', 'native_plan_count')}
              for r in runs]
    (output/'fleet_native_plot_data.json').write_text(json.dumps({'horizon_s': horizon, 'runs': values}, indent=2)+'\n')


def score(metrics, region):
    values = [w['channels']['slow_occupancy']['regions'][region]['rmse'] for w in metrics]
    return float(np.mean(values)) if all(v is not None for v in values) else np.nan


def prediction_figures(path, output):
    data = read(path)
    if data['status'] != 'complete':
        raise ValueError('Prediction evaluation is incomplete')
    groups = sorted([g for g in data['groups'] if g['trigger'] == 'outer_overlap'], key=lambda g: g['num_agents'])
    if not groups:
        raise ValueError('Need Safety rule 2 predictions for fleet comparison')
    counts = [g['num_agents'] for g in groups]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4), constrained_layout=True)
    for axis, region, title in zip(axes, ('roi', 'positive_roi'), ('Junctions and approaches', 'Actual slow-occupied cells')):
        for method, (label, color) in METHODS.items():
            if not all(method in g['metrics'] for g in groups):
                continue
            values = [score(g['metrics'][method], region) for g in groups]
            axis.plot(counts, values, marker='o', color=color, label=label)
            for run in data['runs']:
                if run['trigger'] == 'outer_overlap' and method in run['metrics']:
                    axis.scatter(run['num_agents'], score(run['metrics'][method], region),
                                 s=13, alpha=.3, color=color)
        axis.set(title=title, xlabel='Robots', ylabel='Slow-occupancy RMSE (robots/cell)',
                 xticks=counts, ylim=(0, None)); axis.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    save(figure, output, 'fleet_prediction_errors')


def runtime_figures(path, output):
    data = read(path)
    if data['status'] != 'complete':
        raise ValueError('Timing study is incomplete')
    groups = sorted([g for g in data['groups'] if g['backend'] == 'cpu'], key=lambda g: g['num_agents'])
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4), constrained_layout=True)
    for axis, field, title in zip(axes, ('p50_s', 'p95_s'), ('Median query', '95th-percentile query')):
        for method, (label, color) in METHODS.items():
            if all(method in g['methods'] for g in groups):
                axis.plot([g['num_agents'] for g in groups],
                          [1000*g['methods'][method][field] for g in groups],
                          marker='o', color=color, label=label)
        axis.set(title=title, xlabel='Robots', ylabel='Wall time (ms)', yscale='log',
                 xticks=[g['num_agents'] for g in groups]); axis.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    save(figure, output, 'fleet_prediction_runtime')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, default=ROOT/'results/fleet_prediction/evaluation.json')
    parser.add_argument('--runtime', type=Path, default=ROOT/'results/fleet_prediction/runtime.json')
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = native_runs(args.manifests)
    map_figure(runs, args.output); native_figure(runs, args.output)
    inputs = list(args.manifests)
    if not args.native_only:
        prediction_figures(args.evaluation, args.output); runtime_figures(args.runtime, args.output)
        inputs += [args.evaluation, args.runtime]
    (args.output/'fleet_plot_sources.json').write_text(json.dumps({
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'note': 'Workload points/ranges are descriptive. Cell errors average the three future windows. Runtime is measured separately.'
    }, indent=2)+'\n')
    print(f'Built fleet figures from {len(runs)} completed recordings in {args.output}')


if __name__ == '__main__':
    main()
