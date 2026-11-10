#!/usr/bin/env python3
"""Analyze matched native safety ablations; never rerun a simulation."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
sys.path.insert(0, str(REPORT / 'code'))
from features import pose_array, rasterize
from fleet_execution import objects, digest, write_json
from trigger_diagnostics import analyze_triggers

CONDITIONS = ('disabled', 'body_in_outer', 'outer_overlap')
LABELS = {'disabled': 'No safety rule', 'body_in_outer': 'Safety rule 1',
          'outer_overlap': 'Safety rule 2'}
COLORS = {'disabled': '#64748b', 'body_in_outer': '#d97706', 'outer_overlap': '#2563eb'}


def analyze_run(directory):
    metric_path = directory / 'route_metrics.json'
    sources = {name: digest(directory / name) for name in
               ('route_metrics.json', 'simulator.log', 'scenario.json', 'map.json',
                'assignment.json', 'config.json', 'tasks.json')}
    sources['analysis.py'] = digest(__file__)
    sources['trigger_diagnostics.py'] = digest(HERE / 'trigger_diagnostics.py')
    cache = directory / 'safety_analysis.json'
    raster_path = directory / 'slow_maps.npz'
    if cache.exists() and raster_path.exists():
        old = json.loads(cache.read_text())
        if old['source_sha256'] == sources:
            return old
    metrics = json.loads(metric_path.read_text())
    scenario = json.loads((directory / 'scenario.json').read_text())
    map_doc = json.loads((directory / 'map.json').read_text())
    assignment = json.loads((directory / 'assignment.json').read_text())
    states = np.array([pose_array(e) for e in objects(directory / 'simulator.log') if e['type'] == 'tick'])
    expected = metrics['horizon_s'] * 10
    if len(states) != expected:
        raise ValueError('Incomplete pose history')
    shape = (len(map_doc['layout']), len(map_doc['layout'][0]))
    rasters, slow = rasterize(states, scenario, shape, dt=.1)
    outcomes = metrics['task_outcomes']
    completion = np.array([metrics['horizon_s'] if r['delivery_completed_s'] is None
                           else r['delivery_completed_s'] for r in outcomes])
    ids = [r['robot_id'] for r in outcomes]
    if ids != list(range(len(ids))):
        raise ValueError('Task outcomes must use canonical agent ordering')
    # Integer physical ticks avoid a floating-point boundary error. Native
    # acknowledgements precede the matching physical observation by one tick.
    physical_completion = np.array([expected if r['delivery_completed_s'] is None
                                    else round(r['delivery_completed_s'] * 10) + 1 for r in outcomes])
    active = np.arange(len(states))[:, None] < physical_completion[None, :]
    active_slow = slow & active
    active_map = np.zeros(shape)
    for t in range(20, len(states)):
        rc = np.floor(states[t, active_slow[t], :2] + .5).astype(int)
        np.add.at(active_map, (rc[:, 0], rc[:, 1]), .1)
    duration = metrics['horizon_s'] - 2
    all_slow_map = rasters[20:, 1].sum(axis=0) * .1 / duration
    active_map /= duration
    counters = analyze_triggers(directory)
    original = {k: metrics[k] for k in ('run_id', 'workload_seed', 'num_agents', 'safety_condition',
                'horizon_s', 'deliveries_completed', 'restricted_mean_flow_s', 'completion_fraction',
                'primary_planner_failures', 'planning_calls', 'recording_wall_s')}
    row = {**original, 'prefix27_mean_flow_s': float(completion[:27].mean()),
           'slow_robot_s': float(slow.sum() * .1),
           'active_slow_robot_s': float(active_slow.sum() * .1),
           'slow_s_per_robot': float(slow.sum() * .1 / len(ids)),
           'active_slow_s_per_robot': float(active_slow.sum() * .1 / len(ids)),
           'geometric_lower_bound_mean_s': float(np.mean([r['geometric_transport_lower_bound_s']
                                                         for r in assignment['assignments']])),
           'task_outcomes': outcomes, 'diagnostics': counters, 'source_sha256': sources}
    np.savez_compressed(raster_path, slow=all_slow_map, active_slow=active_map)
    write_json(cache, row)
    print('ANALYZED', row['run_id'], flush=True)
    return row


def grouped(rows):
    return {(n, c): [r for r in rows if r['num_agents'] == n and r['safety_condition'] == c]
            for n in (27, 54, 81) for c in CONDITIONS}


def line_panel(ax, groups, field, ylabel):
    for c in CONDITIONS:
        triples = [(n, [r[field] for r in groups[n, c]]) for n in (27, 54, 81) if groups[n, c]]
        x = [n for n, _ in triples]
        if not x:
            continue
        mean = np.array([np.mean(v) for _, v in triples])
        low = np.array([np.min(v) for _, v in triples])
        high = np.array([np.max(v) for _, v in triples])
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], color=COLORS[c],
                    marker='o', capsize=4, lw=2, label=LABELS[c])
    ax.set(xlabel='Robots sharing the map', ylabel=ylabel, xticks=[27, 54, 81])
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)


def save_figure(fig, name, output):
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plots(rows, root, output):
    groups = grouped(rows)
    horizon = rows[0]['horizon_s']
    calls = rows[0]['planning_calls']
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), layout='constrained')
    for ax, field, label in zip(axes, ['deliveries_completed', 'restricted_mean_flow_s', 'active_slow_s_per_robot'],
                                [f'Deliveries by {horizon:g} s', 'Mean capped completion time (s)', 'Active slow seconds per robot']):
        line_panel(ax, groups, field, label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    save_figure(fig, 'outcomes', output)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    line_panel(axes[0], groups, 'prefix27_mean_flow_s', 'Same first 27 tasks: mean capped time (s)')
    line_panel(axes[1], groups, 'primary_planner_failures', f'Unsuccessful PBS calls / {calls}')
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    save_figure(fig, 'load_and_planning', output)
    first = root / rows[0]['run_id']
    layout = json.loads((first / 'map.json').read_text())['layout']
    floor = np.array([[v not in '@T' for v in line] for line in layout])
    maps = {}
    for key, values in groups.items():
        if values:
            maps[key] = np.mean([np.load(root/r['run_id']/'slow_maps.npz')['active_slow'] for r in values], axis=0)
    vmax = max(float(m.max()) for m in maps.values()) or 1
    fig, axes = plt.subplots(3, 3, figsize=(9.4, 8.8), layout='constrained')
    for row_index, c in enumerate(CONDITIONS):
        for col_index, n in enumerate((27, 54, 81)):
            ax = axes[row_index, col_index]
            ax.imshow(floor, cmap='gray', vmin=0, vmax=1, alpha=.18)
            if (n, c) in maps:
                data = np.ma.masked_where(maps[n, c] <= .001, maps[n, c])
                im = ax.imshow(data, cmap='YlOrRd', norm=Normalize(0, vmax))
            else:
                ax.text(.5, .5, 'Not recorded', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9)
            ax.set(xlim=(5, 55), ylim=(55, 5), xticks=[], yticks=[])
            if row_index == 0:
                ax.set_title(f'{n} robots')
            if col_index == 0:
                ax.set_ylabel(LABELS[c], fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.colorbar(im, ax=axes, fraction=.025, label=f'Mean active slow robots per cell (2–{horizon:g} s)')
    save_figure(fig, 'spatial', output)


def aggregate(rows):
    groups = grouped(rows)
    fields = ['deliveries_completed', 'restricted_mean_flow_s', 'prefix27_mean_flow_s',
              'active_slow_s_per_robot', 'primary_planner_failures']
    summary = []
    for (n, c), values in groups.items():
        if values:
            summary.append({'num_agents': n, 'condition': c, 'runs': len(values),
                **{k: {'mean': float(np.mean([r[k] for r in values])),
                       'min': float(min(r[k] for r in values)),
                       'max': float(max(r[k] for r in values))} for k in fields}})
    pairs = []
    by_key = {(r['workload_seed'], r['num_agents'], r['safety_condition']): r for r in rows}
    for r in rows:
        if r['safety_condition'] == 'disabled':
            continue
        off = by_key.get((r['workload_seed'], r['num_agents'], 'disabled'))
        if off:
            pairs.append({'seed': r['workload_seed'], 'num_agents': r['num_agents'],
                          'condition': r['safety_condition'],
                          **{f'{k}_on_minus_off': r[k] - off[k] for k in fields}})
    interaction = []
    for seed in sorted({r['workload_seed'] for r in rows}):
        for n in (54, 81):
            keys = [(seed, n, 'outer_overlap'), (seed, 27, 'outer_overlap'),
                    (seed, n, 'disabled'), (seed, 27, 'disabled')]
            if all(k in by_key for k in keys):
                a, b, c, d = [by_key[k]['prefix27_mean_flow_s'] for k in keys]
                interaction.append({'seed': seed, 'num_agents': n, 'extra_load_effect_off_s': c-d,
                                    'extra_load_effect_on_s': a-b, 'interaction_s': a-b-c+d})
    return summary, pairs, interaction


def tex_table(summary, output):
    lines = [r'\begin{tabular}{rlrrrr}\toprule',
             r'Robots & Safety rule & Delivered & Mean time (s) & Active slow (s/robot) & PBS failures \\ \midrule']
    for row in summary:
        values = [row[k]['mean'] for k in ['deliveries_completed', 'restricted_mean_flow_s',
                                          'active_slow_s_per_robot', 'primary_planner_failures']]
        lines.append(f"{row['num_agents']} & {LABELS[row['condition']]} & " + ' & '.join(f'{v:.1f}' for v in values) + r' \\')
    lines += [r'\bottomrule\end{tabular}']
    (output/'results_table.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, help='Study created by run_safety_capacity.py')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if Path(args.study).name != args.study or args.study in ('', '.', '..'):
        parser.error('Choose a simple study name')
    root = REPORT/'data/safety_capacity'/args.study
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete' and not args.partial:
        raise ValueError('Study has not completed')
    specs = manifest['runs']
    if len({s['run_id'] for s in specs}) != len(specs):
        raise ValueError('Duplicate run IDs in manifest')
    if any(s['num_agents'] not in (27, 54, 81) or s['safety_condition'] not in CONDITIONS for s in specs):
        raise ValueError('Expected a fleet safety study; isolated controls are checked by verify.py')
    completed = [s for s in specs if (root/s['run_id']/'route_metrics.json').exists()]
    if len(completed) != len(specs) and not args.partial:
        raise ValueError('Completed study is missing run outputs')
    rows = [analyze_run(root/s['run_id']) for s in completed]
    if not rows:
        raise ValueError('No completed runs')
    rows.sort(key=lambda r: (r['num_agents'], r['safety_condition'], r['workload_seed']))
    if len({(r['horizon_s'], r['planning_calls']) for r in rows}) != 1:
        raise ValueError('Plotting requires a common observation horizon and planning schedule')
    for spec in completed:
        row = next(r for r in rows if r['run_id'] == spec['run_id'])
        if any(row[k] != spec[k] for k in ('workload_seed', 'num_agents', 'safety_condition', 'horizon_s')):
            raise ValueError(f'Metric/spec mismatch: {spec["run_id"]}')
    output = root/'analysis'
    output.mkdir(parents=True, exist_ok=True)
    summary, pairs, interaction = aggregate(rows)
    write_json(output/'results.json', {'study': args.study, 'status': manifest['status'], 'run_count': len(rows),
               'summary': summary, 'paired_effects': pairs, 'common_task_load_effects': interaction, 'runs': rows,
               'interpretation': 'Paired effects are total safety-response effects under the fixed planner. Remaining no-response delay is not pure physical road capacity. Ranges describe only the recorded workloads, not population confidence intervals.'})
    keys = [k for k, value in rows[0].items() if not isinstance(value, (dict, list))]
    with (output/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    tex_table(summary, output)
    plots(rows, root, output)
    print('WROTE', output)


if __name__ == '__main__':
    main()
