#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('benchmark/data/route_search')
OUT = ROOT / 'visualizations'
PARTS = ['cells', 'regions9', 'regions36', 'regions144', 'junction_corridor']
NAMES = ['Cells', '9 regions', '36 regions', '144 regions', 'Junction + corridor']


def completed_studies():
    found = {}
    for study in sorted(ROOT.iterdir()):
        manifest, output = study / 'manifest.json', study / 'results.json'
        if not (manifest.is_file() and output.is_file()):
            continue
        meta = json.loads(manifest.read_text())
        if meta.get('status') != 'complete':
            continue

        for row in json.loads(output.read_text()):
            part = row.get(
                'partition',
                meta.get('policy', {}).get('spatial_partition', 'cells')
            )
            key = (
                int(row['num_agents']),
                int(row['workload_seed']),
                row['method'],
                part
            )
            if part in PARTS and key not in found:
                found[key] = (study, row)
    return found


def plot_metrics(data, seed, method):
    measures = [
        ('deliveries_completed', 'Completed deliveries', 'deliveries'),
        ('restricted_mean_flow_s', 'Restricted mean flow time (s)', 'flow_time'),
        ('recording_wall_s', 'Computer runtime (s)', 'runtime'),
        ('selected_modifier_reviews', 'Selected modified plans', 'interventions')
    ]

    x = np.arange(len(PARTS))

    for field, ylabel, name in measures:
        fig, ax = plt.subplots(figsize=(11, 5.5))

        fleets = (27, 54, 81)
        colors = ('#2563eb', '#d97706', '#0d9488')

        for i, (fleet, color) in enumerate(zip(fleets, colors)):
            vals = [
                float(data[(fleet, seed, method, p)][1][field])
                if (fleet, seed, method, p) in data else np.nan
                for p in PARTS
            ]

            if np.isfinite(vals).any():
                bars = ax.bar(
                    x + (i-1)*.25, vals, .24,
                    label=f'{fleet} robots', color=color
                )

                for bar, value in zip(bars, vals):
                    if np.isfinite(value):
                        fmt = (
                            f'{value:.2f}'
                            if field == 'restricted_mean_flow_s'
                            else f'{value:.0f}'
                        )
                        ax.annotate(
                            fmt,
                            (bar.get_x()+bar.get_width()/2, value),
                            ha='center', va='bottom', fontsize=8,
                            xytext=(0,3), textcoords='offset points'
                        )

        ax.set_xticks(x, NAMES, rotation=12, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(
            f'{ylabel} | {method} | seed {seed} | 240 s simulated'
        )
        ax.legend()
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        file = OUT / f'{name}_{method}_seed{seed}.png'
        fig.savefig(file, dpi=170)
        plt.close(fig)
        print('Saved', file)



def draw_rectangular_grid(ax, scenario, partition):
    """Draw complete region boundaries, including over obstacles."""
    from matplotlib.patches import Rectangle

    factors = {
        "regions9": 1,
        "regions36": 2,
        "regions144": 4,
    }

    factor = factors.get(partition)

    if factor is None:
        return

    for zone in scenario["zones"]:
        r0, r1 = zone["bounds"]["rows"]
        c0, c1 = zone["bounds"]["cols"]

        row_groups = np.array_split(
            np.arange(r0, r1 + 1), factor
        )
        col_groups = np.array_split(
            np.arange(c0, c1 + 1), factor
        )

        for rows in row_groups:
            for cols in col_groups:
                if len(rows) == 0 or len(cols) == 0:
                    continue

                ax.add_patch(Rectangle(
                    (cols[0] - 0.5, rows[0] - 0.5),
                    len(cols),
                    len(rows),
                    fill=False,
                    edgecolor="#38bdf8",
                    linewidth=0.8 if factor < 4 else 0.55,
                    alpha=0.9,
                    zorder=10,
                ))


def plot_cost_maps(data, fleet, seed, method, tick):
    sys.path.insert(0, str(Path.cwd() / 'benchmark/code'))
    from spatial_partitions import partition_labels

    cases = []

    for part, name in zip(PARTS, NAMES):
        key = (fleet, seed, method, part)
        if key not in data:
            continue

        study = data[key][0]
        run = study / f'w{seed}_fleet{fleet}_{method}'
        response_file = run / 'bridge' / f'response_{tick}.json'

        if not response_file.is_file():
            print('No saved cost map:', response_file)
            continue

        layout = json.loads(
            (run/'map.json').read_text()
        )['layout']
        scenario = json.loads(
            (run/'scenario.json').read_text()
        )
        response = json.loads(response_file.read_text())

        candidate = next(
            (c for c in response['route_candidates']
             if c['name'] == 'strong'),
            None
        )
        if candidate is None:
            continue

        road = np.array(
            [[ch not in '@T' for ch in row] for row in layout],
            bool
        )
        cost = np.array(
            candidate['cell_penalties'], float
        ).reshape(road.shape)

        labels = (
            None if part == 'cells'
            else partition_labels(road, scenario, part)
        )

        cases.append((name, road, cost, labels, part, scenario))

    if not cases:
        print(f'No completed {fleet}-robot cost maps available yet.')
        return

    if any(
        not np.array_equal(c[1], cases[0][1])
        for c in cases
    ):
        raise ValueError('Warehouse maps differ!')

    vmax = max(float(c[2].max()) for c in cases)

    fig, axes = plt.subplots(
        2, len(cases),
        figsize=(len(cases)*3.9, 8),
        squeeze=False
    )

    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#303743')

    for col, (name, road, cost, labels, part, scenario) in enumerate(cases):
        top, bottom = axes[:, col]

        top.set_facecolor('#303743')

        if labels is None:
            top.imshow(
                road, cmap='gray', vmin=0, vmax=1,
                interpolation='nearest'
            )
        else:
            top.imshow(
                np.ma.masked_where(~road, labels),
                cmap='tab20', interpolation='nearest'
            )

        draw_rectangular_grid(top, scenario, part)
        top.set_title(name + '\nRegions')

        im = bottom.imshow(
            np.ma.masked_where(~road, cost),
            cmap=cmap, vmin=0,
            vmax=max(vmax, 1e-9),
            interpolation='nearest'
        )
        draw_rectangular_grid(bottom, scenario, part)
        bottom.set_title(f'Max penalty: {cost.max():.2f}')

        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(
        im, ax=axes[1,:].tolist(), shrink=.7,
        label='Additional PBS move cost per cell'
    )

    fig.suptitle(
        f'{fleet} robots | {method} | tick {tick} '
        f'({tick/10:g} s) | strong candidate'
    )

    file = OUT / f'cost_maps_{fleet}robots_{method}_tick{tick}.png'
    fig.savefig(file, dpi=170, bbox_inches='tight')
    plt.close(fig)

    print('Saved', file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fleet', type=int, choices=(27,54,81), default=54)
    parser.add_argument('--seed', type=int, default=8401)
    parser.add_argument(
        '--method', choices=('nominal','rollout'), default='rollout'
    )
    parser.add_argument('--tick', type=int, default=120)
    args = parser.parse_args()

    if not ROOT.is_dir():
        raise SystemExit('Run from ~/research/jesse-warehouse')

    OUT.mkdir(exist_ok=True)
    data = completed_studies()

    print('Completed runs loaded:', len(data))

    plot_metrics(data, args.seed, args.method)
    plot_cost_maps(data, args.fleet, args.seed, args.method, args.tick)

    csv_path = OUT / 'partition_metrics.csv'

    with csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'study','fleet','seed','method','partition',
            'deliveries','restricted_flow_s','wall_s',
            'modifiers','changed_routes'
        ])

        for (fleet,seed,method,part),(study,row) in sorted(data.items()):
            writer.writerow([
                study.name, fleet, seed, method, part,
                row['deliveries_completed'],
                row['restricted_mean_flow_s'],
                row['recording_wall_s'],
                row['selected_modifier_reviews'],
                row.get('changed_route_geometry_reviews','')
            ])

    print('Saved', csv_path)


if __name__ == '__main__':
    main()
