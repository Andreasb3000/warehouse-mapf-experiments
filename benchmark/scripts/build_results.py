#!/usr/bin/env python3
"""Plot shared predictor evaluation without historical report assets."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'zero': ('Zero', '#a3a3a3'),
    'persistence': ('Persistence', '#64748b'),
    'kinematic': ('Plan projection', '#d97706'),
    'joint': ('Joint rollout', '#2563eb'),
    'local_ridge': ('Local ridge', '#15803d'),
    'convlstm': ('ConvLSTM', '#7c3aed'),
    'residual_convlstm': ('Residual ConvLSTM', '#be185d'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=ROOT/'results/evaluation.json')
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    parser.add_argument('--region', choices=('roads', 'roi', 'positive', 'positive_roi'), default='roi')
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHODS if m in data['methods']]
    if not methods:
        raise ValueError('Evaluation contains no supported predictors')
    plt.rcParams.update({'font.family': 'sans-serif', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    windows = data['windows_s']
    figure, axes = plt.subplots(2, len(windows), figsize=(13, 7), squeeze=False)
    plotted = []
    for row, channel in enumerate(('occupancy', 'slow_occupancy')):
        for col, window in enumerate(windows):
            values = [data['methods'][m][col]['channels'][channel]['regions'][args.region]['rmse']
                      for m in methods]
            axis = axes[row, col]
            axis.bar(range(len(methods)), [np.nan if v is None else v for v in values],
                     color=[METHODS[m][1] for m in methods])
            axis.set_xticks(range(len(methods)), [METHODS[m][0] for m in methods],
                            rotation=45, ha='right', fontsize=8)
            axis.set_title(f'{window[0]:g}–{window[1]:g} s')
            axis.set_ylabel(('Occupancy' if channel == 'occupancy' else 'Slow occupancy') + ' RMSE')
            axis.set_ylim(bottom=0)
            plotted.append({'channel': channel, 'window_s': window,
                            'rmse': dict(zip(methods, values))})
    figure.suptitle(f"Held-out forecasts · {args.region} · {data['test_origins']} origins")
    figure.tight_layout()
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output/f'prediction_accuracy.{suffix}', dpi=180)
    plt.close(figure)
    (args.output/'prediction_accuracy.json').write_text(json.dumps({
        'evaluation_sha256': hashlib.sha256(args.evaluation.read_bytes()).hexdigest(),
        'region': args.region, 'units': 'robots per cell', 'values': plotted,
        'note': 'Each window/channel is scored independently; lower RMSE is better.'}, indent=2)+'\n')
    print(args.output/'prediction_accuracy.pdf')


if __name__ == '__main__':
    main()
