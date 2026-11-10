#!/usr/bin/env python3
"""Build admission-control figures and LaTeX only from a complete frozen study.

Default: data/entry_gates/central_gate_v3. All declared matched recordings
must finish before any production output is written. --self-test uses a
temporary synthetic fixture solely to test rendering and integrity checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from archive_entry_gate_study import archive_study, verify_source_snapshot

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'none': ('No gate', '#64748b'),
           'reactive': ('Current-state gate', '#d97706'),
           'forecast': ('Forecast gate', '#2563eb')}
COUNTS = (27, 54, 81)
SEEDS = (8301, 8302, 8303)
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load_study(directory, workspace=ROOT.parent):
    """Reject partial, changed, duplicate, or unmatched evidence before writing."""
    directory, workspace = Path(directory), Path(workspace)
    manifest = read(directory / 'manifest.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Study manifest must be complete before building report artifacts')
    if not manifest.get('source_sha256'):
        raise ValueError('Missing frozen implementation hashes')
    if (directory / 'source_snapshot').exists():
        # Historical evidence stays reproducible after the checkout develops.
        # The verifier also binds archived compiled-server sources and native
        # build metadata to this exact image and immutable study protocol.
        verify_source_snapshot(directory, manifest)
    else:
        for relative, expected in manifest['source_sha256'].items():
            if digest(workspace / relative) != expected:
                raise ValueError(f'Frozen source changed: {relative}')
    policy = manifest['policy']
    expected_policy = {'version': 'central-entry-gate-v3',
                       'protect_uncommitted_clearance': False,
                       'center': [30., 30.], 'half_width': 3., 'interval_ticks': 20,
                       'forecast_horizon_s': 5., 'slow_pressure_threshold_robots': .5,
                       'max_continuous_hold_ticks': 60, 'forced_release_ticks': 20}
    for key, value in expected_policy.items():
        if policy.get(key) != value:
            raise ValueError(f'Protocol differs from the reported central gate: {key}')
    runs = read(directory / 'results.json')
    specs = manifest['runs']
    counts = sorted({s['num_agents'] for s in specs})
    seeds = sorted({s['workload_seed'] for s in specs})
    horizons = {s['horizon_s'] for s in specs}
    if len(horizons) != 1:
        raise ValueError('All matched conditions must use one horizon')
    horizon = next(iter(horizons))
    expected = {(n, seed, method) for n in counts for seed in seeds for method in METHODS}
    key = lambda r: (r['num_agents'], r['workload_seed'], r['method'])
    if len(runs) != len(expected) or {key(r) for r in runs} != expected:
        raise ValueError('Require every method for each declared fleet/workload pair')
    if len(specs) != len(runs) or {s['run_id'] for s in specs} != {r['run_id'] for r in runs}:
        raise ValueError('Results do not match manifest run IDs')
    specs_by_id = {s['run_id']: s for s in specs}
    paired_inputs = defaultdict(lambda: defaultdict(set))
    map_hashes = set()
    for run in runs:
        spec = specs_by_id[run['run_id']]
        if any(run[k] != spec[k] for k in ('run_id', 'num_agents', 'workload_seed',
                                           'method', 'trigger', 'horizon_s')):
            raise ValueError(f'Run disagrees with frozen specification: {run["run_id"]}')
        if run['trigger'] != 'outer_overlap' or run['horizon_s'] != horizon:
            raise ValueError('This comparison requires Safety rule 2 and a shared horizon')
        if spec['image_id'] != manifest['image_id']:
            raise ValueError('Native image differs from frozen image')
        folder = directory / run['run_id']
        if read(folder / 'run_spec.json') != spec or read(folder / 'gate_metrics.json') != run:
            raise ValueError(f'Run metadata or metrics changed: {run["run_id"]}')
        hashes = run.get('source_sha256', {})
        required = {'simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'run_spec.json'}
        if not required <= set(hashes):
            raise ValueError(f'Missing native evidence hashes: {run["run_id"]}')
        for name, expected_hash in hashes.items():
            if digest(folder / name) != expected_hash:
                raise ValueError(f'Native evidence changed: {run["run_id"]}/{name}')
        for name in ('scenario.json', 'tasks.json'):
            paired_inputs[key(run)[:2]][name].add(hashes[name])
        map_hashes.add(digest(folder / 'map.json'))
        outcomes = run['task_outcomes']
        times = [t['delivery_completed_s'] for t in outcomes if t['delivery_completed_s'] is not None]
        if len(outcomes) != run['num_agents'] or len(times) != run['deliveries_completed']:
            raise ValueError('Delivery curve and all-task outcomes disagree')
        if any(not 0 <= t <= run['horizon_s'] for t in times):
            raise ValueError('Delivery outside the stated native horizon')
        if run['planning_calls'] != horizon / 2:
            raise ValueError('Expected one planning call every two seconds')
    if len(map_hashes) != 1 or any(len(hashes) != 1 for group in paired_inputs.values() for hashes in group.values()):
        raise ValueError('Methods must share the same floor and paired workload inputs')
    return manifest, sorted(runs, key=key)


def save(fig, directory, name):
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def groups(runs):
    result = defaultdict(list)
    for run in runs:
        result[run['num_agents'], run['method']].append(run)
    return {key: sorted(value, key=lambda r: r['workload_seed']) for key, value in result.items()}


def make_figures(runs, directory):
    grouped = groups(runs)
    counts = sorted({r['num_agents'] for r in runs})
    horizon = runs[0]['horizon_s']
    grid = np.arange(round(horizon * 10) + 1) / 10
    fig, axes = plt.subplots(1, len(counts), figsize=(max(4, 3.4 * len(counts)), 3.15),
                             constrained_layout=True, squeeze=False)
    for ax, count in zip(axes.ravel(), counts):
        for method, (label, color) in METHODS.items():
            curves = []
            for run in grouped[count, method]:
                times = sorted(t['delivery_completed_s'] for t in run['task_outcomes']
                               if t['delivery_completed_s'] is not None)
                curve = np.searchsorted(times, grid, side='right')
                curves.append(curve)
                ax.step(grid, curve, where='post', color=color, alpha=.20, lw=.7)
            ax.step(grid, np.mean(curves, axis=0), where='post', color=color, lw=2.0, label=label)
        ax.set(title=f'{count} robots', xlabel='Simulated time (s)', ylabel='Completed deliveries',
               xlim=(0, horizon), ylim=(0, count + 1), xticks=np.linspace(0, horizon, 5))
        ax.grid(alpha=.16)
    handles = [Line2D([], [], color=color, lw=2, label=label) for label, color in METHODS.values()]
    handles.append(Line2D([], [], color='#64748b', lw=.7, alpha=.5, label='Individual workload'))
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=8)
    save(fig, directory, 'entry_gate_outcomes')
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    panels = [('restricted_mean_flow_s', 1., 'Task outcomes in simulator time', 'Mean restricted flow time (s)'),
              ('host_decision_p95_s', 1000., 'Runtime reported separately', 'Host decision p95 (ms)')]
    for ax, (field, scale, title, ylabel) in zip(axes, panels):
        for method, (label, color) in METHODS.items():
            values = scale * np.array([[r[field] for r in grouped[count, method]] for count in counts])
            for seed_values in values.T:
                ax.plot(counts, seed_values, color=color, alpha=.18, lw=.7)
            ax.plot(counts, values.mean(axis=1), color=color, marker='o', ms=4, lw=1.8, label=label)
        ax.set(title=title, xlabel='Number of robots', ylabel=ylabel,
               xticks=counts, xlim=(min(counts)-3, max(counts)+3), ylim=(0, None))
        ax.grid(alpha=.16)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, directory, 'entry_gate_flow')


def paired_difference(grouped, count, reference, field):
    baseline = {r['workload_seed']: r[field] for r in grouped[count, reference]}
    return float(np.mean([r[field] - baseline[r['workload_seed']] for r in grouped[count, 'forecast']]))


def write_section(manifest, runs, destination):
    grouped = groups(runs)
    counts = sorted({r['num_agents'] for r in runs})
    seeds = sorted({r['workload_seed'] for r in runs})
    horizon = runs[0]['horizon_s']
    rows = []
    for count in counts:
        for method, (label, _) in METHODS.items():
            entries = grouped[count, method]
            mean = lambda name: float(np.mean([r[name] for r in entries]))
            failed = 100 * sum(r['primary_planner_failures'] for r in entries) / sum(r['planning_calls'] for r in entries)
            rows.append(f'{count} & {label} & {mean("deliveries_completed"):.1f} & '
                        f'{mean("restricted_mean_flow_s"):.1f} & '
                        f'{failed:.1f} & {mean("empty_queue_gate_wait_robot_s"):.1f} ' + r"\\")
    section = r"""\section{A simple junction gate}
The native planner chooses routes. An admission rule controls when robots may enter the central junction. The comparison includes no gate, a gate using current slow traffic, and the same gate using a five-second joint-rollout forecast. Rollout includes safety responses but does not anticipate later replans or gate interventions.

Every two seconds after a twelve-second warmup, the gate prioritizes robots already inside the central six-meter square and committed entrants when mean slow traffic reaches half a robot. Otherwise, the longest-waiting eligible entrant is released. A requested hold lasts at most six consecutive seconds before a two-second release. This is an experimental traffic controller, not a safety or progress guarantee.

Computation is measured separately and is not added to task duration. Methods share the map, initial states and tasks within each fleet/workload pair. All runs use Safety rule 2. The primary PBS planner has a one-CPU-second budget with PIBT fallback; budget-sensitive results may vary across machines.

"""
    section += (f'The completed study contains {len(runs)} runs: '
                f'{len(counts)} fleet sizes, {len(seeds)} workloads, and three methods, '
                f'each lasting {horizon:g} simulated seconds.\n\n')
    section += r"""\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/entry_gate_outcomes.pdf}
\caption{Completed deliveries. Thick curves average the declared workloads; thin curves show individual workloads.}
\end{figure}

\begin{table}[H]\centering\small
\begin{tabular}{@{}rlrrrr@{}}\toprule
$N$ & Method & Delivered & Flow (s) & PBS fail (\%) & Gate wait \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule\end{tabular}
\caption{Workload means. Flow includes every task, capping unfinished completion times at the run horizon. PBS failures are unsuccessful primary calls. Gate wait measures empty-queue waiting induced by native dispatch denials, in robot-seconds.}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=.94\linewidth]{figures/entry_gate_flow.pdf}
\caption{Left: mean capped task completion time, lower is better. Right: mean per-run 95th percentile of host decision time. Native planning and response handoff are excluded. No gate has no predictor query.}
\end{figure}
"""
    destination.write_text(section.strip() + '\n')


def preserve_render_revision(study, output_root):
    """Keep this user-requested reporting change outside frozen execution files."""
    study, output_root = Path(study), Path(output_root)
    builder = Path(__file__)
    artifact_paths = [output_root / 'results/entry_gate_results.tex'] + [
        output_root / f'figures/{name}.{extension}'
        for name in ('entry_gate_outcomes', 'entry_gate_flow') for extension in ('pdf', 'png')]
    output_hashes = {str(path.relative_to(output_root)): digest(path) for path in artifact_paths}
    builder_hash = digest(builder)
    artifact_hash = hashlib.sha256(json.dumps(output_hashes, sort_keys=True).encode()).hexdigest()
    revision = output_root / 'results/entry_gate_report_revisions' / f'native_time_{builder_hash[:12]}_{artifact_hash[:12]}'
    archive = study / 'source_snapshot'
    original_builder = archive / 'benchmark/scripts/build_entry_gate_results.py'
    archived_helper = archive / 'benchmark/scripts/archive_entry_gate_study.py'
    helper = archived_helper if archived_helper.exists() else builder.with_name('archive_entry_gate_study.py')
    record = {
        'schema_version': 1,
        'scope': 'Reporting-only revision requested by the user: compare native simulator outcomes; show runtime separately without adding it to task duration.',
        'execution_changed': False,
        'raw_metrics_changed': False,
        'study_manifest_sha256': digest(study / 'manifest.json'),
        'study_results_sha256': digest(study / 'results.json'),
        'frozen_source_archive_index_sha256': digest(archive / 'archive_manifest.json') if archive.exists() else None,
        'original_archived_builder_sha256': digest(original_builder) if original_builder.exists() else None,
        'builder_sha256': builder_hash,
        'archive_verifier_sha256': digest(helper),
        'runtime_aggregation': 'Mean of per-run host decision p95 values, in milliseconds; no gate has no predictor query and is shown as zero.',
        'output_sha256': output_hashes,
    }
    metadata = revision / 'revision.json'
    if metadata.exists():
        if read(metadata) != record:
            raise ValueError('Existing report revision provenance differs')
        return revision
    revision.mkdir(parents=True, exist_ok=False)
    shutil.copy2(builder, revision / builder.name)
    shutil.copy2(helper, revision / 'archive_entry_gate_study.py')
    for path in artifact_paths:
        target = revision / path.relative_to(output_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    metadata.write_text(json.dumps(record, indent=2) + '\n')
    return revision


def build(study, output_root=ROOT, workspace=ROOT.parent):
    manifest, runs = load_study(study, workspace)
    output_root = Path(output_root)
    figures, results = output_root / 'figures', output_root / 'results'
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    make_figures(runs, figures)
    write_section(manifest, runs, results / 'entry_gate_results.tex')
    preserve_render_revision(study, output_root)
    return results / 'entry_gate_results.tex'


def self_test():
    """Synthetic values stay in a temporary directory and are never reported."""
    with tempfile.TemporaryDirectory(prefix='entry-gate-builder-test-') as name:
        root = Path(name)
        study = root / 'study'; study.mkdir()
        (root / 'policy.py').write_text('# frozen fixture only\n')
        native_source = root / 'simulator/server/src/fixture.cpp'
        native_source.parent.mkdir(parents=True)
        native_source.write_text('// frozen native fixture only\n')
        policy = dict(version='central-entry-gate-v3', protect_uncommitted_clearance=False,
                      center=[30., 30.], half_width=3., interval_ticks=20,
                      forecast_horizon_s=5., slow_pressure_threshold_robots=.5,
                      max_continuous_hold_ticks=60, forced_release_ticks=20)
        specs, runs = [], []
        for count in COUNTS:
            for seed in SEEDS:
                for method in METHODS:
                    spec = dict(run_id=f'{count}_{seed}_{method}', num_agents=count,
                                workload_seed=seed, method=method, trigger='outer_overlap',
                                horizon_s=240, image_id='fixture-only')
                    folder = study / spec['run_id']; folder.mkdir()
                    for filename in ('simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'map.json'):
                        (folder / filename).write_text('{}\n')
                    (folder / 'run_spec.json').write_text(json.dumps(spec))
                    row = {**spec, 'deliveries_completed': 2,
                           'task_outcomes': [dict(delivery_completed_s=20. + i if i < 2 else None) for i in range(count)],
                           'restricted_mean_flow_s': 200., 'charged_restricted_mean_flow_s': 201.,
                           'primary_planner_failures': 3, 'planning_calls': 120,
                           'empty_queue_gate_wait_robot_s': 1., 'host_decision_p50_s': .01,
                           'host_decision_p95_s': .02,
                           'source_sha256': {filename: digest(folder / filename) for filename in
                               ('simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'run_spec.json')}}
                    (folder / 'gate_metrics.json').write_text(json.dumps(row))
                    specs.append(spec); runs.append(row)
        manifest = dict(status='complete', policy=policy, runs=specs, image_id='fixture-only',
                        source_sha256={'policy.py': digest(root / 'policy.py')},
                        compiled_server_sha256={'src/fixture.cpp': digest(native_source)})
        (study / 'manifest.json').write_text(json.dumps(manifest))
        (study / 'results.json').write_text(json.dumps(runs))
        tex = build(study, root / 'output', root)
        assert '\\section{' in tex.read_text()
        assert 'Charged' not in tex.read_text() and 'charged' not in tex.read_text()
        assert 'is not added to task duration' in tex.read_text()
        assert len(list((root / 'output/figures').glob('*'))) == 4
        if shutil.which('pdflatex'):
            from latex_support import PREAMBLE
            preamble = PREAMBLE
            document = preamble + r'''\begin{document}
\begin{equation} e_i(t)\in\{0,1\}.\label{eq:fields}\end{equation}
\input{results/entry_gate_results.tex}
\begin{thebibliography}{9}
\bibitem{virtual-lights} Synthetic compilation reference only.
\bibitem{reservations} Synthetic compilation reference only.
\end{thebibliography}\end{document}
'''
            (root / 'output/check.tex').write_text(document)
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'check.tex'],
                                    cwd=root / 'output', capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stdout[-4000:])
            if 'Overfull \\hbox' in result.stdout:
                raise AssertionError('Generated LaTeX exceeds the report text width')
        manifest['status'] = 'running'
        (study / 'manifest.json').write_text(json.dumps(manifest))
        try:
            load_study(study, root)
        except ValueError:
            pass
        else:
            raise AssertionError('An incomplete manifest was accepted')
        manifest['status'] = 'complete'
        (study / 'manifest.json').write_text(json.dumps(manifest))
        (root / 'policy.py').write_text('# changed after freeze\n')
        try:
            load_study(study, root)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed frozen source was accepted')
        # Archive while running, then allow the live manifest to finish. Source
        # changes in the workspace must no longer prevent historical rendering.
        (root / 'policy.py').write_text('# frozen fixture only\n')
        native_verification = root / 'native-verification.json'
        native_verification.write_text(json.dumps(dict(status='verified', gate_image_id='fixture-only',
                                        gate_sources=manifest['compiled_server_sha256'])))
        smoke_audit = root / 'smoke-audit.json'
        smoke_audit.write_text(json.dumps({'image_version': 'v3', 'fixture_only': True}))
        manifest['status'] = 'running'
        (study / 'manifest.json').write_text(json.dumps(manifest))
        snapshot = archive_study(study, workspace=root, native_verification=native_verification,
                                 smoke_audit=smoke_audit, include_report_tools=False)
        manifest['status'] = 'complete'
        (study / 'manifest.json').write_text(json.dumps(manifest))
        (root / 'policy.py').write_text('# later workspace development\n')
        native_source.write_text('// later workspace development\n')
        load_study(study, root)
        # Idempotent verification must likewise use the preserved evidence.
        assert archive_study(study, workspace=root) == snapshot
        archived_native = snapshot / 'simulator/server/src/fixture.cpp'
        archived_native.write_text('// corrupted archived native source\n')
        try:
            load_study(study, root)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed archived compiled-server source was accepted')
        archived_native.write_text('// frozen native fixture only\n')
        (snapshot / 'policy.py').write_text('# corrupted archived Python source\n')
        try:
            load_study(study, root)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed archived Python source was accepted')
    print('Entry-gate report builder: temporary rendering, available LaTeX compilation, and integrity checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'data/entry_gates/central_gate_v3')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(build(args.study, args.output_root))


if __name__ == '__main__':
    main()
