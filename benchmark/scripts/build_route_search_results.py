#!/usr/bin/env python3
"""Render completed native route-search comparisons; never fill pending results.

Each study is exploratory unless listed in --evaluation-studies. Source
snapshots, native evidence, matched inputs, candidate choices, and reported
outcomes are verified before producing any report artifact. Native simulation
time excludes computation; host and candidate PBS costs are reported separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from route_preview_figure import make_preview_figure

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'none': ('Zero costs', '#64748b'), 'nominal': ('Nominal ETA', '#d97706'),
           'rollout': ('Joint-rollout ETA', '#2563eb')}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def directory(path):
    path = Path(path).resolve()
    return path.parent if path.name == 'manifest.json' else path


def tex(value):
    replacements = {'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#', '$': r'\$',
                    '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def geometry(candidate):
    """Remove timestamps and repeated WAIT samples, retaining task transitions."""
    plan = candidate['plan']
    paths = plan['plan'] if isinstance(plan, dict) else plan
    result = []
    for path in paths:
        condensed = []
        for sample in path:
            point = (sample[0], sample[1], sample[3])
            if not condensed or condensed[-1] != point:
                condensed.append(point)
        result.append(condensed)
    return result


def validate_build_proof(manifest, source_root):
    """Use the manifest's proof location, independent of versioned filenames."""
    name = manifest.get('build_proof')
    if not name:
        if manifest.get('correction'):
            raise ValueError('A corrected study must identify its frozen native build proof')
        return
    if name not in manifest['source_sha256']:
        raise ValueError('Native build proof is absent from frozen sources')
    proof = read(Path(source_root) / name)
    if (proof['image_id'] != manifest['image_id']
            or proof['compiled_source_sha256'] != manifest['compiled_source_sha256']):
        raise ValueError('Build proof, image, or compiled sources differ')
    if 'compiled_native_source_sha256' in proof:
        # Fresh export rebuilds the physical client; it does not compare its
        # binaries against an old image. Every client source must be frozen.
        native = proof['compiled_native_source_sha256']
        clients = {key: value for key, value in native.items() if key.startswith('client/')}
        if not clients or any(proof['compiled_source_sha256'].get(k) != v for k, v in native.items()):
            raise ValueError('Fresh proof omits or changes compiled physical-client sources')
        required = {'safety_geometry', 'finite_tasks', 'pibt_terminal_parking',
                    'pbs_terminal_parking', 'pbs_path_export_service_markers'}
        if not required <= set(proof.get('native_checks', [])):
            raise ValueError('Fresh build proof omits required native regression checks')
        artifacts = proof.get('build_artifacts_sha256', {})
        if not artifacts:
            raise ValueError('Fresh build proof omits build artifact hashes')
        for relative, expected in artifacts.items():
            path = Path(name).parent / relative
            if manifest['source_sha256'].get(str(path)) != expected or digest(Path(source_root) / path) != expected:
                raise ValueError(f'Fresh build artifact differs: {path}')
    elif proof.get('physical_client_binaries_identical') is not True:
        raise ValueError('Legacy build proof omits physical-client continuity')
    if manifest.get('correction') and 'pbs_path_export_service_markers' not in proof.get('native_checks', []):
        raise ValueError('Corrected study lacks the native service-marker regression check')


def audit_queries(folder, row, policy):
    queries = [json.loads(line) for line in (folder / 'host_queries.jsonl').read_text().splitlines()]
    expected = list(range(0, round(row['horizon_s'] * 10), round(policy['refresh_s'] * 10)))
    if [q['clock'] for q in queries] != expected:
        raise ValueError('Missing, duplicated, or unordered route-search reviews')
    warmup = round(policy['warmup_s'] * 10)
    reviews = [q for q in queries if q['clock'] >= warmup]
    totals = dict(review_opportunities=len(reviews), selected_modifier_reviews=0,
                  changed_route_geometry_reviews=0, candidate_planning_calls=0,
                  candidate_planning_failures=0, skipped_candidates=0,
                  invalid_eta_candidates=0, eta_scored_candidates=0,
                  zero_target_invalid_reviews=0, baseline_failed_reviews=0,
                  active_search_reviews=0, identical_cost_menu_reviews=0,
                  modifier_same_geometry_reviews=0, alternative_comparison_reviews=0,
                  baseline_eta_abstention_reviews=0)
    host_times, pbs_batch_times, pbs_call_times = [], [], []
    reasons = defaultdict(int)
    for query in queries:
        clock = query['clock']
        batch = read(folder / 'plan_search' / f'request_{clock}.json')
        response = read(folder / 'plan_search' / f'response_{clock}.json')
        proposal = read(folder / 'bridge' / f'response_{clock}.json')
        candidates = batch['candidates']
        scores = query['selection']['candidates']
        if (batch['clock'] != clock or proposal['clock'] != clock
                or response != {'clock': clock, 'selected_index': query['selected_index']}
                or batch['boundary'] != 'post_commit_before_plan_selection'
                or not 1 <= len(candidates) <= 3 or len(scores) != len(candidates)
                or len(proposal['route_candidates']) != len(candidates)):
            raise ValueError('Candidate handshake/selection evidence disagrees')
        if any(proposal['cell_penalties']) or any(proposal['route_candidates'][0]['cell_penalties']):
            raise ValueError('Candidate zero must have zero route costs')
        if query['selected_index'] and (clock < warmup or row['method'] == 'none'):
            raise ValueError('Modifier selected in warmup or zero-cost control')
        selected = candidates[query['selected_index']]
        if query['selected_name'] != selected['name'] or query['selection']['selected_index'] != query['selected_index']:
            raise ValueError('Selected candidate metadata differs')
        if query['selected_index'] and not selected['success']:
            raise ValueError('Failed alternative selected')
        failed_zero = not candidates[0]['success']
        if query['baseline_failed'] != failed_zero:
            raise ValueError('Baseline failure label differs')
        if failed_zero and (query['selected_index'] or any(c['evaluated'] for c in candidates[1:])):
            raise ValueError('Failed baseline must retain native fallback and skip alternatives')
        margin = query['selection']['minimum_mean_eta_gain_s']
        if not np.isclose(margin, row['minimum_mean_eta_gain_s']):
            raise ValueError('Selection margin differs from frozen specification')
        eligible = [s for s in scores if s['eligible']]
        if eligible:
            if not scores[0]['eligible']:
                raise ValueError('Cannot compare against an invalid zero-cost ETA')
            best = min(eligible, key=lambda s: (s['mean_eta_s'], s['index']))
            gain = scores[0]['mean_eta_s'] - best['mean_eta_s']
            chosen = best['index'] if gain > margin else 0
            if chosen != query['selected_index']:
                raise ValueError('Selection violates the strict common-ETA margin rule')
        elif query['selected_index']:
            raise ValueError('Selected route lacks a valid ETA comparison')
        batch_time = 0.
        for index, (candidate, score, costs) in enumerate(zip(candidates, scores, proposal['route_candidates'])):
            values = np.asarray(costs['cell_penalties'])
            if (candidate['index'] != index or score['index'] != index or candidate['name'] != costs['name']
                    or score['name'] != candidate['name']
                    or candidate['evaluated'] != score['evaluated'] or candidate['success'] != score['success']
                    or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1000)
                    or np.count_nonzero(values) != candidate['nonzero_penalty_cells']
                    or not np.isclose(values.max(), candidate['max_cell_penalty'])
                    or canonical_digest(candidate['plan']) != score['native_plan_sha256']):
                raise ValueError('Candidate costs, status, or exact plan differs from host scoring')
            wall = float(candidate['planner_wall_s'])
            if not np.isfinite(wall) or wall < 0:
                raise ValueError('Invalid candidate PBS wall time')
            if candidate['evaluated']:
                totals['candidate_planning_calls'] += 1
                totals['candidate_planning_failures'] += not candidate['success']
                batch_time += wall; pbs_call_times.append(wall)
            else:
                totals['skipped_candidates'] += 1
            if score['eligible']:
                totals['eta_scored_candidates'] += 1
            elif score['rejection_reason'] not in (None, 'planner_failed_or_skipped'):
                totals['invalid_eta_candidates'] += 1
        if clock >= warmup:
            chosen = query['selected_index'] != 0
            changed = chosen and geometry(selected) != geometry(candidates[0])
            totals['selected_modifier_reviews'] += chosen
            totals['changed_route_geometry_reviews'] += changed
            totals['modifier_same_geometry_reviews'] += chosen and not changed
            totals['baseline_failed_reviews'] += failed_zero
            totals['zero_target_invalid_reviews'] += query['selection']['selection_reason'] == 'zero_target_invalid'
            totals['identical_cost_menu_reviews'] += len(candidates) > 1 and query['proposal']['distinct_penalty_fields'] == 1
            pbs_batch_times.append(batch_time)
            reasons[query['selection']['selection_reason']] += 1
        if query['reason'] == 'decision':
            totals['active_search_reviews'] += 1
            totals['alternative_comparison_reviews'] += len(eligible) > 1
            totals['baseline_eta_abstention_reviews'] += query['selection']['selection_reason'] in (
                'zero_target_invalid', 'zero_score_invalid')
            host_times.append(query['host_query_s'])
    for key in ('selected_modifier_reviews', 'changed_route_geometry_reviews', 'candidate_planning_calls',
                'candidate_planning_failures', 'baseline_failed_reviews', 'active_search_reviews'):
        if totals[key] != row[key]:
            raise ValueError(f'Run summary differs from candidate evidence: {key}')
    if len(queries) != row['planning_calls'] or len(reviews) != row['scheduled_reviews_after_warmup']:
        raise ValueError('Scheduled review denominator differs')
    totals.update(modifier_percent=100 * totals['selected_modifier_reviews'] / len(reviews) if reviews else 0.,
                  geometry_change_percent=100 * totals['changed_route_geometry_reviews'] / len(reviews) if reviews else 0.,
                  host_query_p95_s=float(np.quantile(host_times, .95)) if host_times else 0.,
                  pbs_batch_p95_s=float(np.quantile(pbs_batch_times, .95)) if pbs_batch_times else 0.,
                  candidate_pbs_total_s=float(sum(pbs_call_times)),
                  candidate_pbs_call_p95_s=float(np.quantile(pbs_call_times, .95)) if pbs_call_times else 0.,
                  selection_reasons=dict(reasons))
    return totals


def load_study(path, role='pilot', workspace=ROOT.parent):
    path = directory(path)
    manifest = read(path / 'manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('accounting') != 'native_time_only':
        raise ValueError(f'Require complete native-time route study: {path}')
    if manifest['policy'].get('oracle') is not False or manifest['policy'].get('future_replanning_modeled') is not False:
        raise ValueError('Renderer expects causal, fixed-plan candidate forecasts')
    for name, expected in {'cost_forecast_horizon_s': 20., 'eta_horizon_s': 30.,
                           'minimum_mean_eta_gain_s': .5, 'cost_weights': [0., 2., 8.]}.items():
        if manifest['policy'].get(name) != expected:
            raise ValueError(f'Renderer method description requires the declared prototype setting: {name}')
    source_root = path / 'source_snapshot' if (path / 'source_snapshot').exists() else Path(workspace)
    if not manifest.get('source_sha256') or not manifest.get('compiled_source_sha256'):
        raise ValueError('Missing frozen Python/native source provenance')
    for name, expected in manifest['source_sha256'].items():
        if digest(source_root / name) != expected:
            raise ValueError(f'Frozen execution source differs: {name}')
    for name, expected in manifest['compiled_source_sha256'].items():
        if digest(source_root / 'simulator' / name) != expected:
            raise ValueError(f'Compiled source differs: {name}')
    validate_build_proof(manifest, source_root)
    rows, specs = read(path / 'results.json'), manifest['runs']
    by_id = {spec['run_id']: spec for spec in specs}
    if (len(by_id) != len(specs) or len(rows) != len(specs)
            or len({r['run_id'] for r in rows}) != len(rows)
            or {r['run_id'] for r in rows} != set(by_id)):
        raise ValueError('Completed results differ from the frozen run list')
    methods, horizons = {r['method'] for r in rows}, {r['horizon_s'] for r in rows}
    if 'none' not in methods or not methods <= METHODS.keys() or len(methods) < 2 or len(horizons) != 1:
        raise ValueError('Need a matched zero-cost control and comparison at one common horizon')
    paired_hashes, seen, enriched, map_hashes = defaultdict(lambda: defaultdict(set)), set(), [], set()
    for row in rows:
        folder, spec = path / row['run_id'], by_id[row['run_id']]
        key = row['num_agents'], row['workload_seed'], row['method']
        if key in seen or any(row[k] != value for k, value in spec.items()):
            raise ValueError('Duplicate run or changed frozen specification')
        seen.add(key)
        if row['image_id'] != manifest['image_id'] or row['trigger'] != 'outer_overlap':
            raise ValueError('Run image or safety rule differs')
        if read(folder / 'run_spec.json') != spec or read(folder / 'route_metrics.json') != row:
            raise ValueError('Saved run metadata/metrics changed')
        hashes = row['source_sha256']
        required = {'simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'map.json', 'run_spec.json'}
        if not required <= hashes.keys():
            raise ValueError('Missing raw evidence hashes')
        for name, expected in {**hashes, **row.get('bridge_sha256', {})}.items():
            if digest(folder / name) != expected:
                raise ValueError(f'Run evidence changed: {row["run_id"]}/{name}')
        expected_bridge = {str(p.relative_to(folder)) for sub in ('bridge', 'plan_search')
                           for p in (folder / sub).glob('*.json')}
        if not expected_bridge <= (hashes.keys() | row.get('bridge_sha256', {}).keys()):
            raise ValueError('Candidate request/response evidence lacks saved hashes')
        for name in ('scenario.json', 'tasks.json'):
            paired_hashes[key[:2]][name].add(hashes[name])
        map_hashes.add(hashes['map.json'])
        outcomes = row['task_outcomes']
        completed = [t['delivery_completed_s'] for t in outcomes if t['delivery_completed_s'] is not None]
        if (len(outcomes) != row['num_agents'] or len(completed) != row['deliveries_completed']
                or any(not 0 <= time <= row['horizon_s'] for time in completed)
                or not np.isclose(np.mean([t['restricted_flow_s'] for t in outcomes]), row['restricted_mean_flow_s'])):
            raise ValueError('All-task native outcomes disagree with the reported metrics')
        enriched.append({**row, **audit_queries(folder, row, manifest['policy']),
                         '_study_path': str(path), '_study_name': path.name})
    for count, seed in {key[:2] for key in seen}:
        if {m for n, s, m in seen if (n, s) == (count, seed)} != methods:
            raise ValueError('Unmatched workload/method pair')
    if len(map_hashes) != 1 or any(len(h) != 1 for names in paired_hashes.values() for h in names.values()):
        raise ValueError('Matched methods must share map, starts, tasks, and scenario')
    return dict(path=path, name=path.name, role=role, manifest=manifest, rows=enriched,
                seeds=sorted({r['workload_seed'] for r in rows}), horizon_s=next(iter(horizons)),
                map_hash=next(iter(map_hashes)))


def combine_compatible_pilots(studies):
    """Connect disjoint fleet batches, preserving each original evidence source.

This never pools repetitions or upgrades exploratory recordings to validation.
Every input is independently audited by load_study before this function runs.
"""
    if len(studies) < 2:
        return studies
    first = studies[0]
    occupied = set()
    expected_methods = {r['method'] for r in first['rows']}
    for study in studies:
        if study['role'] != 'pilot':
            raise ValueError('Only exploratory batches can be combined')
        for key in ('seeds', 'horizon_s', 'map_hash'):
            if study[key] != first[key]:
                raise ValueError(f'Incompatible exploratory batches: {key}')
        for key in ('image_id', 'policy', 'source_sha256', 'compiled_source_sha256', 'accounting'):
            if study['manifest'][key] != first['manifest'][key]:
                raise ValueError(f'Incompatible frozen batch metadata: {key}')
        if {r['method'] for r in study['rows']} != expected_methods:
            raise ValueError('Combined batches must compare the same selector methods')
        cells = {(r['num_agents'], r['method']) for r in study['rows']}
        if cells & occupied:
            raise ValueError('Cannot combine overlapping fleet/method cells or pool repeated runs')
        occupied |= cells
    return [{**first, 'name': '+'.join(study['name'] for study in studies),
             'rows': [row for study in studies for row in study['rows']], 'batches': studies}]


def grouped(study):
    result = defaultdict(list)
    for row in study['rows']:
        result[row['num_agents'], row['method']].append(row)
    return result


def paired(study, count, method, field):
    zero = {r['workload_seed']: r[field] for r in study['rows'] if r['num_agents'] == count and r['method'] == 'none'}
    return [r[field] - zero[r['workload_seed']] for r in sorted(study['rows'], key=lambda r: r['workload_seed'])
            if r['num_agents'] == count and r['method'] == method]


def make_figure(studies, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)
    all_counts = sorted({r['num_agents'] for s in studies for r in s['rows']})
    identical_choices = all(r['selected_modifier_reviews'] == r['changed_route_geometry_reviews']
                            for study in studies for r in study['rows'])
    legend = [Line2D([], [], color=color, marker='o', label=label)
              for method, (label, color) in METHODS.items() if method != 'none'
              and any(r['method'] == method for s in studies for r in s['rows'])]
    plot_data = []
    for study in studies:
        groups, style = grouped(study), ':' if study['role'] == 'pilot' else '-'
        legend.append(Line2D([], [], color='#475569', ls=style,
                            label=f"{'Pilot' if study['role'] == 'pilot' else 'Fresh evaluation'}: {','.join(map(str, study['seeds']))}"))
        for method, (_, color) in METHODS.items():
            if method == 'none': continue
            counts = sorted(n for n, m in groups if m == method)
            if not counts: continue
            for ax, field in zip(axes[:2], ('deliveries_completed', 'restricted_mean_flow_s')):
                values = [paired(study, n, method, field) for n in counts]
                for count, ys in zip(counts, values): ax.scatter([count] * len(ys), ys, color=color, alpha=.25, s=15)
                ax.plot(counts, [np.mean(v) for v in values], color=color, marker='o', ls=style)
                plot_data.append(dict(study=study['name'], method=method, field=field, counts=counts, values=values))
            frequency_fields = [('modifier_percent', 'o', color)]
            if not identical_choices:
                frequency_fields.append(('geometry_change_percent', 's', 'none'))
            for field, marker, face in frequency_fields:
                ys = [np.mean([r[field] for r in groups[n, method]]) for n in counts]
                axes[2].plot(counts, ys, color=color, marker=marker, markerfacecolor=face,
                             markersize=8 if marker == 's' else 5, ls=style)
                plot_data.append(dict(study=study['name'], method=method, field=field, counts=counts, means=ys))
    if identical_choices:
        legend.append(Line2D([], [], color='#475569', marker='o', ls='', label='All selections changed accepted routes'))
    else:
        legend.extend([Line2D([], [], color='#475569', marker='o', ls='', label='Nonzero modifier selected'),
                       Line2D([], [], color='#475569', marker='s', markerfacecolor='none', ls='', label='Accepted route geometry changed')])
    for ax, title, ylabel in zip(axes, ('Deliveries versus zero costs', 'All-task time versus zero costs', 'How often routes are modified'),
                                 ('Completed deliveries: change ↑', 'Restricted mean time: change (s) ↓', 'Post-warmup reviews (%)')):
        ax.set(title=title, xlabel='Robots', ylabel=ylabel, xticks=all_counts)
        ax.axhline(0, color='#94a3b8', lw=.7); ax.grid(axis='y', alpha=.18)
        ax.set_xlim(min(all_counts) - 6, max(all_counts) + 6)
    axes[2].set_ylim(bottom=-1)
    fig.legend(handles=legend, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    for suffix in ('pdf', 'png'):
        fig.savefig(output.with_suffix('.' + suffix), dpi=190)
    plt.close(fig)
    return plot_data


def study_section(study):
    groups = grouped(study)
    identical_choices = all(r['selected_modifier_reviews'] == r['changed_route_geometry_reviews'] for r in study['rows'])
    label = 'Exploratory pilot' if study['role'] == 'pilot' else 'Fresh-workload evaluation'
    correction = study['manifest'].get('correction')
    correction_text = (r'\noindent Corrected rerun: ' + tex(correction['issue'].rstrip(' .')) + '. '
                       + tex(correction.get('scope', 'All route conditions were rerun.'))
                       + ' Earlier route recordings are archived and excluded from this comparison.'
                       if correction else '')
    host = study['manifest'].get('host', {})
    host_text = (f"Host measurements use {tex(host['platform'])} with {host['threads']} predictor thread(s); "
                 'PBS runs in a linux/amd64 container.' if host.get('platform') and host.get('threads') else '')
    lines = [r'\paragraph{' + label + '.}',
             f"Workload seed(s) {', '.join(map(str, study['seeds']))}; {study['horizon_s']:g} simulated seconds per run. "
             + ('These workloads were already available during prototype development.' if study['role'] == 'pilot'
                else 'These workloads are reported separately from prototype selection.'),
             ('Compatible batches are shown as one nested-workload comparison, with disjoint fleet/method cells and identical frozen protocol, '
              'source hashes, and simulator image. They do not add independent workload seeds.' if study.get('batches') else ''),
             correction_text,
             r'\begin{center}\small\setlength{\tabcolsep}{4pt}\begin{tabular}{'
             + ('rlrrrr' if identical_choices else 'rlrrrrr') + r'}\toprule',
             r'$N$ & Selector & Delivered & Time (s) & Select (\%) & '
             + ('' if identical_choices else r'Route (\%) & ') + r'Fail (\%)\\\midrule']
    for (count, method), rows in sorted(groups.items(), key=lambda x: (x[0][0], list(METHODS).index(x[0][1]))):
        mean = lambda field: float(np.mean([r[field] for r in rows]))
        lines.append(f"{count} & {METHODS[method][0]} & {mean('deliveries_completed'):.1f} & {mean('restricted_mean_flow_s'):.2f} & "
                     f"{mean('modifier_percent'):.1f} & "
                     + ('' if identical_choices else f"{mean('geometry_change_percent'):.1f} & ")
                     +
                     f"{100 * sum(r['primary_planner_failures'] for r in rows) / sum(r['planning_calls'] for r in rows):.1f}" + r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
              r'\noindent Time is the all-original-task restricted mean: unfinished deliveries contribute the run horizon. '
              r'Modifier and geometry percentages share all scheduled post-warmup reviews as denominator, including no-ops; '
              r'geometry compares accepted spatial/task sequences after removing timestamps and repeated WAIT samples. '
              + ('Every selected modifier changed an accepted spatial path, so one percentage is shown. ' if identical_choices else '')
              + r'Later replanning can replace unexecuted portions. PBS failure refers to the selected solve, followed by native fallback.',
              r'\begin{center}\footnotesize\setlength{\tabcolsep}{3pt}\begin{tabular}{rlrrrrr}\toprule',
              r'$N$ & Selector & PBS calls/fail/skip & ETA abstain & Host p95 (ms) & PBS p95 (s) & Barrier p95 (s)\\\midrule']
    for (count, method), rows in sorted(groups.items(), key=lambda x: (x[0][0], list(METHODS).index(x[0][1]))):
        mean = lambda field: float(np.mean([r[field] for r in rows]))
        counts = '/'.join(str(sum(r[field] for r in rows)) for field in
                          ('candidate_planning_calls', 'candidate_planning_failures', 'skipped_candidates'))
        host = '--' if method == 'none' else f"{1000 * mean('host_query_p95_s'):.0f}"
        lines.append(f"{count} & {METHODS[method][0]} & {counts} & {sum(r['invalid_eta_candidates'] for r in rows)} & {host} & "
                     f"{mean('pbs_batch_p95_s'):.2f} & {mean('barrier_p95_s'):.2f}" + r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
              r'\noindent Counts sum the runs. Host p95 covers active proposal plus selection queries; '
              r'PBS batch p95 covers post-warmup reviews and sums their candidate solves. '
              r'The whole native barrier covers all reviews and includes host work, candidate PBS solves, and communication. '
              r'Component p95s are not additive. When multiple seeds exist, entries average run-level p95s. '
              r'These are separate runtime measurements, never charges added to simulated task times. '
              r'Scorer abstentions retain zero or exclude an alternative; they include accepted services followed by native maneuvers that violate the terminal-service assumption.',
              host_text]
    barrier_values = [float(np.mean([r['barrier_p95_s'] for r in rows])) for rows in groups.values()]
    interval = study['manifest']['policy']['refresh_s']
    lines.append(f"The whole-barrier p95 exceeds the {interval:g} s review interval in "
                 f"{sum(value > interval for value in barrier_values)} of {len(barrier_values)} conditions. "
                 'A rare nonzero choice does not imply a cheap review: retaining zero can still require evaluating the full candidate menu. '
                 'Native physics waits at this barrier; these experiments demonstrate no hard real-time bound.')
    lines.append(r'\noindent ' + outcome_pattern(study))
    limited = []
    for method in ('nominal', 'rollout'):
        rows = groups.get((81, method), [])
        if not rows: continue
        reviews = sum(r['review_opportunities'] for r in rows)
        valid = sum(r['alternative_comparison_reviews'] for r in rows)
        failures = sum(r['baseline_failed_reviews'] for r in rows)
        abstentions = sum(r['baseline_eta_abstention_reviews'] for r in rows)
        limited.append(f"{METHODS[method][0]} had {valid}/{reviews} reviews with a comparable alternative "
                       f"({failures} baseline failures; {abstentions} baseline-scorer abstentions)")
    if limited:
        lines.append('At 81 robots, ' + '; '.join(limited) + '. These limits reduce opportunities to change a route.')
    lines.append('These results describe one nested workload; independent workload validation remains outstanding.')
    return '\n'.join(lines)


def outcome_pattern(study):
    """Describe completion/time patterns without ranking negligible differences."""
    groups = grouped(study)
    counts = sorted({n for n, _ in groups})
    sentences = []
    for count in counts:
        methods = [method for method in METHODS if (count, method) in groups]
        delivery = {method: np.mean([r['deliveries_completed'] for r in groups[count, method]]) for method in methods}
        flow = {method: np.mean([r['restricted_mean_flow_s'] for r in groups[count, method]]) for method in methods}
        alternatives = [method for method in methods if method != 'none']
        if all(np.isclose(value, count) for value in delivery.values()):
            if all(flow[method] < flow['none'] for method in alternatives):
                sentences.append(f'At {count} robots, both selectors finish the same complete batch earlier.')
            else:
                sentences.append(f'At {count} robots, all methods finish the batch; time distinguishes them.')
        elif all(delivery[method] > delivery['none'] for method in alternatives):
            sentence = f'At {count} robots, both selectors also increase completed deliveries'
            if 'rollout' in alternatives and all(delivery['rollout'] >= delivery[m] and flow['rollout'] <= flow[m]
                                                for m in methods):
                sentence += ', with joint ETA giving the strongest observed result'
            sentences.append(sentence + '.')
        elif len({float(value) for value in delivery.values()}) == 1:
            sentences.append(f'At {count} robots, delivery counts do not improve; these recordings do not establish a reliable time benefit.')
        else:
            sentences.append(f'At {count} robots, the selectors produce different delivery/time tradeoffs.')
    if any(not r['selected_modifier_reviews'] and r['method'] != 'none' for r in study['rows']):
        sentences.append('Where no modifier was selected, time differences from zero costs reflect execution/planner variation.')
    return ' '.join(sentences)


def render(studies, output_root):
    output_root = Path(output_root)
    (output_root / 'figures').mkdir(parents=True, exist_ok=True)
    (output_root / 'results').mkdir(parents=True, exist_ok=True)
    plot_data = make_figure(studies, output_root / 'figures/route_search_outcomes')
    preview = make_preview_figure(studies, output_root / 'figures/route_search_preview')
    if preview is None:
        for suffix in ('pdf', 'png'):
            stale = output_root / f'figures/route_search_preview.{suffix}'
            if stale.exists(): stale.unlink()
    lines = [r'\section{Selecting routing costs using predicted outcomes}\label{sec:route-search}',
             r'With the \emph{Safety rule 2} safety rule, the simulator replans jointly every 2 seconds. At each post-warmup review, '
             r'the host proposes zero, mild, and strong nonnegative road costs from one 20-second slow-traffic forecast '
             r'(weights $0,2,8$; the existing three windows and $3\times3$ expansion). '
             r'Costs are added to MOVE destinations; WAIT costs are unchanged. '
             r'PBS solves these candidates from one commitment cut, with the same starts, tasks, random seed, and per-candidate budget. '
             r'The zero-cost control solves only its unchanged problem. Nominal and rollout use the same forecast-based menu construction '
             r'and candidate budget. Both rank actual candidate plans: the comparison tests joint ETA with safety and dependency effects '
             r'against nominal free-flow ETA. Their closed-loop states and later menus can diverge.',
             r'Guidance Graph Optimization changes movement/wait edge costs to guide a lifelong MAPF planner~\cite{ggo}; '
             r'its online extension optimizes adaptive guidance from current traffic, specifically with PIBT~\cite{online-ggo}. '
             r'Our three-cost-menu PBS experiment uses the same high-level idea of influencing costs while leaving route choice to the planner. '
             r'It does not reproduce those optimization algorithms or their experimental settings.',
             r'For robot $i$, let $T_i^k$ be seconds from the current origin until completion of the same pending terminal service on candidate $k$, '
             r'and $R_i^k(H)$ its remaining nominal route work at horizon $H=30$ s. The rollout score is',
             r'\[ J_k=\frac{1}{N}\sum_{i=1}^{N}\widehat T_i^k,\qquad '
             r'\widehat T_i^k=\begin{cases}T_i^k,&\text{terminal service completes by }H,\\'
             r'H+R_i^k(H),&\text{otherwise.}\end{cases} \]',
             r'Robots with native final-service acknowledgements contribute zero. Nominal selection uses $R_i^k(0)$ instead. '
             r'Remaining work charges travel, explicit turns, and service; a longer detour therefore costs time. '
             r'All pending service IDs/goals must agree across candidates. Choose the minimum-score successful candidate only if '
             r'$J_0-J_k>0.5$ s; zero wins ties and smaller gains. A failed zero-cost solve keeps the normal native fallback and skips alternatives. '
             r'The simulator installs exactly the evaluated action graph, with no host admission hold.',
             r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/route_search_outcomes.pdf}',
             r'\caption{Paired native-time outcomes and route-modification frequency. Positive delivery changes and negative time changes are favorable. '
             r'Points/lines are descriptive workload observations/means, not confidence intervals; study stages remain separate.}\end{figure}']
    lines.extend(study_section(study) for study in studies)
    if preview is not None:
        lines += [r'\begin{figure}[ht]\centering\includegraphics[width=.95\linewidth]{figures/route_search_preview.pdf}',
                  r'\caption{Candidate route previews, not realized future trajectories. '
                  + f"At {preview['clock'] / 10:g} s in workload {preview['workload_seed']} ({preview['num_agents']} robots), "
                  + f"the rollout selector chose the {tex(preview['selected_name'])} cost field. "
                  + 'This origin has the largest predicted fleet-mean ETA gain among selected rollout branches with changed routes; '
                  + 'up to five changed robots with greatest predicted individual ETA gains are shown. '
                  + 'The two panels retain the same observed starts and terminal services. Aggregate outcomes use complete recordings.'
                  + r'}\end{figure}']
    lines += [r'\paragraph{Interpretation limits.} Candidate rollouts model safety and dependencies but hold the accepted plans fixed; '
              r'future PBS replanning is absent. Their free-flow tail is optimistic about future safety, polling, and dependency delays. '
              r'The target is the common accepted terminal service, which may be a pickup before an unplanned later delivery. '
              r'The native all-task outcomes remain the decision criterion. Nominal and rollout spend extra PBS solves compared with zero costs; '
              r'comparing them to each other is therefore necessary to assess the added value of the joint ETA model. '
              r'CPU-budgeted planning can vary between recordings even with the same workload. '
              r'Frozen execution sources and candidate request/response evidence are retained with each study.']
    (output_root / 'results/route_search_results.tex').write_text('\n\n'.join(lines) + '\n')
    batches = [batch for study in studies for batch in study.get('batches', [study])]
    provenance = dict(builder_sha256=digest(__file__), accounting='native_time_only', preview=preview,
                     display_groups=[dict(name=s['name'], batches=[b['name'] for b in s.get('batches', [s])]) for s in studies], studies=[
        dict(name=s['name'], role=s['role'], manifest_sha256=digest(s['path'] / 'manifest.json'),
             results_sha256=digest(s['path'] / 'results.json'), seeds=s['seeds'],
             correction=s['manifest'].get('correction')) for s in batches], plots=plot_data,
        run_summaries=[{key: row[key] for key in (
            'run_id', 'num_agents', 'workload_seed', 'method', 'deliveries_completed', 'restricted_mean_flow_s',
            'selected_modifier_reviews', 'changed_route_geometry_reviews', 'review_opportunities',
            'active_search_reviews', 'alternative_comparison_reviews', 'baseline_failed_reviews',
            'baseline_eta_abstention_reviews', 'candidate_planning_calls', 'candidate_planning_failures',
            'skipped_candidates', 'invalid_eta_candidates', 'host_query_p95_s', 'pbs_batch_p95_s',
            'barrier_p95_s', 'candidate_pbs_total_s', 'total_barrier_s', 'recording_wall_s', 'selection_reasons')}
            for study in studies for row in study['rows']])
    (output_root / 'results/route_search_results_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def self_test():
    # Pure geometry and selection audits; all fixture files stay temporary.
    short = {'plan': {'plan': [[[1, 2, 0, -1], [1, 3, 1, -1], [1, 3, 2, 7]]]}}
    wait = {'plan': {'plan': [[[1, 2, 0, -1], [1, 2, 1, -1], [1, 3, 4, -1], [1, 3, 9, 7]]]}}
    assert geometry(short) == geometry(wait)
    wrong = read_json_copy(wait); wrong['plan']['plan'][0][-1][-1] = 8
    assert geometry(short) != geometry(wrong)
    first = dict(name='one', role='pilot', seeds=[7], horizon_s=240, map_hash='map',
                 manifest=dict(image_id='image', policy={'a': 1}, source_sha256={'p': 'hash'},
                               compiled_source_sha256={'c': 'hash'}, accounting='native_time_only'),
                 rows=[dict(num_agents=27, method=method) for method in ('none', 'rollout')])
    second = read_json_copy(first); second['name'] = 'two'
    for row in second['rows']: row['num_agents'] = 54
    combined = combine_compatible_pilots([first, second])
    assert len(combined) == 1 and len(combined[0]['rows']) == 4 and len(combined[0]['batches']) == 2
    for bad in (first, {**second, 'seeds': [8]},
                {**second, 'manifest': {**second['manifest'], 'source_sha256': {'p': 'changed'}}}):
        try: combine_compatible_pilots([first, bad])
        except ValueError: pass
        else: raise AssertionError('Incompatible or overlapping batches combined')
    with tempfile.TemporaryDirectory(prefix='route-render-test-') as temporary:
        folder = Path(temporary)
        proof = dict(image_id='corrected', compiled_source_sha256={'solver': 'hash'},
                     physical_client_binaries_identical=True,
                     native_checks=['pbs_path_export_service_markers'])
        (folder / 'version-independent-proof.json').write_text(json.dumps(proof))
        proof_manifest = dict(build_proof='version-independent-proof.json',
                              source_sha256={'version-independent-proof.json': 'verified-earlier'},
                              image_id='corrected', compiled_source_sha256={'solver': 'hash'},
                              correction={'issue': 'service marker'})
        validate_build_proof(proof_manifest, folder)
        try: validate_build_proof({**proof_manifest, 'image_id': 'different'}, folder)
        except ValueError: pass
        else: raise AssertionError('Build-proof image mismatch accepted')
        for name in ('bridge', 'plan_search'): (folder / name).mkdir()
        rows = []
        for clock in (0, 20):
            candidates, scores, menu = [], [], []
            for index, eta in enumerate((10., 9., 9.5)):
                name = ('zero', 'mild', 'strong')[index]
                candidate = dict(index=index, name=name, evaluated=True, success=True,
                                 nonzero_penalty_cells=int(index != 0), max_cell_penalty=float(index),
                                 planner_wall_s=.01, plan=short['plan'])
                candidates.append(candidate); menu.append(dict(name=name, cell_penalties=[float(index)]))
                scores.append(dict(index=index, name=name, evaluated=True, success=True,
                                   native_plan_sha256=canonical_digest(candidate['plan']), eligible=True,
                                   mean_eta_s=eta, rejection_reason=None))
            selected = 1
            query = dict(clock=clock, selected_index=selected, selected_name='mild', baseline_failed=False,
                         reason='decision', host_query_s=.1, proposal={'distinct_penalty_fields': 3},
                         selection=dict(candidates=scores, selected_index=selected,
                                        minimum_mean_eta_gain_s=.5, selection_reason='predicted_mean_eta_improvement'))
            rows.append(query)
            for sub, kind, data in (
                ('plan_search', 'request', dict(clock=clock, boundary='post_commit_before_plan_selection', candidates=candidates)),
                ('plan_search', 'response', dict(clock=clock, selected_index=selected)),
                ('bridge', 'response', dict(clock=clock, cell_penalties=[0.], route_candidates=menu))):
                (folder / sub / f'{kind}_{clock}.json').write_text(json.dumps(data))
        (folder / 'host_queries.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        spec = dict(horizon_s=4, method='rollout', minimum_mean_eta_gain_s=.5,
                    selected_modifier_reviews=2, changed_route_geometry_reviews=0,
                    candidate_planning_calls=6, candidate_planning_failures=0,
                    baseline_failed_reviews=0, active_search_reviews=2, planning_calls=2,
                    scheduled_reviews_after_warmup=2)
        summary = audit_queries(folder, spec, {'refresh_s': 2., 'warmup_s': 0.})
        assert summary['modifier_percent'] == 100. and summary['geometry_change_percent'] == 0.
        assert summary['pbs_batch_p95_s'] == .03 and summary['eta_scored_candidates'] == 6
        rows[0]['selection']['candidates'][1]['mean_eta_s'] = 9.5
        (folder / 'host_queries.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        try: audit_queries(folder, spec, {'refresh_s': 2., 'warmup_s': 0.})
        except ValueError: pass
        else: raise AssertionError('Margin violation accepted')
    print('Route-search builder self-test passed; no study results were fabricated.')


def read_json_copy(value):
    return json.loads(json.dumps(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', nargs='+', type=Path)
    parser.add_argument('--evaluation-studies', nargs='*', type=Path, default=[])
    parser.add_argument('--combine-compatible-pilots', action='store_true',
                        help='Connect disjoint fleet batches only when seeds, image, protocol, sources and map match.')
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.studies: parser.error('Supply completed --studies, or --self-test')
    paths = [directory(path) for path in args.studies]
    evaluation = {directory(path) for path in args.evaluation_studies}
    if len(set(paths)) != len(paths) or not evaluation <= set(paths):
        parser.error('Study paths must be unique and evaluation paths included in --studies')
    studies = [load_study(path, 'evaluation' if path in evaluation else 'pilot') for path in paths]
    pilot_seeds = {seed for s in studies if s['role'] == 'pilot' for seed in s['seeds']}
    if pilot_seeds & {seed for s in studies if s['role'] == 'evaluation' for seed in s['seeds']}:
        raise ValueError('Fresh evaluation cannot reuse a pilot workload seed')
    if args.combine_compatible_pilots:
        studies = combine_compatible_pilots([s for s in studies if s['role'] == 'pilot']) + [
            s for s in studies if s['role'] != 'pilot']
    render(studies, args.output_root)
    print('Rendered verified route-search studies:', ', '.join(s['name'] for s in studies))


if __name__ == '__main__':
    main()
