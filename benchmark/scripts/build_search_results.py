#!/usr/bin/env python3
"""Render completed, paired admission-search studies using native time only.

All inputs are exploratory by default. Explicit --evaluation-studies labels
separate fresh-workload evaluation without pooling it with horizon selection.
--followup-studies labels a later density selected after inspecting that first
evaluation, rather than implying the follow-up density was selected beforehand.
Nothing is written until every requested study passes evidence checks.
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

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'none': ('No gate', '#64748b'), 'reactive': ('Current-state gate', '#d97706'),
           'search_h5': ('Search, 5 s', '#2563eb'), 'search_h10': ('Search, 10 s', '#7c3aed'),
           'search_h10_hold6': ('Search, 6 s hold', '#15803d')}
ROLES = {'pilot': 'Exploratory pilot', 'evaluation': 'Fresh-workload evaluation',
         'followup': 'Follow-up at the selected density'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def directory(path):
    path = Path(path).resolve()
    return path.parent if path.name == 'manifest.json' else path


def tex(value):
    replacements = {'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#', '$': r'\$',
                    '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def query_summary(queries, horizon_s, refresh_s, warmup_s, method):
    expected = list(range(0, round(horizon_s * 10), round(refresh_s * 10)))
    if [q['clock'] for q in queries] != expected:
        raise ValueError('Missing, duplicate, or out-of-order decision review records')
    reviews = [q for q in queries if q['clock'] >= round(warmup_s * 10)]
    if any(q['held_robot_ids'] for q in queries if q['clock'] < round(warmup_s * 10)):
        raise ValueError('Intervention during the declared warmup')
    if method == 'none' and any(q['held_robot_ids'] for q in queries):
        raise ValueError('No-gate control contains an intervention')
    active = [q for q in reviews if q['reason'] == 'decision']
    result = {'review_opportunities': len(reviews),
              'intervention_reviews': sum(bool(q['held_robot_ids']) for q in reviews),
              'decision_reviews': len(active), 'actual_query_count': 0, 'candidate_branches_including_noop': 0,
              'candidate_interventions_tested': 0, 'positive_gain_candidates': 0,
              'positive_gain_origins': 0, 'no_alternative_queries': 0,
              'no_positive_gain_queries': 0, 'below_margin_queries': 0,
              'fresh_hold_selections': 0, 'continuing_hold_reviews': 0, 'forced_release_reviews': 0}
    fresh_times = []
    if method.startswith('search_'):
        episode = None
        for query in active:
            detail = query['diagnostics']
            candidates = detail['candidates']
            if len(candidates) != detail['candidate_count'] or not 0 <= len(candidates) <= 9:
                raise ValueError('Counterfactual candidate count disagrees with search budget')
            clock = query['clock']
            if not candidates:
                if method != 'search_h10_hold6' or episode is None:
                    raise ValueError('Only a recorded long-hold episode permits a zero-candidate review')
                same_episode = (detail.get('active_until_clock') == episode['active_until']
                                and detail.get('forced_release_until_clock') == episode['release_until']
                                and detail.get('selection_origin_clock') == episode['origin'])
                if not same_episode or detail.get('new_hold_started'):
                    raise ValueError('Continuation/release does not match its original hold episode')
                if detail.get('continuing_hold'):
                    if (detail.get('selection_reason') != 'continuing_selected_hold'
                            or not episode['origin'] < clock < episode['active_until']
                            or query['held_robot_ids'] != episode['ids']):
                        raise ValueError('Long-hold continuation changed IDs or exceeded its expiry')
                    result['continuing_hold_reviews'] += 1
                else:
                    if (detail.get('selection_reason') != 'forced_release_after_hold'
                            or query['held_robot_ids'] or not episode['active_until'] <= clock < episode['release_until']):
                        raise ValueError('Invalid forced-release review')
                    result['forced_release_reviews'] += 1
                if detail.get('best_gain_s') is not None or detail.get('selected_gain_s') is not None:
                    raise ValueError('A continuation/release cannot claim a new predicted gain')
                continue
            if episode is not None and clock < episode['release_until']:
                raise ValueError('Fresh search occurred during the retained hold/release episode')
            if candidates[0]['held_robot_ids'] or candidates[0]['name'] != 'noop':
                raise ValueError('Search must include the no-op counterfactual first')
            best_gain = float(detail['best_gain_s'])
            margin = float(detail['required_gain_s'])
            if not np.isfinite([best_gain, margin]).all():
                raise ValueError('Nonfinite search gain or margin')
            chosen = bool(query['held_robot_ids'])
            if chosen != (best_gain > margin + 1e-9):
                raise ValueError('Selected intervention violates the frozen no-op margin rule')
            result['actual_query_count'] += 1
            result['fresh_hold_selections'] += chosen
            fresh_times.append(query['host_query_s'])
            if method == 'search_h10_hold6':
                if bool(detail.get('new_hold_started')) != chosen or detail.get('continuing_hold'):
                    raise ValueError('Fresh long-hold selection flags disagree with the chosen action')
                if chosen:
                    episode = {'origin': clock, 'ids': query['held_robot_ids'],
                               'active_until': clock + 60, 'release_until': clock + 80}
                    if (detail.get('active_until_clock') != episode['active_until']
                            or detail.get('forced_release_until_clock') != episode['release_until']):
                        raise ValueError('Long-hold episode must last six seconds followed by two seconds release')
                else:
                    episode = None
            result['candidate_branches_including_noop'] += len(candidates)
            result['candidate_interventions_tested'] += len(candidates) - 1
            result['positive_gain_candidates'] += sum(c['gain_over_noop_s'] > 1e-9 for c in candidates[1:])
            result['positive_gain_origins'] += best_gain > 1e-9
            if not chosen:
                category = ('no_alternative_queries' if len(candidates) == 1 else
                            'no_positive_gain_queries' if best_gain <= 1e-9 else 'below_margin_queries')
                result[category] += 1
    else:
        result['actual_query_count'] = len(active)
        fresh_times = [query['host_query_s'] for query in active]
    result['intervention_percent'] = 100 * result['intervention_reviews'] / len(reviews) if reviews else 0.
    result['fresh_query_p95_s'] = float(np.quantile(fresh_times, .95)) if fresh_times else 0.
    return result


def load_study(path, role='pilot', workspace=ROOT.parent):
    path, workspace = directory(path), Path(workspace)
    manifest = read(path / 'manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('accounting') != 'native_time_only':
        raise ValueError(f'Need a complete native-time study: {path}')
    policy = manifest['policy']
    if policy.get('oracle') is not False or policy.get('future_replanning_modeled') is not False:
        raise ValueError('This renderer expects causal fixed-plan counterfactual search')
    source_root = path / 'source_snapshot' if (path / 'source_snapshot').exists() else workspace
    if not manifest.get('source_sha256'):
        raise ValueError('Missing frozen execution-source hashes')
    for name, expected in manifest['source_sha256'].items():
        if digest(source_root / name) != expected:
            raise ValueError(f'Frozen search source changed: {name}')
    for name, expected in manifest.get('compiled_server_sha256', {}).items():
        if digest(source_root / 'simulator/server' / name) != expected:
            raise ValueError(f'Compiled native source changed: {name}')
    rows, specs = read(path / 'results.json'), manifest['runs']
    specs_by_id = {r['run_id']: r for r in specs}
    if (len(rows) != len(specs) or len(specs_by_id) != len(specs)
            or {r['run_id'] for r in rows} != set(specs_by_id)
            or len({r['run_id'] for r in rows}) != len(rows)):
        raise ValueError('Results do not match the complete frozen run list')
    horizons = {r['horizon_s'] for r in rows}
    methods = {r['method'] for r in rows}
    if len(horizons) != 1 or not methods <= METHODS.keys() or 'none' not in methods or len(methods) < 2:
        raise ValueError('Each study needs one horizon, no gate, and at least one comparison method')
    seen, paired_inputs, map_hashes = set(), defaultdict(lambda: defaultdict(set)), set()
    enriched = []
    for row in rows:
        spec, folder = specs_by_id[row['run_id']], path / row['run_id']
        key = (row['num_agents'], row['workload_seed'], row['method'])
        if key in seen:
            raise ValueError('Duplicate fleet/workload/method recording')
        seen.add(key)
        if any(spec[k] != row[k] for k in ('num_agents', 'workload_seed', 'method', 'horizon_s', 'trigger')):
            raise ValueError('Run outcome does not match its frozen specification')
        if spec['image_id'] != manifest['image_id'] or row['trigger'] != 'outer_overlap':
            raise ValueError('Native image or safety rule differs from the declared comparison')
        if read(folder / 'run_spec.json') != spec or read(folder / 'search_metrics.json') != row:
            raise ValueError('Run metadata/metrics changed after scoring')
        hashes = row.get('source_sha256', {})
        if not {'simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'run_spec.json'} <= hashes.keys():
            raise ValueError('Missing native evidence hashes')
        for name, expected in hashes.items():
            if digest(folder / name) != expected:
                raise ValueError(f'Native evidence changed: {row["run_id"]}/{name}')
        for name in ('scenario.json', 'tasks.json'):
            paired_inputs[key[:2]][name].add(hashes[name])
        map_hashes.add(digest(folder / 'map.json'))
        outcomes = row['task_outcomes']
        completed = [t['delivery_completed_s'] for t in outcomes if t['delivery_completed_s'] is not None]
        if (len(outcomes) != row['num_agents'] or len(completed) != row['deliveries_completed']
                or any(not 0 <= time <= row['horizon_s'] for time in completed)
                or not np.isclose(np.mean([t['restricted_flow_s'] for t in outcomes]), row['restricted_mean_flow_s'])):
            raise ValueError('All-task outcomes and summary metrics disagree')
        queries = [json.loads(line) for line in (folder / 'host_queries.jsonl').read_text().splitlines()]
        summary = query_summary(queries, row['horizon_s'], policy['refresh_s'], policy['warmup_s'], row['method'])
        if row['planning_calls'] != len(queries):
            raise ValueError('Planning-call count and decision boundaries disagree')
        if row['intervention_decisions'] != summary['intervention_reviews']:
            raise ValueError('Intervention count differs from recorded selected actions')
        enriched.append({**row, **summary})
    for count, seed in {key[:2] for key in seen}:
        if {m for n, s, m in seen if (n, s) == (count, seed)} != methods:
            raise ValueError('Every method must have a matched no-gate workload')
    if len(map_hashes) != 1 or any(len(values) != 1 for group in paired_inputs.values() for values in group.values()):
        raise ValueError('Paired methods must share map, tasks, and initial scenario')
    return {'path': path, 'name': path.name, 'role': role, 'manifest': manifest,
            'rows': enriched, 'horizon_s': next(iter(horizons)), 'map_hash': next(iter(map_hashes)),
            'seeds': sorted({r['workload_seed'] for r in enriched})}


def grouped(study):
    groups = defaultdict(list)
    for row in study['rows']:
        groups[row['num_agents'], row['method']].append(row)
    return groups


def paired(study, count, method, field):
    baseline = {r['workload_seed']: r[field] for r in study['rows']
                if r['num_agents'] == count and r['method'] == 'none'}
    return [r[field] - baseline[r['workload_seed']] for r in sorted(study['rows'], key=lambda r: r['workload_seed'])
            if r['num_agents'] == count and r['method'] == method]


def make_figure(studies, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)
    plot_data = []
    all_counts = sorted({row['num_agents'] for study in studies for row in study['rows']})
    stage_handles = []
    role_counts = defaultdict(int)
    for study in studies:
        role_index = role_counts[study['role']]
        role_counts[study['role']] += 1
        marker = ('D' if study['role'] == 'followup' else 'o') if role_index == 0 else ('s', '^', 'v')[min(role_index - 1, 2)]
        linestyle = ':' if study['role'] == 'pilot' else '--' if study['role'] == 'followup' else '-'
        seed_label = ','.join(map(str, study['seeds']))
        stage_label = {'pilot': 'Pilot', 'evaluation': 'First fresh evaluation', 'followup': 'Selected-density follow-up'}[study['role']]
        stage_handles.append(Line2D([], [], color='#475569', marker=marker, ls=linestyle,
                            markerfacecolor='white' if study['role'] == 'pilot' else '#475569',
                            label=f'{stage_label} ({seed_label})'))
        groups = grouped(study)
        for method, (_, color) in METHODS.items():
            if method == 'none':
                continue
            counts = sorted(n for n, m in groups if m == method)
            if not counts:
                continue
            for ax, field in zip(axes, ('deliveries_completed', 'restricted_mean_flow_s', 'intervention_percent')):
                values = [paired(study, count, method, field) if field != 'intervention_percent'
                          else [r[field] for r in groups[count, method]] for count in counts]
                means = [float(np.mean(value)) for value in values]
                for count, observations in zip(counts, values):
                    ax.scatter(np.full(len(observations), count), observations, color=color, alpha=.22, s=14, zorder=2)
                ax.plot(counts, means, color=color, lw=1.6,
                        linestyle=linestyle, marker=marker, ms=6,
                        markerfacecolor='white' if study['role'] == 'pilot' else color, markeredgewidth=1.4)
                plot_data.append({'study': study['name'], 'role': study['role'], 'method': method,
                                  'field': field, 'fleet_sizes': counts, 'values': values, 'means': means})
    for ax in axes[:2]:
        ax.axhline(0, color='#475569', lw=.8, alpha=.6)
    axes[0].set(title='Delivery change', ylabel='Completed deliveries minus no gate')
    axes[1].set(title='Completion-time change', ylabel='Native mean flow minus no gate (s)')
    axes[2].set(title='How often we intervene', ylabel='Scheduled review ticks with a hold (%)', ylim=(-2, 102))
    for ax in axes:
        ax.set(xlabel='Number of robots', xticks=all_counts)
        if len(all_counts) == 1:
            ax.set_xlim(all_counts[0] - 4, all_counts[0] + 4)
        ax.grid(alpha=.16)
    present = {r['method'] for study in studies for r in study['rows']}
    handles = [Line2D([], [], color=color, lw=1.8, label=label)
               for method, (label, color) in METHODS.items() if method != 'none' and method in present]
    handles.extend(stage_handles)
    fig.legend(handles=handles, loc='outside lower center', ncol=3 if len(handles) > 5 else len(handles), frameon=False, fontsize=8)
    fig.savefig(output / 'search_outcomes.pdf', bbox_inches='tight')
    fig.savefig(output / 'search_outcomes.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return plot_data


def study_table(study):
    groups = grouped(study)
    rows = []
    for count in sorted({key[0] for key in groups}):
        for method, (label, _) in METHODS.items():
            entries = groups.get((count, method))
            if not entries:
                continue
            mean = lambda key: float(np.mean([r[key] for r in entries]))
            failure = 100 * sum(r['primary_planner_failures'] for r in entries) / sum(r['planning_calls'] for r in entries)
            hold = 100 * sum(r['intervention_reviews'] for r in entries) / sum(r['review_opportunities'] for r in entries)
            rows.append(f'{count} & {label} & {mean("deliveries_completed"):.1f} & '
                        f'{mean("restricted_mean_flow_s"):.1f} & {hold:.1f} & {failure:.1f} & '
                        f'{1000 * mean("fresh_query_p95_s"):.0f} \\\\')
    role = ROLES[study['role']]
    seeds = ', '.join(str(seed) for seed in study['seeds'])
    policy = study['manifest']['policy']
    reviews = sorted({r['review_opportunities'] for r in study['rows']})
    return (r'\begin{table}[H]\centering\small' + '\n'
            + r'\begin{tabular}{@{}rlrrrrr@{}}\toprule' + '\n'
            + r'$N$ & Method & Delivered & Flow (s) & Hold (\%) & PBS fail (\%) & p95 (ms) \\ \midrule' + '\n'
            + '\n'.join(rows) + '\n' + r'\bottomrule\end{tabular}' + '\n'
            + f'\\caption{{{role}: \\texttt{{{tex(study["name"])}}}, workload seeds {seeds}, '
              f'{study["horizon_s"]:g} simulated seconds. Means are over workloads. '
              f'Hold percentage uses all {"/".join(map(str, reviews))} scheduled post-warmup reviews per run, '
              'including no-op and after-completion reviews. Runtime is the mean of per-run fresh-query 95th percentiles '
              '(continuation/release reviews excluded); '
              'no gate has no query and is shown as zero. '
              f'Frozen gain margins: $\\delta={policy["absolute_gain_s"]:g}$, '
              f'$\\rho={policy["relative_gain_fraction"]:g}$.}}\n' + r'\end{table}' + '\n')


def search_audit(study):
    paragraphs = []
    for method, (label, _) in METHODS.items():
        entries = [r for r in study['rows'] if r['method'] == method]
        if not method.startswith('search_') or not entries:
            continue
        total = lambda key: sum(r[key] for r in entries)
        carry = (f', {total("continuing_hold_reviews")} continuation reviews' if method == 'search_h10_hold6' else '')
        paragraphs.append(f'{label}: {total("positive_gain_candidates")}/{total("candidate_interventions_tested")} '
            f'candidate interventions predicted positive gains; {total("fresh_hold_selections")}/{total("actual_query_count")} '
            f'fresh searches selected a hold{carry}; holds were requested at '
            f'{total("intervention_reviews")}/{total("review_opportunities")} scheduled reviews. '
            f'Fresh no-action decisions: {total("no_alternative_queries")} with no alternative, '
            f'{total("no_positive_gain_queries")} with no positive gain, '
            f'{total("below_margin_queries")} below the gain margin.')
    return ' '.join(paragraphs)


def interpretation(study):
    groups = grouped(study)
    inactive = []
    patterns = []
    for method, (label, _) in METHODS.items():
        entries = [r for r in study['rows'] if r['method'] == method]
        if method == 'none' or not entries:
            continue
        favorable, mixed = [], []
        for count in sorted(n for n, m in groups if m == method):
            if not sum(r['intervention_reviews'] for r in groups[count, method]):
                inactive.append(f'{label} ({count} robots)')
                continue
            delivery = np.mean(paired(study, count, method, 'deliveries_completed'))
            flow = np.mean(paired(study, count, method, 'restricted_mean_flow_s'))
            (favorable if delivery >= 0 and flow < 0 else mixed).append(str(count))
        if favorable:
            patterns.append(f'{label} has favorable paired native means at {", ".join(favorable)} robots; this remains exploratory.')
        if mixed:
            patterns.append(f'{label} does not improve both native outcome means at {", ".join(mixed)} robots.')
    if inactive:
        patterns.insert(0, '; '.join(inactive) + ' requested no holds. Their paired outcome differences reflect '
                        'execution/planner variation, not a demonstrated benefit from a search intervention.')
    return ' '.join(patterns)


def make_section(studies):
    text = r'''\section{Search over short admission interventions}
\label{sec:search-admission}
The controller now compares an intervention with doing nothing before applying it, using a short-horizon search related to model-predictive control~\cite{mpc}. After native planning, a fresh accepted-plan snapshot initializes every branch. Candidates are no action, the nearest eligible robot on each of four approaches, and each entire eligible approach: at most nine distinct branches. Robots already inside the central junction and committed/queued entry actions are excluded. The native admission mechanism preserves retained action commitments.

Let $w_i^a(k)$ be the accepted work advanced by robot $i$ under candidate $a$ at step $k$: signed MOVE distance divided by nominal speed, TURN progress divided by nominal turn rate, and actual SERVICE time consumed. Search maximizes
\begin{equation}
 S(a)=\frac1K\sum_{k=0}^{K-1}\sum_i w_i^a(k),\quad K=H/0.1,
 \qquad S(a^*)-S(0)>\max\{\delta,\rho|S(0)|\}.
\end{equation}
The inequality is required before acting; ties and insufficient gains choose no action. This is all-fleet progress in nominal work-seconds, not predicted delivery count. Waiting earns no progress; a short accepted-plan endpoint earns no task-completion bonus. The original search variants predict $H=5$ or $10$ seconds and select a two-second hold before searching again. An additional \emph{Search, 6 s hold} variant uses $H=10$ and predicts a six-second hold: chosen IDs are retained over three two-second reviews without a new search. Each native response still expires after two seconds. Both designs enforce a two-second release after six seconds of consecutive requested holding. The default absolute margin is $\delta=0.5$ and relative margin $\rho=0$; thresholds remain exploratory.

\textbf{Outcome accounting.} All deliveries and flow times use the native simulator clock; computation is reported separately. Restricted flow averages $\min(C_i,T)$ over all original tasks, including unfinished tasks at the horizon $T$. PBS failures are primary planning calls that fail within budget and invoke the unchanged fallback; they do not establish infeasibility.
'''
    for role, heading in ROLES.items():
        selected = [study for study in studies if study['role'] == role]
        if not selected:
            continue
        text += '\n\\subsection{' + heading + '}\n'
        if role == 'pilot':
            text += 'These workloads support method/horizon selection. They are not independent evidence for the selected design.\n'
        elif role == 'evaluation':
            text += 'These workload seeds are disjoint from the displayed pilot. They are shown separately; a small fresh-seed comparison is not general validation.\n'
        else:
            text += ('This density was selected after inspecting the first fresh-workload results. New seeds compare the chosen search '
                     'with matched controls at that selected density; this is an adaptive follow-up, not an independently preselected '
                     'full fleet-size evaluation. It is reported separately without pooling stages.\n')
        for study in selected:
            text += study_table(study) + '\n' + interpretation(study) + ' ' + search_audit(study) + '\n'
    text += r'''
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/search_outcomes.pdf}
\caption{Changes relative to the same study's matched no-gate workload: more deliveries and less flow time are favorable. Colored points/lines are workload means; pale points are individual paired workloads, not confidence intervals. Distinct markers identify the pilot, first fresh evaluation, and any adaptive follow-up; stages are never pooled. The third panel uses every scheduled post-warmup review as its denominator (114 for a 240\,s run). A method requesting no holds makes no control intervention; its outcome differences reflect execution/planner variation.}
\label{fig:search-outcomes}
\end{figure}

\textbf{Interpretation limits.} These are action-conditioned surrogate predictions, not native simulator checkpoints or an oracle. Accepted plans remain fixed within each branch; future PBS replanning and commitment promotions, exact angular/contact dynamics, and exact callback ordering are not reproduced. Native guards may bypass a retained hold after a later commitment promotion. A predicted progress gain can therefore fail to improve native task outcomes. Selecting no action frequently is intended when modeled gains are absent or too small; it does not establish that no beneficial action exists. There is no MPC performance, stability, or deadlock guarantee. The experiment tests this candidate set and progress objective on this floor. Short, CPU-budgeted native runs can vary between executions, especially at 81 robots. Independent workload seeds and repeated native runs are needed before claiming a reliable effect.
'''
    return text.strip() + '\n'


def build(paths, evaluation_paths=(), output_root=ROOT, workspace=ROOT.parent, followup_paths=()):
    paths = [directory(path) for path in paths]
    evaluations = {directory(path) for path in evaluation_paths}
    followups = {directory(path) for path in followup_paths}
    if (len(set(paths)) != len(paths) or not evaluations | followups <= set(paths)
            or evaluations & followups):
        raise ValueError('Study paths must be unique; evaluation/follow-up labels must be disjoint subsets of --studies')
    studies = [load_study(path, 'followup' if path in followups else 'evaluation' if path in evaluations else 'pilot', workspace)
               for path in paths]
    role_seeds = {role: {seed for study in studies if study['role'] == role for seed in study['seeds']} for role in ROLES}
    for first, second in [('pilot', 'evaluation'), ('pilot', 'followup'), ('evaluation', 'followup')]:
        if role_seeds[first] & role_seeds[second]:
            raise ValueError('Pilot, first fresh evaluation, and follow-up must use disjoint workload seeds')
    if len({study['map_hash'] for study in studies}) != 1:
        raise ValueError('Combined fleet-size figure requires the same map across studies')
    output_root = Path(output_root)
    figures, results = output_root / 'figures', output_root / 'results'
    figures.mkdir(parents=True, exist_ok=True); results.mkdir(parents=True, exist_ok=True)
    plot_data = make_figure(studies, figures)
    (results / 'search_results.tex').write_text(make_section(studies))
    provenance = {'schema_version': 1, 'accounting': 'native_time_only', 'builder_sha256': digest(Path(__file__)),
                  'studies': [dict(name=s['name'], role=s['role'], seeds=s['seeds'],
                                   manifest_sha256=digest(s['path'] / 'manifest.json'),
                                   results_sha256=digest(s['path'] / 'results.json')) for s in studies],
                  'plot_data': plot_data}
    (results / 'search_results_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return results / 'search_results.tex'


def self_test():
    """Temporary synthetic fixtures only; no production results are invented."""
    with tempfile.TemporaryDirectory(prefix='search-results-check-') as name:
        root = Path(name)
        (root / 'prototype.py').write_text('# fixture\n')
        source = root / 'simulator/server/src/fixture.cpp'; source.parent.mkdir(parents=True)
        source.write_text('// fixture\n')
        for label, counts, seeds, methods in [('pilot', [81], [8399], ['none', 'reactive', 'search_h5', 'search_h10']),
                ('evaluation', [27, 54, 81], [8401, 8402], ['none', 'search_h10_hold6']),
                ('followup', [54], [8403], ['none', 'reactive', 'search_h10_hold6'])]:
            study = root / label; study.mkdir()
            specs, results = [], []
            for count in counts:
                for seed in seeds:
                    for method in methods:
                        spec = dict(run_id=f'{count}_{seed}_{method}', num_agents=count, workload_seed=seed,
                                    method=method, horizon_s=240, trigger='outer_overlap', image_id='fixture')
                        folder = study / spec['run_id']; folder.mkdir()
                        for filename in ('simulator.log', 'scenario.json', 'tasks.json', 'map.json'):
                            (folder / filename).write_text('{}\n')
                        queries, episode_origin = [], None
                        for clock in range(0, 2400, 20):
                            active = clock >= 120 and method != 'none'
                            chosen = active and clock % 200 == 0
                            detail = {}
                            if method.startswith('search_') and active:
                                gain = 1. if chosen else 0.
                                detail = dict(candidate_count=2, best_gain_s=gain, required_gain_s=.5,
                                    candidates=[dict(name='noop', held_robot_ids=[], gain_over_noop_s=0.),
                                                dict(name='single_north', held_robot_ids=[0], gain_over_noop_s=gain)])
                                if method == 'search_h10_hold6':
                                    continuing = episode_origin is not None and clock < episode_origin + 60
                                    releasing = (episode_origin is not None and not continuing
                                                 and clock < episode_origin + 80)
                                    if continuing or releasing:
                                        chosen = continuing
                                        detail = dict(candidate_count=0, candidates=[], best_gain_s=None, selected_gain_s=None,
                                            new_hold_started=False, continuing_hold=continuing,
                                            active_until_clock=episode_origin + 60,
                                            forced_release_until_clock=episode_origin + 80,
                                            selection_origin_clock=episode_origin,
                                            selection_reason='continuing_selected_hold' if continuing else 'forced_release_after_hold')
                                    else:
                                        episode_origin = clock if chosen else None
                                        detail.update(new_hold_started=chosen, continuing_hold=False,
                                            active_until_clock=clock + 60 if chosen else None,
                                            forced_release_until_clock=clock + 80 if chosen else None,
                                            selection_origin_clock=episode_origin)
                            queries.append(dict(clock=clock, reason='decision' if active else 'warmup' if clock < 120 else 'no_gate',
                                                held_robot_ids=[0] if chosen else [], diagnostics=detail,
                                                host_query_s=.1 if detail.get('candidate_count', 1) else .001))
                        (folder / 'host_queries.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in queries))
                        (folder / 'run_spec.json').write_text(json.dumps(spec))
                        delivered = count // 2 + (1 if method.startswith('search_') else 0)
                        tasks = [dict(delivery_completed_s=100. if i < delivered else None,
                                      restricted_flow_s=100. if i < delivered else 240.) for i in range(count)]
                        row = {**spec, 'deliveries_completed': delivered, 'task_outcomes': tasks,
                               'restricted_mean_flow_s': float(np.mean([t['restricted_flow_s'] for t in tasks])),
                               'primary_planner_failures': 2, 'planning_calls': 120,
                               'intervention_decisions': sum(bool(q['held_robot_ids']) for q in queries),
                               'host_decision_p95_s': 0. if method == 'none' else .1,
                               'source_sha256': {filename: digest(folder / filename) for filename in
                                   ('simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'run_spec.json')}}
                        (folder / 'search_metrics.json').write_text(json.dumps(row))
                        specs.append(spec); results.append(row)
            manifest = dict(status='complete', accounting='native_time_only', image_id='fixture', runs=specs,
                            source_sha256={'prototype.py': digest(root / 'prototype.py')},
                            compiled_server_sha256={'src/fixture.cpp': digest(source)},
                            policy=dict(refresh_s=2., warmup_s=12., oracle=False, future_replanning_modeled=False,
                                        absolute_gain_s=.5, relative_gain_fraction=0.))
            (study / 'manifest.json').write_text(json.dumps(manifest))
            (study / 'results.json').write_text(json.dumps(results))
        destination = build([root / 'pilot', root / 'evaluation', root / 'followup'], [root / 'evaluation'],
                            root / 'output', root, followup_paths=[root / 'followup'])
        content = destination.read_text()
        assert 'Fresh-workload evaluation' in content and 'Charged' not in content
        assert 'adaptive follow-up' in content
        parsed = load_study(root / 'pilot', workspace=root)
        assert all(row['review_opportunities'] == 114 for row in parsed['rows'])
        evaluated = load_study(root / 'evaluation', role='evaluation', workspace=root)
        long_rows = [row for row in evaluated['rows'] if row['method'] == 'search_h10_hold6']
        assert all(row['continuing_hold_reviews'] == 2 * row['fresh_hold_selections'] for row in long_rows)
        assert all(row['intervention_reviews'] == 3 * row['fresh_hold_selections'] for row in long_rows)
        assert all(row['actual_query_count'] < row['decision_reviews'] for row in long_rows)
        bad_queries = [dict(clock=clock, reason='warmup' if clock < 120 else 'decision', held_robot_ids=[],
                            host_query_s=0., diagnostics={'candidate_count': 0, 'candidates': []})
                       for clock in range(0, 2400, 20)]
        try:
            query_summary(bad_queries, 240, 2, 12, 'search_h10_hold6')
        except ValueError:
            pass
        else:
            raise AssertionError('Unanchored zero-candidate long-hold review was accepted')
        if shutil.which('pdflatex'):
            from latex_support import PREAMBLE
            preamble = PREAMBLE
            (root / 'output/check.tex').write_text(preamble + '\\begin{document}\n' + content + '\\end{document}\n')
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'check.tex'],
                                    cwd=root / 'output', capture_output=True, text=True)
            if result.returncode or 'Overfull \\hbox' in result.stdout:
                raise AssertionError(result.stdout[-5000:])
        incomplete = read(root / 'pilot/manifest.json'); incomplete['status'] = 'running'
        (root / 'pilot/manifest.json').write_text(json.dumps(incomplete))
        try:
            load_study(root / 'pilot', workspace=root)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete study was accepted')
    print('Search-results temporary rendering, LaTeX, pairing, denominator and incomplete-study checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', type=Path, nargs='+')
    parser.add_argument('--evaluation-studies', type=Path, nargs='*', default=[])
    parser.add_argument('--followup-studies', type=Path, nargs='*', default=[])
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not args.studies:
        parser.error('--studies is required unless using --self-test')
    else:
        print(build(args.studies, args.evaluation_studies, args.output_root, followup_paths=args.followup_studies))


if __name__ == '__main__':
    main()
