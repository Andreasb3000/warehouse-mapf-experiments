#!/usr/bin/env python3
"""Verify a completed safety study, optionally with isolated controls.

Uses recorded summaries/caches and their source hashes; does not reparse large
fleet pose logs or independently reimplement the slow-motion detector. Optional
isolated controls are checked directly against their recorded safety counters.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
WORKSPACE = REPORT.parent
sys.path.insert(0, str(REPORT/'code'))
from fleet_execution import write_json
from analyze import aggregate
from trigger_diagnostics import analyze_triggers

CONDITIONS = ('disabled', 'body_in_outer', 'outer_overlap')
HASH_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message, tolerance=1e-7):
    require(abs(float(a)-float(b)) <= tolerance, message)


def normalized_scenario(scenario):
    result = deepcopy(scenario)
    result['safety'].pop('robot_trigger', None)
    for agent in result['agents']:
        agent['safety'].pop('robot_trigger', None)
        agent['safety'].pop('enabled', None)
    return result


def verify_manifest(root, count=None):
    manifest = read(root/'manifest.json')
    require(manifest['status'] == 'complete', f'{root.name} is not complete')
    if count is None:
        count = len(manifest['runs'])
    require(count > 0 and len(manifest['runs']) == count, f'{root.name}: wrong expected run count')
    ids = {s['run_id'] for s in manifest['runs']}
    require(len(ids) == count, f'{root.name}: duplicate run IDs')
    actual = {p.parent.name for p in root.glob('*/route_metrics.json')}
    require(actual == ids, f'{root.name}: completed runs differ from manifest')
    sources = {**manifest['source_sha256'],
               **{str(Path('simulator')/p): h for p, h in manifest['compiled_source_sha256'].items()}}
    for relative, expected in sources.items():
        require(sha(root/'source_snapshot'/relative) == expected,
                f'{root.name}: archived source mismatch: {relative}')
        require(sha(WORKSPACE/relative) == expected,
                f'{root.name}: current execution source differs: {relative}')
    return manifest


def verify_run(directory, spec):
    require(read(directory/'run_spec.json') == spec, f'{spec["run_id"]}: saved spec mismatch')
    metrics = read(directory/'route_metrics.json')
    for key, value in spec.items():
        require(metrics[key] == value, f'{spec["run_id"]}: metric/spec mismatch: {key}')
    for relative, expected in {**metrics['source_sha256'], **metrics['bridge_sha256']}.items():
        require(sha(directory/relative) == expected, f'{spec["run_id"]}: recording hash mismatch: {relative}')
    execution = read(directory/'execution_summary.json')
    for relative, expected in execution['source_sha256'].items():
        require(sha(directory/relative) == expected, f'{spec["run_id"]}: execution input hash mismatch: {relative}')
    require(metrics['method'] == 'none' and metrics['selected_modifier_reviews'] == 0,
            f'{spec["run_id"]}: unexpected intervention')
    horizon = spec['horizon_s']
    require(metrics['planning_calls'] == horizon / 2 and execution['native_frames'] == horizon * 10,
            f'{spec["run_id"]}: incomplete duration/planning')
    require(execution['num_agents'] == spec['num_agents'], f'{spec["run_id"]}: execution fleet mismatch')
    scenario, tasks, config = (read(directory/name) for name in ('scenario.json', 'tasks.json', 'config.json'))
    n = spec['num_agents']
    require(len(scenario['agents']) == len(tasks['tasks']) == config['parameters']['num_agents'] == n,
            f'{spec["run_id"]}: inconsistent input fleet counts')
    require([a['id'] for a in scenario['agents']] == list(range(n)) and
            [t['robot_id'] for t in tasks['tasks']] == list(range(n)),
            f'{spec["run_id"]}: noncanonical input IDs')
    require(scenario['safety']['enabled'] is True, f'{spec["run_id"]}: global instrumentation disabled')
    for agent in scenario['agents']:
        effective = {**scenario['safety'], **agent['safety']}
        require(effective['enabled'] == (spec['safety_condition'] != 'disabled') and
                effective['robot_trigger'] == spec['trigger'] and effective['inner_stop_enabled'] is False,
                f'{spec["run_id"]}: effective treatment mismatch')
    outcomes = metrics['task_outcomes']
    require([r['robot_id'] for r in outcomes] == list(range(n)), f'{spec["run_id"]}: outcome ID mismatch')
    capped = [horizon if r['delivery_completed_s'] is None else r['delivery_completed_s'] for r in outcomes]
    require(all(0 <= t <= horizon for t in capped), f'{spec["run_id"]}: invalid completion time')
    completed = sum(r['delivery_completed_s'] is not None for r in outcomes)
    require(metrics['deliveries_completed'] == execution['deliveries_completed'] == completed,
            f'{spec["run_id"]}: completion count mismatch')
    close(np.mean(capped), metrics['restricted_mean_flow_s'], f'{spec["run_id"]}: capped-time mean mismatch')
    return metrics, scenario, tasks, config, capped


def verify_disabled(diagnostics, run_id):
    require(all(v == 0 for v in diagnostics['native_whole_recording']['ticks'].values()),
            f'{run_id}: disabled response has nonzero native safety counters')
    require(diagnostics['selected_exposure']['any_selected_trigger_s'] == 0,
            f'{run_id}: disabled response has selected safety exposure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, help='Completed fleet study, already processed by analyze.py')
    parser.add_argument('--solo-study', help='Optional completed nine-run study from run_solo.py; requires seed 8401 / fleet 81 / 240 s in the main study')
    args = parser.parse_args()
    for name in (args.study, args.solo_study):
        if name is not None and (Path(name).name != name or name in ('', '.', '..')):
            parser.error('Choose simple study names')
    root = REPORT/'data/safety_capacity'/args.study
    output = root/'analysis'
    output.mkdir(parents=True, exist_ok=True)
    result = {'schema_version': 1, 'status': 'checking', 'checks': [],
              'scope': 'Completed-case identity, matched inputs, nested tasks, frozen source/recording hashes, recorded counters, analysis completeness and summary arithmetic. No native rerun or fresh slow-label derivation.'}
    try:
        manifest = verify_manifest(root)
        count = len(manifest['runs'])
        seeds = {s['workload_seed'] for s in manifest['runs']}
        fleets = {s['num_agents'] for s in manifest['runs']}
        conditions = {s['safety_condition'] for s in manifest['runs']}
        require(fleets <= {27, 54, 81} and conditions <= set(CONDITIONS), 'Unexpected fleet or safety condition')
        require(len({s['horizon_s'] for s in manifest['runs']}) == 1, 'Mixed observation horizons')
        expected = {(s, n, c) for s in seeds for n in fleets for c in conditions}
        actual = {(s['workload_seed'], s['num_agents'], s['safety_condition']) for s in manifest['runs']}
        require(actual == expected and len(actual) == count, 'Main cohort is not a balanced fleet/seed/condition design')
        result['checks'].append(f'{count} fleet runs complete; declared balanced cohort and frozen native sources verified')
        data = read(output/'results.json')
        require(data['study'] == args.study and data['status'] == 'complete' and data['run_count'] == count, 'Study analysis is incomplete')
        rows = {r['run_id']: r for r in data['runs']}
        require(len(data['runs']) == len(rows) == count and set(rows) == {s['run_id'] for s in manifest['runs']},
                'Report analysis has missing, duplicate or foreign runs')
        groups, prefixes, floor, main_inputs = {}, {}, None, {}
        for spec in manifest['runs']:
            directory = root/spec['run_id']
            metrics, scenario, tasks, config, capped = verify_run(directory, spec)
            map_doc = read(directory/'map.json')
            if floor is None: floor = map_doc
            require(map_doc == floor, f'{spec["run_id"]}: map differs across fleets')
            normalized = {'map': map_doc, 'scenario': normalized_scenario(scenario), 'tasks': tasks,
                          'config': config, 'assignment': read(directory/'assignment.json')}
            group_key = (spec['workload_seed'], spec['num_agents'])
            if group_key in groups:
                require(normalized == groups[group_key], f'{group_key}: non-treatment input mismatch')
            groups[group_key] = normalized
            prefix = {'agents': normalized['scenario']['agents'][:27], 'tasks': tasks['tasks'][:27]}
            if spec['workload_seed'] in prefixes:
                require(prefix == prefixes[spec['workload_seed']], f'{spec["run_id"]}: first 27 tasks differ')
            prefixes[spec['workload_seed']] = prefix
            main_inputs[(spec['workload_seed'], spec['num_agents'], spec['safety_condition'])] = (scenario, tasks)
            row = rows[spec['run_id']]
            require(row == read(directory/'safety_analysis.json'), f'{spec["run_id"]}: stale analysis row')
            sources = {name: sha(directory/name) for name in
                       ('route_metrics.json', 'simulator.log', 'scenario.json', 'map.json',
                        'assignment.json', 'config.json', 'tasks.json')}
            sources.update({'analysis.py': sha(HERE/'analyze.py'),
                            'trigger_diagnostics.py': sha(HERE/'trigger_diagnostics.py')})
            require(row['source_sha256'] == sources, f'{spec["run_id"]}: analysis sources differ')
            close(row['prefix27_mean_flow_s'], np.mean(capped[:27]), f'{spec["run_id"]}: common-task mean differs')
            close(row['restricted_mean_flow_s'], metrics['restricted_mean_flow_s'], f'{spec["run_id"]}: report mean differs')
            require(0 <= row['active_slow_robot_s'] <= row['slow_robot_s'] + 1e-7,
                    f'{spec["run_id"]}: active slow masking inconsistent')
            with np.load(directory/'slow_maps.npz') as rasters:
                shape = (len(map_doc['layout']), len(map_doc['layout'][0]))
                require(rasters['active_slow'].shape == shape, f'{spec["run_id"]}: raster shape differs')
                close(rasters['active_slow'].sum()*(spec['horizon_s'] - 2), row['active_slow_robot_s'],
                      f'{spec["run_id"]}: active spatial mass differs', 1e-5)
            if spec['safety_condition'] == 'disabled':
                verify_disabled(row['diagnostics'], spec['run_id'])
        result['checks'].extend(['All non-treatment inputs match within fleet/workload groups',
                                 'Identical map across fleets; original 27 tasks and starts unchanged within each seed',
                                 'Analysis sources/caches match all completed runs; completion and raster totals agree',
                                 'All recorded no-safety-rule runs have zero native safety counters'])
        summary, pairs, interaction = aggregate(data['runs'])
        require(summary == data['summary'] and pairs == data['paired_effects'] and
                interaction == data['common_task_load_effects'], 'Aggregate/paired report values differ')
        solo_count = 0
        if args.solo_study:
            solo_count = verify_solo(args.solo_study, manifest, main_inputs, floor, output)
            result['checks'].append('Nine isolated controls match their selected original tasks; frozen inputs and recorded safety counters checked')
        result['checks'].extend(['All paired and common-task aggregate arithmetic agrees',
                                 'Frozen/current execution sources and per-run recorded/bridge hashes verified'])
        result.update(status='passed', main_runs=count, solo_runs=solo_count, image_id=manifest['image_id'],
                      artifact_sha256={'results.json': sha(output/'results.json'),
                          **{name: sha(HERE/name) for name in ('analyze.py', 'trigger_diagnostics.py', 'verify.py')}})
        if solo_count:
            result['artifact_sha256']['solo_results.json'] = sha(output/'solo_results.json')
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    result['hashed_files'] = len(HASH_CACHE)
    write_json(output/'verification.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'hashed_files', 'main_runs', 'solo_runs', 'error') if key in result}))
    if result['status'] != 'passed':
        raise SystemExit(1)


def verify_solo(study, manifest, main_inputs, floor, output):
    solo_root = REPORT/'data/safety_capacity'/study
    solo_manifest = verify_manifest(solo_root, 9)
    require(manifest['image_id'] == solo_manifest['image_id'], 'Main and solo images differ')
    require(all((8401, 81, c) in main_inputs for c in CONDITIONS),
            'Solo comparison requires all three fleet-81 / seed-8401 conditions in the main study')
    require(all(s['horizon_s'] == 240 for s in manifest['runs']),
            'Solo comparison requires a 240-second main study')
    chosen = {s['original_robot_id']: s for s in solo_manifest['selections']}
    require(len(chosen) == 3 and {s['selection'] for s in chosen.values()} == {'shortest', 'median', 'longest'},
            'Solo selections differ')
    expected_solo = {(i, c) for i in chosen for c in CONDITIONS}
    require({(s['original_robot_id'], s['safety_condition']) for s in solo_manifest['runs']} == expected_solo,
            'Solo treatments are incomplete')
    solo_rows = []
    for spec in solo_manifest['runs']:
        directory = solo_root/spec['run_id']
        metrics, scenario, tasks, config, _ = verify_run(directory, spec)
        require(read(directory/'map.json') == floor, f'{spec["run_id"]}: solo map differs')
        require(spec['num_agents'] == 1 and spec['workload_seed'] == 8401, f'{spec["run_id"]}: solo setup differs')
        original = spec['original_robot_id']
        original_scenario, original_tasks = main_inputs[(8401, 81, spec['safety_condition'])]
        original_agent = original_scenario['agents'][original]
        require(scenario['agents'][0]['start'] == original_agent['start'] and
                scenario['agents'][0]['goals'] == original_agent['goals'] and
                scenario['agents'][0]['safety'] == original_agent['safety'] and
                tasks['tasks'][0]['stages'] == original_tasks['tasks'][original]['stages'],
                f'{spec["run_id"]}: solo task/response differs from original')
        for name, expected_hash in solo_manifest['prepared_input_sha256'][spec['run_id']].items():
            require(sha(directory/name) == sha(solo_root/'prepared_inputs'/spec['run_id']/name) == expected_hash,
                    f'{spec["run_id"]}: frozen solo input differs')
        diagnostics = analyze_triggers(directory)
        solo_rows.append({'spec': spec, 'metrics': metrics, 'diagnostics': diagnostics})
        if spec['safety_condition'] == 'disabled':
            verify_disabled(diagnostics, spec['run_id'])
    write_json(output/'solo_results.json', solo_rows)
    return len(solo_rows)

if __name__ == '__main__':
    main()
