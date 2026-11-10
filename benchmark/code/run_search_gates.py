#!/usr/bin/env python3
"""Sparse counterfactual admission search after native planning.

Every branch uses current accepted actions; no future truth is read. Native
task times exclude computation. Branch predictions are kinematic surrogates,
not native simulator checkpoints and not an oracle of future PBS replanning.
"""
import argparse
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from features import REPORT, REPO, observation_clock
from fleet_cases import make_fleet_workload
from fleet_execution import digest, objects, write_json
from run_decisions import LiveHistory
from run_entry_gates import score as score_admission
from entry_gate_policy import EntryGatePolicy
from search_admission_policy import SearchAdmissionPolicy
from long_hold_search_policy import LongHoldSearchPolicy
from experiments.zones.run import docker_prefix, mounts, environment, flags
from experiments.junction.scenario import load_scenario
from runtime import validate_native_image

METHODS = ('none', 'reactive', 'search_h5', 'search_h10', 'search_h10_hold6')


def score(directory, wall_s):
    result = score_admission(directory, wall_s)
    events = list(objects(directory / 'simulator.log'))
    queries = [json.loads(line) for line in (directory / 'host_queries.jsonl').read_text().splitlines()]
    expected = [q['clock'] for q in queries]
    for kind in ('admission_barrier_complete', 'postdecision_gate_applied'):
        rows = [e for e in events if e['type'] == kind]
        if sorted(e['clock'] for e in rows) != expected:
            raise ValueError(f'Missing or duplicate {kind}')
    applied = {e['clock']: e for e in events if e['type'] == 'postdecision_gate_applied'}
    for query in queries:
        event = applied[query['clock']]
        response = json.loads((directory / 'admission' / f"response_{query['clock']}.json").read_text())
        gate = response.get('entry_gate') or {}
        if (sorted(event['held_robot_ids']) != sorted(query['held_robot_ids']) or
                event['plan_version'] != query['plan_version']):
            raise ValueError('Native gate differs from selected post-planner decision')
        if (event['until_clock'] != gate.get('until_clock', -1) or
                event['protect_clearance'] != gate.get('protect_clearance', True)):
            raise ValueError('Native gate duration or clearance mode differs from response')
    # Retain computational measurements, not computational penalties in outcomes.
    for field in ('charged_restricted_mean_flow_s', 'deliveries_by_equal_charged_horizon',
                  'equal_charged_horizon_mean_flow_s'):
        result.pop(field, None)
    for task in result['task_outcomes']:
        task.pop('charged_restricted_flow_s', None)
        task.pop('synchronous_query_pause_s', None)
    decisions = [q for q in queries if q['reason'] == 'decision']
    admission_times = [e['wall_s'] for e in events if e['type'] == 'admission_barrier_complete']
    result.update(
        accounting='Native simulator time only. Host and barrier times are separate runtime measurements.',
        metric_scope='All original tasks retained; unfinished tasks contribute the native horizon. Gates restrict future dispatch and preserve committed/queued movement. Counterfactual scores are accepted-work proxies, not predicted delivery times.',
        decision_count=len(decisions),
        intervention_decisions=sum(bool(q['held_robot_ids']) for q in decisions),
        no_intervention_decisions=sum(not q['held_robot_ids'] for q in decisions),
        intervention_fraction=(sum(bool(q['held_robot_ids']) for q in decisions) / len(decisions)) if decisions else 0.,
        search_candidates_evaluated=sum(q['diagnostics'].get('candidate_count', 0) for q in decisions),
        postplan_admission_barrier_total_s=sum(admission_times),
        postplan_admission_barrier_p95_s=float(np.quantile(admission_times, .95)),
        runtime_accounting='The admission barrier is a component of the full client routing barrier; do not add them. Neither is charged to simulated task times.',
        branch_model='Joint kinematic safety-aware rollout with action admission; accepted plans fixed after origin; no future PBS calls.',
    )
    # Replace the intermediate scorer output so this study exposes one accounting.
    write_json(directory / 'gate_metrics.json', result)
    write_json(directory / 'search_metrics.json', result)
    return result


def run_case(directory, spec, docker_host, timeout_s=3600):
    directory = Path(directory)
    if (directory / 'search_metrics.json').exists():
        if json.loads((directory / 'run_spec.json').read_text()) != spec:
            raise ValueError('Existing run specification differs')
        return json.loads((directory / 'search_metrics.json').read_text())
    if directory.exists():
        raise ValueError(f'Incomplete recording exists; use a fresh study: {directory}')
    directory.mkdir(parents=True)
    map_doc, scenario, config, tasks, assignment = make_fleet_workload(
        spec['workload_seed'], duration_s=spec['horizon_s'], fleet_size=spec['num_agents'], trigger='outer_overlap')
    config.update(image=spec['image_id'], wall_timeout_s=timeout_s)
    config['parameters'].update(sim_window_tick=20, single_agent_solver='ASTAR')
    for name, value in [('map.json', map_doc), ('scenario.json', scenario), ('config.json', config),
                        ('tasks.json', tasks), ('assignment.json', assignment), ('run_spec.json', spec)]:
        write_json(directory / name, value)
    load_scenario(directory / 'scenario.json', map_doc['layout'], spec['num_agents'])
    bridge, admission = directory / 'bridge', directory / 'admission'
    bridge.mkdir(); admission.mkdir()
    docker = docker_prefix(docker_host)
    container = 'admission-search-' + uuid.uuid4().hex[:10]
    params = dict(config['parameters'], container=True, visualizer='web', save_stats=True,
                  scenario_file='/inputs/scenario.json', stats_name='/results/stats.json',
                  argos_config_filepath='/results/simulation.argos', output_log='/results/simulator.log')
    command = docker + ['run', '--rm', '--init', '--name', container, '--platform', 'linux/amd64',
                        '--ulimit', 'nofile=65536:65536', *mounts(directory), *environment(config),
                        '-e', 'LSMART_ROUTING_DIR=/bridge', '-e', 'LSMART_ROUTING_INTERVAL_TICKS=20',
                        '-e', 'LSMART_ADMISSION_DIR=/admission', '-e', 'LSMART_ADMISSION_TIMEOUT_SECONDS=120',
                        '-v', f'{bridge}:/bridge', '-v', f'{admission}:/admission', '-v', f'{directory}:/results',
                        spec['image_id'], 'python3', 'run_lifelong.py', '/inputs/map.json', *flags(params)]
    write_json(directory / 'command.json', command)
    method = spec['method']
    policy = (LongHoldSearchPolicy(absolute_gain_s=spec['absolute_gain_s'],
                relative_gain_fraction=spec['relative_gain_fraction']) if method == 'search_h10_hold6' else
              SearchAdmissionPolicy(horizon_s=5. if method == 'search_h5' else 10.,
                absolute_gain_s=spec['absolute_gain_s'], relative_gain_fraction=spec['relative_gain_fraction'])
              if method.startswith('search_') else EntryGatePolicy())
    history = LiveHistory(directory / 'simulator.log')
    penalties = [0.] * (len(map_doc['layout']) * len(map_doc['layout'][0]))
    clock = 0; phase = 'pre'; started = time.perf_counter()
    print('START', spec['run_id'], flush=True)
    with (directory / 'launcher.log').open('w') as launcher, threadpool_limits(limits=1):
        process = subprocess.Popen(command, stdout=launcher, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.perf_counter() - started > timeout_s:
                    raise TimeoutError(spec['run_id'])
                source = bridge if phase == 'pre' else admission
                path = source / f'request_{clock}.json'
                if not path.exists():
                    time.sleep(.002); continue
                event = json.loads(path.read_text())
                if event['clock'] != clock:
                    raise ValueError('Handshake clock mismatch')
                observation_clock(event)
                if phase == 'pre':
                    # Routes are produced before choosing a timing intervention.
                    write_json(bridge / f'response_{clock}.json', {'clock': clock, 'cell_penalties': penalties})
                    phase = 'post'; continue
                if event['boundary'] != 'post_planner_before_dispatch':
                    raise ValueError('Search must use freshly accepted native plans')
                if any(c['poll_interval_ticks'] != 10 for c in event['controllers']):
                    raise ValueError('Accounting requires the native ten-tick action poll')
                began = time.perf_counter(); history.update()
                response = {'clock': clock, 'plan_version': event['plan_version']}
                held, diagnostics = [], {}
                if clock < 120 or history.goals >= 2 * spec['num_agents'] or method == 'none':
                    reason = 'warmup' if clock < 120 else ('all_completed' if history.goals >= 2 * spec['num_agents'] else 'no_gate')
                else:
                    reason = 'decision'
                    gate, diagnostics = policy.decide(event, history.past(event), scenario, map_doc,
                                                      method=method if method.startswith('search_') else 'reactive')
                    response['entry_gate'] = gate; held = gate['held_robot_ids']
                row = {'clock': clock, 'observation_clock': event['observation_clock'], 'origin_s': clock / 10,
                       'plan_version': event['plan_version'], 'planner_source': event['planner_source'],
                       'reason': reason, 'held_robot_ids': held, 'diagnostics': diagnostics,
                       'host_query_s': time.perf_counter() - began}
                with (directory / 'host_queries.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                write_json(admission / f'response_{clock}.json', response)
                clock += 20; phase = 'pre'
            if process.returncode:
                raise RuntimeError(f'Native search run exited {process.returncode}: {directory}')
        except BaseException:
            subprocess.run(docker + ['stop', '--time', '2', container], capture_output=True, timeout=20)
            process.wait(timeout=30)
            raise
    result = score(directory, time.perf_counter() - started)
    print('DONE', spec['run_id'], f"delivered={result['deliveries_completed']}/{spec['num_agents']} flow={result['restricted_mean_flow_s']:.2f}s interventions={result['intervention_decisions']}/{result['decision_count']} wall={result['recording_wall_s']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--docker-host', default=None)
    parser.add_argument('--fleets', type=int, nargs='+', choices=[27, 54, 81], default=[81])
    parser.add_argument('--seeds', type=int, nargs='+', default=[8399])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--horizon-s', type=int, default=240)
    parser.add_argument('--absolute-gain-s', type=float, default=.5)
    parser.add_argument('--relative-gain-fraction', type=float, default=0.)
    args = parser.parse_args()
    if Path(args.study).name != args.study or args.study in ('', '.', '..'):
        parser.error('Choose a fresh simple study name')
    if args.horizon_s < 20 or args.horizon_s % 2:
        parser.error('Duration must be even and at least 20 s')
    if any(not np.isfinite(x) or x < 0 for x in (args.absolute_gain_s, args.relative_gain_fraction)):
        parser.error('Improvement margins must be finite and nonnegative')
    if any(len(values) != len(set(values)) for values in (args.fleets, args.seeds, args.methods)):
        parser.error('Fleets, seeds and methods must not contain duplicates')
    docker = docker_prefix(args.docker_host)
    image = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    validate_native_image(docker, image)
    compiled = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64', image,
                          'cat', '/usr/project/admission-source-sha256.json'], text=True))
    for relative, recorded in compiled.items():
        if digest(REPO / 'server' / relative) != recorded:
            raise ValueError(f'Compiled source mismatch: server/{relative}')
    paths = [Path(__file__), REPORT/'code/search_admission_policy.py', REPORT/'code/entry_gate_policy.py',
             REPORT/'code/long_hold_search_policy.py',
             REPORT/'code/run_entry_gates.py', REPORT/'code/features.py', REPORT/'code/fleet_cases.py',
             REPORT/'code/fleet_execution.py', REPORT/'code/run_decisions.py', REPORT/'code/decision_metrics.py',
             REPO/'experiments/prediction/joint.py', REPO/'experiments/prediction/admission.py', REPO/'run_lifelong.py',
             REPORT/'native/build.py', REPORT/'native/Dockerfile',
             REPORT/'native/build_proof.json', REPORT/'native/native_manifest.py']
    hashes = {str(p.relative_to(REPORT.parent)): digest(p) for p in paths}
    specs = [dict(run_id=f'w{s}_fleet{n}_{m}', workload_seed=s, num_agents=n, method=m,
                  trigger='outer_overlap', horizon_s=args.horizon_s, image_id=image,
                  absolute_gain_s=args.absolute_gain_s, relative_gain_fraction=args.relative_gain_fraction)
             for s in args.seeds for n in args.fleets for m in args.methods]
    random.Random(20261102).shuffle(specs)
    manifest = {'schema_version': 1, 'status': 'prepared', 'accounting': 'native_time_only',
                'source_sha256': hashes, 'compiled_server_sha256': compiled, 'image_id': image,
                'host': {'platform': platform.platform(), 'machine': platform.machine(),
                         'python': platform.python_version(), 'threads': 1}, 'runs': specs,
                'policy': {'version': 'postplan_sparse_search_v1', 'refresh_s': 2., 'warmup_s': 12.,
                           'absolute_gain_s': args.absolute_gain_s, 'relative_gain_fraction': args.relative_gain_fraction,
                           'candidate_scope': 'No action plus nearest single and all eligible robots per approach; at most nine branches.',
                           'objective': 'Time-average of total fleet accepted-work progress; proxy, not delivery-time oracle.',
                           'hold_duration_s': {'search_h5': 2., 'search_h10': 2., 'search_h10_hold6': 6.},
                           'long_hold_execution': 'Retain selected IDs for three two-second native responses without searching again, then release for two seconds. Native commitment guards remain authoritative.',
                           'oracle': False, 'future_replanning_modeled': False},
                'scope': 'Exploratory sparse admission search; do nothing by default; computation measured separately.'}
    root = REPORT / 'data/admission_search' / args.study
    path = root / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('source_sha256', 'compiled_server_sha256', 'image_id', 'runs', 'policy'):
            if old[key] != manifest[key]:
                raise ValueError(f'Resume mismatch: {key}')
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'source_snapshot'
    for relative, expected in hashes.items():
        target = archive / relative
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f'Archived source mismatch: {relative}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPORT.parent / relative, target)
    for relative, expected in compiled.items():
        target = archive / 'simulator/server' / relative
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f'Archived native source mismatch: {relative}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / 'server' / relative, target)
    manifest['status'] = 'running'; write_json(path, manifest)
    try:
        results = [run_case(root / spec['run_id'], spec, args.docker_host) for spec in specs]
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc)); write_json(path, manifest); raise
    write_json(root / 'results.json', results)
    manifest['status'] = 'complete'; write_json(path, manifest)
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
