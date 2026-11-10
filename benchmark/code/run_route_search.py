#!/usr/bin/env python3
"""Compare native route-cost candidates without issuing admission holds.

The native driver obtains one MAPF instance, previews successful candidates,
and installs exactly the selected plan. Outcomes use native simulation time;
candidate planning and host prediction costs are measured separately.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from features import REPORT, REPO, observation_clock, pose_array
from fleet_cases import make_fleet_workload
from fleet_execution import digest, objects, summarize, write_json
from decision_metrics import task_completions
from run_decisions import LiveHistory
from route_search_policy import RouteSearchPolicy
from experiments.baseline.analyze import _objects
from experiments.zones.run import docker_prefix, mounts, environment, flags
from experiments.junction.scenario import load_scenario

METHODS = ('none', 'nominal', 'rollout')


def build_proof_artifacts(path, image_id, compiled):
    """Validate the selected image's proof and archive its actual build inputs."""
    path = Path(path).resolve()
    proof = json.loads(path.read_text())
    if proof.get('image_id') != image_id:
        raise ValueError('Build proof does not match the selected image')
    if proof.get('compiled_source_sha256') != compiled:
        raise ValueError('Build proof does not match the image source manifest')
    artifacts = proof.get('build_artifacts_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Build proof has no recorded build artifacts')
    paths = {path}
    for name, expected in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Build artifact names must be local filenames')
        artifact = path.parent / name
        if digest(artifact) != expected:
            raise ValueError(f'Build artifact differs from the selected proof: {name}')
        paths.add(artifact)
    return sorted(paths)


class TaskHistory(LiveHistory):
    """Causal poses plus physically acknowledged original task stages."""
    def __init__(self, path, tasks):
        super().__init__(path)
        self.tasks = {t['robot_id']: t for t in tasks}
        self.stages = {i: 0 for i in self.tasks}
        self.completed_robot_ids = set()
        self.last_ack_clock = -1

    def update(self):
        if not self.path.exists():
            return
        with self.path.open() as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        lines = (self.fragment + chunk).split('\n')
        self.fragment = lines.pop()
        for event in _objects('\n'.join(lines)):
            if event.get('type') == 'tick':
                self.frames[event['clock']] = pose_array(event)
            elif event.get('type') == 'goal_reached':
                rid = event['agent']
                if rid not in self.tasks or self.stages[rid] >= len(self.tasks[rid]['stages']):
                    raise ValueError('Unexpected or duplicate native goal acknowledgement')
                task = self.tasks[rid]
                stage = task['stages'][self.stages[rid]]
                if math.dist((event['row'], event['col']), stage['location']) > .02:
                    raise ValueError('Native acknowledgement is not the next original task stage')
                if event['clock'] / 10 < task.get('release_at_s', 0):
                    raise ValueError('Native task completed before release')
                self.last_ack_clock = max(self.last_ack_clock, event['clock'])
                self.stages[rid] += 1
                self.goals += 1
                if self.stages[rid] == len(task['stages']):
                    self.completed_robot_ids.add(rid)
        if self.frames:
            cutoff = max(self.frames) - 140
            self.frames = {k: v for k, v in self.frames.items() if k >= cutoff}


def geometry(candidate):
    """Raw planner position/task sequence, excluding planned timestamps."""
    payload = candidate['plan']
    paths = payload['plan'] if isinstance(payload, dict) else payload
    result = []
    for path in paths:
        compressed = []
        for step in path:
            point = (step[0], step[1], step[3])
            if not compressed or point != compressed[-1]:
                compressed.append(point)
        result.append(compressed)
    return result


def score(directory, wall_s):
    spec = json.loads((directory / 'run_spec.json').read_text())
    native = summarize(directory, wall_s)
    events = list(objects(directory / 'simulator.log'))
    queries = [json.loads(line) for line in (directory / 'host_queries.jsonl').read_text().splitlines()]
    expected = list(range(0, spec['horizon_s'] * 10, 20))
    if [q['clock'] for q in queries] != expected:
        raise ValueError('Missing or duplicate host route decisions')
    by_kind = {}
    for kind in ('routing_plan', 'routing_barrier_complete', 'plan_snapshot', 'route_search_accepted'):
        rows = [e for e in events if e['type'] == kind]
        if len(rows) != len(expected) or sorted(e['clock'] for e in rows) != expected:
            raise ValueError(f'Missing or duplicate {kind}')
        by_kind[kind] = {e['clock']: e for e in rows}
    if any(e['type'] in ('entry_gate_blocked', 'postdecision_gate_applied') for e in events):
        raise ValueError('Route-only study unexpectedly used admission gating')
    candidate_calls, candidate_failures, alternate_routes = 0, 0, 0
    for q in queries:
        clock = q['clock']
        batch = json.loads((directory / 'plan_search' / f'request_{clock}.json').read_text())
        response = json.loads((directory / 'plan_search' / f'response_{clock}.json').read_text())
        if response != {'clock': clock, 'selected_index': q['selected_index']}:
            raise ValueError('Saved route response differs from selected decision')
        candidates = batch['candidates']
        selected = candidates[q['selected_index']]
        if selected['index'] != q['selected_index']:
            raise ValueError('Native candidate indexing differs')
        if not selected['success'] and q['selected_index'] != 0:
            raise ValueError('Failed alternative was selected')
        actual = by_kind['plan_snapshot'][clock]
        plan = by_kind['routing_plan'][clock]
        acceptance = by_kind['route_search_accepted'][clock]
        if (acceptance['selected_index'] != q['selected_index'] or
                acceptance['selected_name'] != selected['name'] or
                acceptance['plan_version'] != actual['plan_version']):
            raise ValueError('Native acceptance differs from the selected route candidate')
        if bool(plan['success']) != bool(selected['success']):
            raise ValueError('Selected success and native planner outcome differ')
        if actual['planner_source'] != ('primary' if selected['success'] else 'fallback'):
            raise ValueError('Selected candidate and accepted planner source differ')
        if selected['success']:
            preview = selected['preview_snapshot']
            if actual['robots'] != preview['robots'] or actual['plan_version'] != preview['plan_version']:
                raise ValueError('Installed native action graph differs from evaluated preview')
            if not acceptance['preview_verified']:
                raise ValueError('Native server did not verify the installed route preview')
        if plan['nonzero_penalty_cells'] != selected['nonzero_penalty_cells']:
            raise ValueError('Selected cost map differs from executed planner metadata')
        for candidate in candidates:
            candidate_calls += bool(candidate['evaluated'])
            candidate_failures += bool(candidate['evaluated'] and not candidate['success'])
        if q['selected_index'] and geometry(selected) != geometry(candidates[0]):
            alternate_routes += 1
    tasks = json.loads((directory / 'tasks.json').read_text())['tasks']
    outcomes = task_completions(events, tasks, 10, spec['horizon_s'])
    if sum(r['delivery_completed_s'] is not None for r in outcomes) != native['deliveries_completed']:
        raise ValueError('Task ledger and native delivery count differ')
    active = [q for q in queries if q['reason'] == 'decision']
    host_times = [q['host_query_s'] for q in active]
    reviews = [q for q in queries if q['clock'] >= 120]
    barriers = [by_kind['routing_barrier_complete'][c]['wall_s'] for c in expected]
    result = {
        **spec,
        'accounting': 'native_time_only',
        'deliveries_completed': native['deliveries_completed'],
        'deliveries_unfinished': spec['num_agents'] - native['deliveries_completed'],
        'restricted_mean_flow_s': float(np.mean([r['restricted_flow_s'] for r in outcomes])),
        'completion_fraction': native['deliveries_completed'] / spec['num_agents'],
        'planning_calls': len(expected), 'primary_planner_failures': native['native_plan_failures'],
        'candidate_planning_calls': candidate_calls, 'candidate_planning_failures': candidate_failures,
        'scheduled_reviews_after_warmup': len(reviews), 'active_search_reviews': len(active),
        'selected_modifier_reviews': sum(q['selected_index'] != 0 for q in reviews),
        'changed_route_geometry_reviews': alternate_routes,
        'baseline_failed_reviews': sum(q['baseline_failed'] for q in reviews),
        'host_query_p50_s': float(np.median(host_times)) if host_times else 0.,
        'host_query_p95_s': float(np.quantile(host_times, .95)) if host_times else 0.,
        'total_barrier_s': sum(barriers), 'barrier_p95_s': float(np.quantile(barriers, .95)),
        'recording_wall_s': wall_s, 'task_outcomes': outcomes,
        'metric_scope': 'All original tasks retained; unfinished tasks contribute the native horizon. No computation charged. Modifier selections are distinct from changed route geometry. Failed zero-cost solves use normal native fallback and skip alternative selection.',
        'source_sha256': {name: digest(directory / name) for name in
                          ('simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'map.json', 'run_spec.json')},
        'bridge_sha256': {str(p.relative_to(directory)): digest(p)
                          for folder in ('bridge', 'plan_search')
                          for p in sorted((directory / folder).glob('*.json'))},
    }
    write_json(directory / 'route_metrics.json', result)
    return result


def run_case(directory, spec, docker_host, timeout_s=5400, *, inputs=None):
    if (directory / 'route_metrics.json').exists():
        if json.loads((directory / 'run_spec.json').read_text()) != spec:
            raise ValueError('Completed recording has a different specification')
        return json.loads((directory / 'route_metrics.json').read_text())
    if directory.exists():
        raise ValueError(f'Preserve incomplete run and use a fresh study: {directory}')
    directory.mkdir(parents=True)
    map_doc, scenario, config, tasks, assignment = deepcopy(inputs) if inputs is not None else make_fleet_workload(
        spec['workload_seed'], duration_s=spec['horizon_s'], fleet_size=spec['num_agents'], trigger='outer_overlap')
    config.update(image=spec['image_id'], wall_timeout_s=timeout_s)
    config['parameters'].update(sim_window_tick=20, single_agent_solver='ASTAR')
    for name, value in [('map.json', map_doc), ('scenario.json', scenario), ('config.json', config),
                        ('tasks.json', tasks), ('assignment.json', assignment), ('run_spec.json', spec)]:
        write_json(directory / name, value)
    load_scenario(directory / 'scenario.json', map_doc['layout'], spec['num_agents'])
    bridge, batch_dir = directory / 'bridge', directory / 'plan_search'
    bridge.mkdir(); batch_dir.mkdir()
    docker = docker_prefix(docker_host)
    container = 'route-search-' + uuid.uuid4().hex[:10]
    params = dict(config['parameters'], container=True, visualizer='web', save_stats=True,
                  scenario_file='/inputs/scenario.json', stats_name='/results/stats.json',
                  argos_config_filepath='/results/simulation.argos', output_log='/results/simulator.log')
    command = docker + ['run', '--rm', '--init', '--name', container, '--platform', 'linux/amd64',
                        '--ulimit', 'nofile=65536:65536', *mounts(directory), *environment(config),
                        '-e', 'LSMART_ROUTING_DIR=/bridge', '-e', 'LSMART_ROUTING_INTERVAL_TICKS=20',
                        '-e', 'LSMART_PLAN_SEARCH_DIR=/plan_search', '-e', 'LSMART_PLAN_SEARCH_TIMEOUT_SECONDS=120',
                        '-v', f'{bridge}:/bridge', '-v', f'{batch_dir}:/plan_search', '-v', f'{directory}:/results',
                        spec['image_id'], 'python3', 'run_lifelong.py', '/inputs/map.json', *flags(params)]
    write_json(directory / 'command.json', command)
    history = TaskHistory(directory / 'simulator.log', tasks['tasks'])
    policy = RouteSearchPolicy(minimum_mean_eta_gain_s=spec['minimum_mean_eta_gain_s'])
    clock, phase = 0, 'propose'
    started = time.perf_counter()
    proposal_diag = {}; proposal_s = 0.; original = None; reason = None
    print('START', spec['run_id'], flush=True)
    with (directory / 'launcher.log').open('w') as launcher, threadpool_limits(limits=1):
        process = subprocess.Popen(command, stdout=launcher, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.perf_counter() - started > timeout_s:
                    raise TimeoutError(spec['run_id'])
                path = (bridge if phase == 'propose' else batch_dir) / f'request_{clock}.json'
                if not path.exists():
                    time.sleep(.002); continue
                event = json.loads(path.read_text())
                if event['clock'] != clock:
                    raise ValueError('Route-search handshake clock mismatch')
                observation_clock(event)
                began = time.perf_counter(); history.update()
                if history.last_ack_clock >= clock:
                    raise ValueError('Task ledger contains an acknowledgement after this frozen origin')
                if phase == 'propose':
                    original = event
                    reason = ('warmup' if clock < 120 else 'all_completed'
                              if len(history.completed_robot_ids) == spec['num_agents'] else 'no_modifier'
                              if spec['method'] == 'none' else 'decision')
                    method = spec['method'] if reason == 'decision' else 'none'
                    past = history.past(event) if reason == 'decision' else None
                    payload, proposal_diag = policy.propose(event, past, scenario, map_doc, method)
                    payload['clock'] = clock
                    proposal_s = time.perf_counter() - began
                    write_json(bridge / f'response_{clock}.json', payload)
                    phase = 'select'; continue
                if event['boundary'] != 'post_commit_before_plan_selection':
                    raise ValueError('Wrong native route selection boundary')
                if event['pose_frame'] != original['pose_frame'] or event['controllers'] != original['controllers']:
                    raise ValueError('Physics or controller state changed during candidate planning')
                method = spec['method'] if reason == 'decision' else 'none'
                response, detail = policy.select(event, scenario, map_doc, method,
                                                 completed_robot_ids=history.completed_robot_ids)
                selected = event['candidates'][response['selected_index']]
                row = {'clock': clock, 'observation_clock': event['observation_clock'], 'origin_s': clock / 10,
                       'reason': reason, 'method': spec['method'], 'selected_index': response['selected_index'],
                       'selected_name': selected['name'], 'baseline_failed': not event['candidates'][0]['success'],
                       'completed_robot_ids': sorted(history.completed_robot_ids),
                       'proposal': proposal_diag, 'selection': detail,
                       'proposal_s': proposal_s, 'selection_s': time.perf_counter() - began,
                       'host_query_s': proposal_s + time.perf_counter() - began}
                with (directory / 'host_queries.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                write_json(batch_dir / f'response_{clock}.json', response)
                clock += 20; phase = 'propose'
            if process.returncode:
                raise RuntimeError(f'Native route search exited {process.returncode}: {directory}')
        except BaseException:
            subprocess.run(docker + ['stop', '--time', '2', container], capture_output=True, timeout=20)
            process.wait(timeout=30)
            raise
    result = score(directory, time.perf_counter() - started)
    print('DONE', spec['run_id'], f"delivered={result['deliveries_completed']}/{spec['num_agents']} flow={result['restricted_mean_flow_s']:.2f}s modifiers={result['selected_modifier_reviews']}/{result['scheduled_reviews_after_warmup']} wall={result['recording_wall_s']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--build-proof', type=Path, default=REPORT/'native/build_proof.json')
    parser.add_argument('--docker-host', default=None)
    parser.add_argument('--fleets', type=int, nargs='+', choices=[27, 54, 81], default=[54])
    parser.add_argument('--seeds', type=int, nargs='+', default=[8401])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--horizon-s', type=int, default=240)
    parser.add_argument('--minimum-mean-eta-gain-s', type=float, default=.5)
    args = parser.parse_args()
    if Path(args.study).name != args.study or args.study in ('', '.', '..'):
        parser.error('Choose a fresh simple study name')
    if args.horizon_s < 20 or args.horizon_s % 2:
        parser.error('Native duration must be even and at least 20 seconds')
    if not math.isfinite(args.minimum_mean_eta_gain_s) or args.minimum_mean_eta_gain_s < 0:
        parser.error('ETA margin must be finite and nonnegative')
    if any(len(v) != len(set(v)) for v in (args.fleets, args.seeds, args.methods)):
        parser.error('Duplicate fleets, seeds or methods')
    docker = docker_prefix(args.docker_host)
    image = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    compiled = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64', image,
                          'cat', '/usr/project/route-search-source-sha256.json'], text=True))
    for relative, expected in compiled.items():
        if digest(REPO / relative) != expected:
            raise ValueError(f'Compiled native source mismatch: {relative}')
    build_paths = build_proof_artifacts(args.build_proof, image, compiled)
    paths = [Path(__file__), REPORT/'code/route_search_policy.py', REPORT/'code/route_eta.py',
             REPORT/'code/features.py', REPORT/'code/fleet_cases.py', REPORT/'code/fleet_execution.py',
             REPORT/'code/run_decisions.py', REPORT/'code/decision_metrics.py',
             REPORT/'code/record_dataset.py', REPO/'experiments/junction/scenario.py',
             REPO/'experiments/zones/run.py', REPO/'experiments/zones/generate.py',
             REPO/'experiments/baseline/analyze.py', REPO/'experiments/prediction/flow_metrics.py',
             REPO/'experiments/prediction/joint.py', REPO/'experiments/prediction/admission.py',
             REPO/'run_lifelong.py', *build_paths]
    hashes = {str(p.relative_to(REPORT.parent)): digest(p) for p in paths}
    specs = [dict(run_id=f'w{s}_fleet{n}_{m}', workload_seed=s, num_agents=n, method=m,
                  trigger='outer_overlap', horizon_s=args.horizon_s, image_id=image,
                  minimum_mean_eta_gain_s=args.minimum_mean_eta_gain_s)
             for s in args.seeds for n in args.fleets for m in args.methods]
    random.Random(20261103).shuffle(specs)
    manifest = {'schema_version': 1, 'status': 'prepared', 'accounting': 'native_time_only',
                'build_proof': str(args.build_proof.resolve().relative_to(REPORT.parent)),
                'source_sha256': hashes, 'compiled_source_sha256': compiled, 'image_id': image, 'runs': specs,
                'host': {'platform': platform.platform(), 'machine': platform.machine(),
                         'python': platform.python_version(), 'threads': 1},
                'policy': {'version': 'native-route-candidate-search-v1', 'refresh_s': 2., 'warmup_s': 12.,
                           'minimum_mean_eta_gain_s': args.minimum_mean_eta_gain_s,
                           'cost_forecast_horizon_s': 20., 'eta_horizon_s': 30., 'cost_weights': [0., 2., 8.],
                           'candidate_budget': 3, 'primary_cpu_budget_per_candidate_s': 1.,
                           'base_failure': 'Skip alternatives and execute unchanged native fallback.',
                           'selection': 'Zero wins ties and sub-margin gains; install exact evaluated plan.',
                           'objective': 'Mean common accepted-service ETA with optimistic nominal tail.',
                           'candidate_generation': 'Nominal and rollout share forecast-generated spatial penalties.',
                           'oracle': False, 'future_replanning_modeled': False},
                'scope': 'Exploratory non-stopping route-cost selection; original starts, tasks and commitments retained.'}
    root = REPORT / 'data/route_search' / args.study
    path = root / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('source_sha256', 'compiled_source_sha256', 'image_id', 'runs', 'policy', 'build_proof'):
            if old[key] != manifest[key]:
                raise ValueError(f'Resume mismatch: {key}')
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'source_snapshot'
    sources = {**hashes, **{str(Path('simulator') / p): h for p, h in compiled.items()}}
    for relative, expected in sources.items():
        target = archive / relative
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f'Archived source mismatch: {relative}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPORT.parent / relative, target)
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
