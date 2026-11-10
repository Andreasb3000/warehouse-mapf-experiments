#!/usr/bin/env python3
"""Paired native admission-control experiment on nested 27/54/81 fleets.

No routing penalties, task reassignment, or physical velocity overrides. The
opt-in server gate delays dispatch of eligible uncommitted entry actions only.
Runs are serial so resource contention does not bias the one-second PBS budget.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import random
import subprocess
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from features import REPORT, REPO, pose_array, rasterize
from fleet_cases import make_fleet_workload
from fleet_execution import digest, objects, summarize, write_json
from run_decisions import LiveHistory
from decision_metrics import task_completions, charge_query_time
from entry_gate_policy import EntryGatePolicy, POLICY
from experiments.zones.run import docker_prefix, mounts, environment, flags
from experiments.junction.scenario import load_scenario
from runtime import validate_native_image

METHODS = ('none', 'reactive', 'forecast')


def score(directory, wall_s=None):
    directory = Path(directory)
    native = summarize(directory, wall_s)
    spec = json.loads((directory / 'run_spec.json').read_text())
    scenario = json.loads((directory / 'scenario.json').read_text())
    tasks = json.loads((directory / 'tasks.json').read_text())['tasks']
    events = list(objects(directory / 'simulator.log'))
    queries = [json.loads(line) for line in (directory / 'host_queries.jsonl').read_text().splitlines()]
    barriers = {e['clock']: e for e in events if e['type'] == 'routing_barrier_complete'}
    plans = {e['clock']: e for e in events if e['type'] == 'routing_plan'}
    accepted = {e['clock']: e for e in events if e['type'] == 'plan_snapshot'}
    expected = list(range(0, round(spec['horizon_s'] * 10), 20))
    for kind in ('routing_barrier_complete', 'routing_plan', 'plan_snapshot'):
        if sum(e['type'] == kind for e in events) != len(expected):
            raise ValueError(f'Duplicate or missing {kind} events')
    if sorted(barriers) != expected or [q['clock'] for q in queries] != expected or sorted(plans) != expected:
        raise ValueError('Missing or mismatched native decision boundaries')
    if any(e.get('nonzero_penalty_cells', 0) != 0 for e in plans.values()):
        raise ValueError('Admission-only experiment unexpectedly applied routing penalties')
    if sorted(accepted) != expected or any(accepted[t]['planner_source'] !=
            ('primary' if plans[t]['success'] else 'fallback') for t in expected):
        raise ValueError('Planner outcomes do not match accepted primary/fallback plans')
    for query in queries:
        query['query_s'] = barriers[query['clock']]['wall_s']
        query['planner'] = plans[query['clock']]
    (directory / 'routing_decisions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in queries))
    rows = task_completions(events, tasks, 10, spec['horizon_s'])
    charge_query_time(rows, queries, spec['horizon_s'])
    blocks = [e for e in events if e['type'] == 'entry_gate_blocked']
    block_ticks = {(e['clock'], e['robot_id']) for e in blocks}
    empty_ticks = {(e['clock'], e['robot_id']) for e in blocks
                   if e.get('queued_action_count') == 0 and e.get('dispatchable_before_gate') == 0}
    # Native controllers request more actions every ten ticks. If both the
    # existing queue and this response are empty, no action can arrive until
    # the next poll. A denial is an event, not a 0.1-second duration sample.
    idle_ticks = {(tick, robot) for clock, robot in empty_ticks
                  for tick in range(clock, min(clock + 10, round(spec['horizon_s'] * 10)))}
    per_robot = Counter(robot for _, robot in idle_ticks)
    frames = [e for e in events if e['type'] == 'tick']
    states = np.stack([pose_array(e) for e in frames])
    shape = (len(json.loads((directory / 'map.json').read_text())['layout']),) * 2
    _, slow = rasterize(states, scenario, shape)
    central_slow = slow & np.all(np.abs(states[:, :, :2] - np.asarray(POLICY['center']))
                                 <= POLICY['half_width'], axis=-1)
    streak = np.zeros(states.shape[1], dtype=int)
    longest_central_slow = 0
    for sample in central_slow:
        streak = (streak + 1) * sample
        longest_central_slow = max(longest_central_slow, int(streak.max()))
    # Distinguish induced stationary waiting from the common slow-traffic proxy.
    stops = set()
    for clock, robot in idle_ticks:
        # Pose c+1 is observed before control c; its output affects pose c+2.
        t = clock + 2
        if 1 <= t < len(states) and np.linalg.norm(states[t, robot, :2] - states[t - 1, robot, :2]) <= .005:
            stops.add((clock, robot))
    active_queries = [q['host_query_s'] for q in queries if q['reason'] == 'decision']
    barrier_times = [q['query_s'] for q in queries]
    charged_completed = [r['delivery_completed_s'] + r['synchronous_query_pause_s']
                         for r in rows if r['delivery_completed_s'] is not None]
    result = {
        **{k: spec[k] for k in ('run_id', 'num_agents', 'workload_seed', 'trigger', 'method', 'horizon_s')},
        'deliveries_completed': native['deliveries_completed'],
        'deliveries_unfinished': native['num_agents'] - native['deliveries_completed'],
        'completion_fraction': native['deliveries_completed'] / native['num_agents'],
        'restricted_mean_flow_s': float(np.mean([r['restricted_flow_s'] for r in rows])),
        'charged_restricted_mean_flow_s': float(np.mean([r['charged_restricted_flow_s'] for r in rows])),
        'deliveries_by_equal_charged_horizon': sum(t <= spec['horizon_s'] for t in charged_completed),
        'equal_charged_horizon_mean_flow_s': float(np.mean([
            min(r['charged_restricted_flow_s'], spec['horizon_s']) for r in rows])),
        'primary_planner_failures': native['native_plan_failures'],
        'planning_calls': native['native_plan_count'],
        'requested_hold_robot_s': 2 * sum(len(q['held_robot_ids']) for q in queries),
        'dispatch_block_events': len(block_ticks),
        'empty_queue_block_events': len(empty_ticks),
        'empty_queue_gate_wait_robot_s': len(idle_ticks) / 10,
        'verified_stationary_gate_wait_robot_s': len(stops) / 10,
        'blocked_robots': len({robot for _, robot in block_ticks}),
        'max_robot_gate_wait_s': max(per_robot.values(), default=0) / 10,
        'slow_traffic_robot_s': float(slow.sum() / 10),
        'central_slow_robot_s': float(central_slow.sum() / 10),
        'max_continuous_central_slow_s': longest_central_slow / 10,
        'forced_release_robot_intervals': sum(len(q.get('diagnostics', {}).get('forced_release_robot_ids', [])) for q in queries),
        'host_decision_p50_s': float(np.median(active_queries)) if active_queries else 0,
        'host_decision_p95_s': float(np.quantile(active_queries, .95)) if active_queries else 0,
        'barrier_p95_s': float(np.quantile(barrier_times, .95)),
        'total_barrier_s': sum(barrier_times),
        'recording_wall_s': wall_s,
        'task_outcomes': rows,
        'metric_scope': 'All assigned tasks; unfinished tasks contribute the configured horizon. Charged flow adds each full synchronous native barrier once before delivery. Equal-charged-horizon values truncate the resulting timeline at the original horizon, under a stop-the-world deployment assumption, not measured real-world throughput. Dispatch blocks are events. Empty-queue gate wait spans from a denial with no earlier dispatchable actions until the next native one-second poll. Its stationary subset requires <=0.005 m movement per physical tick; neither is causal saved delay. Common slow traffic includes deliberate holds.',
        'source_sha256': {name: digest(directory / name) for name in ('simulator.log', 'host_queries.jsonl', 'scenario.json', 'tasks.json', 'run_spec.json')},
    }
    write_json(directory / 'gate_metrics.json', result)
    return result


def run_case(directory, spec, docker_host):
    directory = Path(directory)
    if (directory / 'gate_metrics.json').exists():
        old = json.loads((directory / 'run_spec.json').read_text())
        if old != spec:
            raise ValueError('Resume specification mismatch')
        return json.loads((directory / 'gate_metrics.json').read_text())
    if directory.exists():
        raise ValueError(f'Incomplete run exists; preserve it and choose a fresh study: {directory}')
    directory.mkdir(parents=True)
    values = make_fleet_workload(spec['workload_seed'], duration_s=spec['horizon_s'],
                                 fleet_size=spec['num_agents'], trigger=spec['trigger'])
    map_doc, scenario, config, tasks, assignment = values
    config.update(image=spec['image_id'], wall_timeout_s=1200)
    config['parameters'].update(sim_window_tick=20, single_agent_solver='ASTAR')
    for name, value in [('map.json', map_doc), ('scenario.json', scenario), ('config.json', config),
                        ('tasks.json', tasks), ('assignment.json', assignment), ('run_spec.json', spec)]:
        write_json(directory / name, value)
    load_scenario(directory / 'scenario.json', map_doc['layout'], spec['num_agents'])
    bridge = directory / 'bridge'; bridge.mkdir()
    docker = docker_prefix(docker_host)
    container = 'entry-gate-' + uuid.uuid4().hex[:10]
    params = dict(config['parameters'], container=True, visualizer='web', save_stats=True,
                  scenario_file='/inputs/scenario.json', stats_name='/results/stats.json',
                  argos_config_filepath='/results/simulation.argos', output_log='/results/simulator.log')
    command = docker + ['run', '--rm', '--init', '--name', container, '--platform', 'linux/amd64',
                        '--ulimit', 'nofile=65536:65536', *mounts(directory), *environment(config),
                        '-e', 'LSMART_ROUTING_DIR=/bridge', '-e', 'LSMART_ROUTING_INTERVAL_TICKS=20',
                        '-v', f'{bridge}:/bridge', '-v', f'{directory}:/results',
                        spec['image_id'], 'python3', 'run_lifelong.py', '/inputs/map.json', *flags(params)]
    write_json(directory / 'command.json', command)
    history = LiveHistory(directory / 'simulator.log')
    policy = EntryGatePolicy()
    penalties = [0.] * (len(map_doc['layout']) * len(map_doc['layout'][0]))
    clock = 0; started = time.perf_counter()
    print('START', spec['run_id'], flush=True)
    with (directory / 'launcher.log').open('w') as launcher, threadpool_limits(limits=1):
        process = subprocess.Popen(command, stdout=launcher, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.perf_counter() - started > 1200:
                    raise TimeoutError(spec['run_id'])
                request = bridge / f'request_{clock}.json'
                if not request.exists():
                    time.sleep(.002); continue
                event = json.loads(request.read_text())
                if event['clock'] != clock:
                    raise ValueError('Request clock mismatch')
                if any(c['poll_interval_ticks'] != 10 for c in event['controllers']):
                    raise ValueError('Gate wait accounting assumes the native ten-tick action poll')
                began = time.perf_counter(); history.update()
                payload = {'clock': clock, 'cell_penalties': penalties}
                detail = {}; held = []
                if clock < 120 or history.goals >= 2 * spec['num_agents'] or spec['method'] == 'none':
                    reason = 'warmup' if clock < 120 else ('all_completed' if history.goals >= 2 * spec['num_agents'] else 'no_gate')
                else:
                    reason = 'decision'
                    gate, detail = policy.decide(event, history.past(event), scenario, map_doc, spec['method'])
                    payload['entry_gate'] = gate
                    held = gate['held_robot_ids']
                # Publish only after all diagnostics are serializable; never leave native paused on a host logging error.
                row = {'clock': clock, 'observation_clock': event['observation_clock'], 'origin_s': clock / 10,
                       'reason': reason, 'held_robot_ids': held, 'diagnostics': detail,
                       'host_query_s': time.perf_counter() - began}
                with (directory / 'host_queries.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                write_json(bridge / f'response_{clock}.json', payload)
                clock += 20
            if process.returncode:
                raise RuntimeError(f'Native run exited {process.returncode}: {directory}')
        except BaseException:
            subprocess.run(docker + ['stop', '--time', '2', container], capture_output=True, timeout=20)
            process.wait(timeout=30)
            raise
    result = score(directory, time.perf_counter() - started)
    print('DONE', spec['run_id'], f"deliveries={result['deliveries_completed']}/{spec['num_agents']} flow={result['restricted_mean_flow_s']:.1f}s blocks={result['dispatch_block_events']} wait={result['empty_queue_gate_wait_robot_s']:.1f} robot-s wall={result['recording_wall_s']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--docker-host', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[8301, 8302, 8303])
    parser.add_argument('--fleets', type=int, nargs='+', choices=[27, 54, 81], default=[27, 54, 81])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--horizon-s', type=int, default=240)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if Path(args.study).name != args.study or args.study in ('.', '..', ''):
        parser.error('Choose a simple fresh study directory name')
    if args.horizon_s < 20 or args.horizon_s % 2:
        parser.error('Horizon must be even and at least 20 seconds')
    root = REPORT / 'data/entry_gates' / args.study
    docker = docker_prefix(args.docker_host)
    image = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    validate_native_image(docker, image)
    compiled = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64',
                          image, 'cat', '/usr/project/entry-gate-source-sha256.json'], text=True))
    for relative, recorded in compiled.items():
        if digest(REPO / 'server' / relative) != recorded:
            raise ValueError(f'Native image differs from checkout: server/{relative}')
    hashes = {str(path.relative_to(REPORT.parent)): digest(path) for path in [Path(__file__),
              REPORT / 'code/entry_gate_policy.py', REPORT / 'code/features.py', REPORT / 'code/fleet_cases.py',
              REPORT / 'code/decision_metrics.py', REPORT / 'code/fleet_execution.py', REPORT / 'code/run_decisions.py',
              REPO / 'experiments/prediction/joint.py', REPO / 'run_lifelong.py']}
    specs = [dict(run_id=f'w{seed}_fleet{count}_{method}', workload_seed=seed, num_agents=count,
                  trigger='outer_overlap', method=method, horizon_s=args.horizon_s, image_id=image)
             for seed in args.seeds for count in args.fleets for method in args.methods]
    random.Random(20261101).shuffle(specs)
    manifest = {'schema_version': 1, 'status': 'prepared', 'purpose': 'Exploratory matched junction-admission experiment; no routing penalties.',
                'policy': POLICY, 'source_sha256': hashes, 'compiled_server_sha256': compiled,
                'image_id': image, 'runs': specs,
                'host': {'platform': platform.platform(), 'machine': platform.machine(),
                         'python': platform.python_version(), 'numpy': np.__version__,
                         'predictor_threads': 1, 'simulator_platform': 'linux/amd64 in Docker on host'},
                'scope': 'One central junction, Outer fields overlap; fleet counts and seeds are listed in runs. Exploratory pilot, not statistical validation.',
                'execution': f'Serial runs; {args.horizon_s} simulated seconds each; one CPU-second PBS budget with unchanged PIBT fallback.'}
    path = root / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ('policy', 'source_sha256', 'image_id', 'runs'):
            if previous[key] != manifest[key]:
                raise ValueError(f'Resume mismatch: {key}')
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    if args.prepare_only:
        print(path); return
    manifest['status'] = 'running'; write_json(path, manifest)
    try:
        results = [run_case(root / spec['run_id'], spec, args.docker_host) for spec in specs]
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc)); write_json(path, manifest)
        raise
    write_json(root / 'results.json', results)
    manifest['status'] = 'complete'; write_json(path, manifest)
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
