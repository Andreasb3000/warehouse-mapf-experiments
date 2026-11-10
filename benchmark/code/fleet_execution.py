#!/usr/bin/env python3
"""Standard-library native recorder for arbitrary finite fleet sizes.

Zero spatial penalties leave routing to the same native PBS/ADG planner. No
predictor or congestion model is required. The simulator, not this recorder,
advances robot motion and acknowledges pickup/delivery tasks.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objects(path):
    decoder = json.JSONDecoder()
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            offset = 0
            while True:
                start = line.find('{', offset)
                if start < 0:
                    break
                try:
                    value, end = decoder.raw_decode(line, start)
                except ValueError:
                    offset = start + 1
                    continue
                if isinstance(value, dict) and 'type' in value:
                    yield value
                offset = end


SAFETY_DEFAULTS = {
    'enabled': True, 'robot_trigger': 'body_in_outer', 'inner_stop_enabled': True,
    'slow_front_m': 1.2, 'slow_back_m': .2, 'slow_half_width_m': .3,
    'stop_front_m': .3, 'stop_back_m': .15, 'stop_half_width_m': .18,
    'speed_scale': .2, 'robot_radius_m': .085,
}


def validate_safety_configuration(event, scenario):
    """Match the effective native settings, including per-agent overrides.

    Per-agent disabled responses retain instrumentation and physical bodies.
    Global disabling omits native safety metadata and is not this recorder's
    supported instrumented control.
    """
    robots = event['robots']
    agents = {a['id']: a for a in scenario['agents']}
    if len(robots) != len(agents) or {r['agent_id'] for r in robots} != set(agents):
        raise ValueError('Native safety robot IDs differ from scenario')
    for robot in robots:
        expected = {**SAFETY_DEFAULTS, **scenario.get('safety', {}),
                    **agents[robot['agent_id']].get('safety', {})}
        if any(robot.get(key) != expected[key] for key in SAFETY_DEFAULTS):
            raise ValueError('Native safety configuration differs from scenario')


def summarize(directory, wall_s=None):
    directory = Path(directory)
    scenario = json.loads((directory / 'scenario.json').read_text())
    tasks = json.loads((directory / 'tasks.json').read_text())['tasks']
    config = json.loads((directory / 'config.json').read_text())
    count = len(scenario['agents']); ids = set(range(count))
    if {t['robot_id'] for t in tasks} != ids or len(tasks) != count:
        raise ValueError('Exactly one finite pickup/delivery task is required per robot')
    by_robot = {t['robot_id']: t for t in tasks}
    progress = {i: 0 for i in ids}; outcomes = {i: {'robot_id': i} for i in ids}
    previous = {}; travel = {i: 0.0 for i in ids}; frames = 0; goals = 0
    failures = 0; plans = 0; barriers = []; safety_seen = False
    ticks = config['parameters']['ticks_per_second']
    for event in objects(directory / 'simulator.log'):
        kind = event['type']
        if kind == 'tick':
            if event['clock'] != frames or {a['id'] for a in event['agents']} != ids:
                raise ValueError('Missing, duplicate or mismatched native frame')
            for agent in event['agents']:
                i = agent['id']; point = (agent['x'], agent['y'])
                if i in previous:
                    travel[i] += math.dist(point, previous[i])
                previous[i] = point
            frames += 1
        elif kind == 'goal_reached':
            i = event['agent']
            if i not in ids or progress[i] >= len(by_robot[i]['stages']):
                raise ValueError('Unexpected or duplicate native task acknowledgement')
            stage = by_robot[i]['stages'][progress[i]]
            if math.dist((event['row'], event['col']), stage['location']) > .02:
                raise ValueError('Acknowledged task coordinate differs from assigned stage')
            field = 'pickup_completed_s' if stage['operation'] == 'pickup' else 'delivery_completed_s'
            outcomes[i][field] = event['clock'] / ticks
            progress[i] += 1; goals += 1
        elif kind == 'routing_plan':
            plans += 1; failures += not event.get('success', False)
        elif kind == 'routing_barrier_complete':
            barriers.append(event['clock'])
        elif kind == 'safety_config':
            validate_safety_configuration(event, scenario)
            safety_seen = True
    expected_ticks = config['parameters']['sim_duration']
    if frames != expected_ticks or not safety_seen or barriers != list(range(0, expected_ticks, 20)):
        raise ValueError('Incomplete native recording or missing safety/barrier metadata')
    summary = {
        'scope': 'Native recording integrity and task acknowledgements; no congestion definition or acceptance gate.',
        'num_agents': count, 'task_count': len(tasks),
        'pickups_completed': sum('pickup_completed_s' in v for v in outcomes.values()),
        'deliveries_completed': sum('delivery_completed_s' in v for v in outcomes.values()),
        'goal_ack_count': goals, 'native_frames': frames, 'native_duration_s': frames / ticks,
        'recording_wall_s': wall_s, 'native_plan_failures': failures, 'native_plan_count': plans,
        'routing_barriers': len(barriers), 'agents_moved': sum(v > .01 for v in travel.values()),
        'travel_by_robot_m': {str(i): v for i, v in sorted(travel.items())},
        'task_outcomes': [outcomes[i] for i in sorted(ids)],
        'source_sha256': {p: digest(directory / p) for p in ('config.json', 'map.json', 'scenario.json', 'tasks.json', 'simulator.log')},
    }
    write_json(directory / 'execution_summary.json', summary)
    return summary


def run(directory, *, backend='docker', docker_host=None, runtime=None, repo=None, timeout_s=1200):
    directory = Path(directory).resolve()
    if (directory / 'execution_summary.json').exists():
        raise ValueError('A completed recording exists; use a fresh run directory')
    if (directory / 'simulator.log').exists() or (directory / 'launcher.log').exists():
        raise ValueError('An incomplete recording exists; preserve it and use a fresh run directory')
    config = json.loads((directory / 'config.json').read_text())
    scenario = json.loads((directory / 'scenario.json').read_text())
    map_doc = json.loads((directory / 'map.json').read_text())
    count = len(scenario['agents'])
    if config['parameters']['num_agents'] != count:
        raise ValueError('Configured fleet differs from actual scenario agents')
    bridge = directory / 'bridge'; bridge.mkdir(exist_ok=True)
    stop = None
    if backend == 'docker':
        repo = Path(repo).resolve()
        prefix = ['docker']
        if docker_host:
            prefix += ['--host', docker_host]
        name = 'fleet-native-' + uuid.uuid4().hex[:10]
        params = dict(config['parameters'], container=True, visualizer='web', save_stats=True,
                      scenario_file='/inputs/scenario.json', stats_name='/results/stats.json',
                      argos_config_filepath='/results/simulation.argos', output_log='/results/simulator.log')
        command = prefix + ['run', '--rm', '--init', '--name', name, '--platform', 'linux/amd64',
                            '--ulimit', 'nofile=65536:65536',
                            '-v', f'{directory}:/inputs:ro', '-v', f'{directory}:/results',
                            '-v', f'{bridge}:/bridge',
                            '-v', f'{repo / "run_lifelong.py"}:/usr/project/run_lifelong.py:ro',
                            '-v', f'{repo / "experiments/junction/scenario.py"}:/usr/project/experiments/junction/scenario.py:ro',
                            '-e', 'LSMART_SCENARIO_FILE=/inputs/scenario.json', '-e', 'LSMART_RECORD_PLANS=1',
                            '-e', 'LSMART_ROUTING_DIR=/bridge', '-e', 'LSMART_ROUTING_INTERVAL_TICKS=20',
                            config['image'], 'python3', 'run_lifelong.py', '/inputs/map.json']
        command += [f'--{key}={str(value).lower() if isinstance(value, bool) else value}' for key, value in params.items()]
        stop = prefix + ['stop', '--time', '2', name]
    elif backend == 'bwrap':
        runtime = Path(runtime).resolve()
        command = ['python3', str(runtime / 'launch_native.py'), '--runtime', str(runtime), '--run-dir', str(directory)]
    else:
        raise ValueError('Unknown runtime backend')
    write_json(directory / 'command.json', command)
    write_json(directory / 'execution_provenance.json', {
        'backend': backend, 'native_image': config['image'], 'runtime': None if runtime is None else str(runtime),
        'recorder_sha256': digest(__file__), 'input_sha256': {p: digest(directory / p) for p in ('config.json', 'map.json', 'scenario.json', 'tasks.json')},
        'forecast_source': 'zero', 'routing_interval_ticks': 20,
    })
    clock = 0; started = time.perf_counter()
    penalties = [0.0] * (len(map_doc['layout']) * len(map_doc['layout'][0]))
    with (directory / 'launcher.log').open('w') as launcher:
        process = subprocess.Popen(command, stdout=launcher, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.perf_counter() - started > timeout_s:
                    raise TimeoutError(f'Native run exceeded {timeout_s}s: {directory}')
                request = bridge / f'request_{clock}.json'
                if not request.exists():
                    time.sleep(.002); continue
                event = json.loads(request.read_text())
                if event['clock'] != clock or {a['id'] for a in event['pose_frame']['agents']} != set(range(count)):
                    raise ValueError('Native request has mismatched clock or robot IDs')
                write_json(bridge / f'response_{clock}.json', {'clock': clock, 'cell_penalties': penalties})
                clock += 20
            if process.returncode:
                raise RuntimeError(f'Native launcher exited {process.returncode}; inspect {directory / "launcher.log"}')
        except BaseException:
            if stop:
                subprocess.run(stop, capture_output=True, timeout=20)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    return summarize(directory, time.perf_counter() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--backend', choices=['docker', 'bwrap'], default='docker')
    parser.add_argument('--docker-host')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--timeout-s', type=float, default=1200)
    args = parser.parse_args()
    result = run(args.run_directory, backend=args.backend, docker_host=args.docker_host,
                 runtime=args.runtime, repo=args.repo, timeout_s=args.timeout_s)
    print(json.dumps({k: result[k] for k in ('num_agents', 'native_frames', 'deliveries_completed', 'agents_moved', 'recording_wall_s')}, indent=2))
