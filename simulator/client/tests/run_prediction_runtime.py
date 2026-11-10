#!/usr/bin/env python3
"""Check that coherent prediction recording observes execution without changing it."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.baseline.analyze import _objects
from experiments.zones.run import docker_prefix, flags, native_paths
from run_trigger_runtime import fixtures, SAFETY, write_json, require


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def check_snapshots(events, ids):
    frames = {e['clock']: e for e in events if e.get('type') == 'tick'}
    snapshots = [e for e in events if e.get('type') == 'prediction_snapshot']
    require(len(snapshots) >= 50, 'Missing one-second coherent snapshots')
    queue_types, timers, max_pose_error = set(), set(), 0
    for event in snapshots:
        clock = event['clock']
        require(clock % 10 == 0, 'Snapshot did not use the one-second cadence')
        require(event['pose_frame']['clock'] == event['snapshot']['clock'] == clock,
                'Snapshot timestamps disagree')
        # ARGoS poses use the incremented native clock. Server/controller counts
        # advance in PostStep and therefore lag this PreStep boundary by one.
        require(event['snapshot']['server_clock'] == clock - 1, 'Server execution boundary differs from native PreStep')
        require(not event['snapshot']['simulation_frozen'], 'Snapshot captured a frozen incomplete planning boundary')
        require(finite_numbers(event), 'Snapshot contains a nonfinite number')
        poses = {p['id']: p for p in event['pose_frame']['agents']}
        controls = {c['id']: c for c in event['controllers']}
        graphs = {r['id']: r for r in event['snapshot']['robots']}
        require(set(poses) == set(controls) == set(graphs) == ids, 'Snapshot omitted a robot')
        # The terminal PreStep snapshot exists before the final experiment exit;
        # external_visualizer has no subsequent Capture() at that final clock.
        actual = {p['id']: p for p in frames[clock]['agents']} if clock in frames else None
        require(actual is not None or clock == max(frames) + 1, 'Missing origin replay frame')
        for robot_id in ids:
            control, graph = controls[robot_id], graphs[robot_id]
            require(control['clock'] == clock and control['control_step_count'] == clock - 1,
                    'Controller snapshot is from a different execution boundary')
            require(control['poll_phase'] == (clock - 1) % control['poll_interval_ticks'], 'Polling phase is inconsistent')
            for axis in ('x', 'y', 'rz'):
                error = abs(poses[robot_id][axis] - actual[robot_id][axis]) if actual else 0
                max_pose_error = max(max_pose_error, error)
                require(error <= .005001, 'Exact snapshot pose does not match the rounded frame at its origin')
            actions = {a['node_id']: a for a in graph['actions']}
            for action in control['queue']:
                queue_types.add(action['type'])
                require(action['node_ids'], 'Queued action has no graph node IDs')
                require(all(node_id in actions for node_id in action['node_ids']), 'Queue contains a missing graph action')
                require(all(graph['finished_node_id'] < node_id <= graph['enqueued_node_id']
                            for node_id in action['node_ids']), 'Queue crosses finished/enqueued boundaries')
                require(action['goal'] == actions[action['node_ids'][-1]]['goal'], 'Controller and graph goal coordinates differ')
                if action['type'] == 'S':
                    timers.add(action['remaining_ticks'])
                    require(0 <= action['remaining_ticks'] <= control['station_duration_ticks'], 'Invalid station timer')
    require('M' in queue_types and 'S' in queue_types, 'Runtime did not exercise movement and service snapshots')
    require(any(0 < tick < 20 for tick in timers), 'No partially elapsed service state was captured')
    return {'snapshots': len(snapshots), 'max_rounded_pose_difference': max_pose_error,
            'observed_queue_types': sorted(queue_types), 'observed_service_remaining_ticks': sorted(timers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--docker-host', default=None)
    parser.add_argument('--output', type=Path, default=ROOT / '.local/prediction-state-runtime')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    docker = docker_prefix(args.docker_host)
    image_id = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    compiled = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64', image_id,
                              'cat', '/usr/project/server-source-sha256.json'], text=True))
    checkout = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in native_paths()}
    require(compiled == checkout, 'Native image differs from checkout; rebuild before validating')
    report = {'valid': False, 'image_id': image_id, 'compiled_native_source_sha256': compiled,
              'compiled_sources_match_checkout': True, 'cases': {}}
    write_json(output / 'validation.json', report)
    try:
        for name in ('isolated_corner', 'following'):
            fixture = fixtures()[name]
            recordings = {}
            for record in (False, True):
                case_name = name + ('-recorded' if record else '-unrecorded')
                directory = output / case_name
                directory.mkdir()
                layout = fixture['layout']
                map_data = {'name': name, 'layout': layout, 'n_row': len(layout), 'n_col': len(layout[0]),
                            'n_endpoint': 0, 'n_agent_loc': 0, 'maxtime': 1000,
                            'milp_runtime': None, 'optimize_wait': False, 'weight': False}
                scenario = {'name': name, 'map_file': 'map.json', 'initial_orientation': 'north',
                            'safety': dict(SAFETY, robot_trigger='outer_overlap'), 'agents': deepcopy(fixture['agents'])}
                parameters = dict(num_agents=len(scenario['agents']), seed=42, n_threads=1,
                                  sim_duration=600, ticks_per_second=10, planner='PBS',
                                  task_assigner_type='distinct_one_goal', backup_solver='PIBT',
                                  single_agent_solver='SIPP', planner_invoke_policy='default', sim_window_tick=20,
                                  planning_window=20, cutoffTime=1, velocity=100, stop_at_congestion=False,
                                  rotation=False, screen=0, container=True, visualizer='web', save_stats=True,
                                  scenario_file='/results/scenario.json', stats_name='/results/stats.json',
                                  output_log='/results/simulator.log', argos_config_filepath='/results/simulation.argos')
                write_json(directory / 'map.json', map_data)
                write_json(directory / 'scenario.json', scenario)
                write_json(directory / 'config.json', {'scenario': 'scenario.json', 'record_plans': record, 'parameters': parameters})
                container = 'prediction-state-' + uuid.uuid4().hex[:10]
                command = docker + ['run', '--rm', '--init', '--name', container, '--platform', 'linux/amd64',
                           '--ulimit', 'nofile=65536:65536', '-e', 'LSMART_SCENARIO_FILE=/results/scenario.json',
                           '-e', f'LSMART_RECORD_PLANS={int(record)}',
                           '-v', f'{directory}:/results', '-v', f'{ROOT / "run_lifelong.py"}:/usr/project/run_lifelong.py:ro',
                           '-v', f'{ROOT / "experiments/junction/scenario.py"}:/usr/project/experiments/junction/scenario.py:ro',
                           image_id, 'python3', 'run_lifelong.py', '/results/map.json', *flags(parameters)]
                write_json(directory / 'command.json', command)
                print(f'Running {case_name}', flush=True)
                try:
                    with (directory / 'launcher.log').open('w') as stream:
                        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=90)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    subprocess.run(docker + ['stop', '--time', '2', container], capture_output=True)
                    raise
                recordings[record] = list(_objects((directory / 'simulator.log').read_text()))
            for kind in ('tick', 'goal_reached', 'safety_summary'):
                require([e for e in recordings[False] if e.get('type') == kind] ==
                        [e for e in recordings[True] if e.get('type') == kind],
                        f'Recording altered execution {kind} events for {name}')
            require(not any(e.get('type') == 'prediction_snapshot' for e in recordings[False]),
                    'Disabled observation emitted snapshots')
            result = check_snapshots(recordings[True], {a['id'] for a in scenario['agents']})
            if name == 'isolated_corner':
                require('T' in result['observed_queue_types'], 'No turning controller state was captured')
            result['recording_did_not_change_execution'] = True
            report['cases'][name] = result
            write_json(output / 'validation.json', report)
        report['valid'] = True
    except Exception as exc:
        report['error'] = str(exc)
        write_json(output / 'validation.json', report)
        raise
    write_json(output / 'validation.json', report)
    print(json.dumps(report['cases'], indent=2))


if __name__ == '__main__':
    main()
