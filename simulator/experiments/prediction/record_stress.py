#!/usr/bin/env python3
"""Record a matched 24-robot narrow T-junction for congestion prediction tests.

The finite simultaneous batch is a held-out stress workload, not a calibrated
model of a particular warehouse. Run both outer-only safety rules sequentially.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline.analyze import _objects
from experiments.junction.benchmark.generate import build_case, render_svg
from experiments.junction.benchmark.metrics import measure
from experiments.junction.measure import measure_run
from experiments.junction.scenario import load_scenario
from experiments.zones.run import docker_prefix, flags, native_paths, write_json

SAFETY = {'enabled': True, 'inner_stop_enabled': False,
          'slow_front_m': 1.2, 'slow_back_m': .2, 'slow_half_width_m': .3,
          'stop_front_m': .3, 'stop_back_m': .15, 'stop_half_width_m': .18,
          'speed_scale': .2, 'robot_radius_m': .085}
TRIGGERS = ('body_in_outer', 'outer_overlap')


def make_case(trigger, seed=2026, duration_s=120, image='warehouse-congestion:local'):
    if trigger not in TRIGGERS:
        raise ValueError('Unknown robot trigger')
    if type(seed) is not int or seed < 0 or type(duration_s) is not int or duration_s <= 0:
        raise ValueError('Seed must be nonnegative and duration a positive integer')
    layout, scenario, config = build_case(geometry='t', num_agents=24, core_width=1,
                                         name='prediction_t_narrow_24')
    scenario['safety'] = dict(SAFETY, robot_trigger=trigger)
    scenario['notes'].append('Outer-only safety uses the same reference geometry and speed cap as the nine-zone experiment.')
    config['image'] = image
    config['record_plans'] = True
    config['parameters'].update(seed=seed, sim_duration=duration_s * 10)
    return layout, scenario, config


def record(docker, image_id, output, trigger, seed, duration_s, image, provenance):
    directory = output / trigger
    directory.mkdir()
    layout, scenario, config = make_case(trigger, seed, duration_s, image)
    for name, value in (('map.json', layout), ('scenario.json', scenario), ('config.json', config)):
        write_json(directory / name, value)
    (directory / 'layout.svg').write_text(render_svg(layout, scenario))
    load_scenario(directory / 'scenario.json', layout['layout'], 24)
    run_dir = directory / 'run-01'
    run_dir.mkdir()
    for name in ('map.json', 'scenario.json', 'config.json', 'layout.svg'):
        shutil.copyfile(directory / name, run_dir / name)
    inputs = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
              for name in ('map.json', 'scenario.json', 'config.json')}
    write_json(run_dir / 'provenance.json', dict(provenance, input_sha256=inputs,
               robot_trigger=trigger, inner_stop_enabled=False, seed=seed))
    parameters = dict(config['parameters'], container=True, visualizer='web', save_stats=True,
                      scenario_file='/inputs/scenario.json', stats_name='/results/stats.json',
                      argos_config_filepath='/results/simulation.argos', output_log='/results/simulator.log')
    container = 'prediction-stress-' + uuid.uuid4().hex[:10]
    command = docker + ['run', '--rm', '--init', '--name', container, '--platform', 'linux/amd64',
                       '--ulimit', 'nofile=65536:65536', '-e', 'LSMART_RECORD_PLANS=1',
                       '-e', 'LSMART_SCENARIO_FILE=/inputs/scenario.json',
                       '-v', f'{run_dir}:/inputs:ro', '-v', f'{run_dir}:/results',
                       '-v', f'{ROOT / "run_lifelong.py"}:/usr/project/run_lifelong.py:ro',
                       '-v', f'{ROOT / "experiments/junction/scenario.py"}:/usr/project/experiments/junction/scenario.py:ro',
                       image_id, 'python3', 'run_lifelong.py', '/inputs/map.json', *flags(parameters)]
    write_json(run_dir / 'command.json', command)
    print(f'Recording {trigger}: {run_dir}', flush=True)
    started = time.monotonic()
    try:
        with (run_dir / 'launcher.log').open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=config['wall_timeout_s'])
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        subprocess.run(docker + ['stop', '--time', '2', container], capture_output=True)
        raise
    execution = measure_run(run_dir, config['parameters'], scenario, stress=True)
    execution['wall_runtime_s'] = round(time.monotonic() - started, 3)
    write_json(run_dir / 'junction_metrics.json', execution)
    if not execution['valid']:
        raise RuntimeError(f'Invalid recording: {execution["errors"]}')
    events = list(_objects((run_dir / 'simulator.log').read_text()))
    snapshots = [e for e in events if e.get('type') == 'prediction_snapshot']
    policies = [e for e in events if e.get('type') == 'safety_config']
    if (len(policies) != 1 or len(policies[0]['robots']) != 24 or
        any(r['robot_trigger'] != trigger or r['inner_stop_enabled'] or not r['enabled']
            for r in policies[0]['robots'])):
        raise RuntimeError('Native safety configuration does not match the requested outer-only rule')
    if len(snapshots) != duration_s or any(
            len(e['pose_frame']['agents']) != 24 or len(e['controllers']) != 24
            or e['clock'] != e['snapshot']['clock'] or e['clock'] != e['pose_frame']['clock']
            or {c['id'] for c in e['controllers']} != set(range(24)) for e in snapshots):
        raise RuntimeError('Incomplete coherent prediction recording')
    difficulty = measure(run_dir)
    acknowledgements = [e for e in events if e.get('type') == 'goal_reached']
    result = {'valid': True, 'run_directory': str(run_dir), 'robot_trigger': trigger,
              'seed': seed, 'agents': 24, 'frames': config['parameters']['sim_duration'],
              'prediction_snapshots': len(snapshots),
              'goal_acknowledgements': len(acknowledgements),
              'completed_robot_ids': sorted({e['agent'] for e in acknowledgements}),
              'difficulty': difficulty}
    write_json(run_dir / 'recording_summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'difficulty'}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--docker-host', default=os.environ.get('DOCKER_HOST') or
                        None)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--duration-s', type=int, default=120)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'experiments/junction/results/joint-predictor-stress-20261026')
    args = parser.parse_args()
    make_case('body_in_outer', args.seed, args.duration_s, args.image)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    docker = docker_prefix(args.docker_host)
    image_id = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    compiled = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64', image_id,
                         'cat', '/usr/project/server-source-sha256.json'], text=True))
    checkout = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in native_paths()}
    if compiled != checkout:
        raise RuntimeError('Native image differs from checkout; rebuild using experiments/zones/run.py --build')
    source_paths = [Path(__file__), ROOT / 'experiments/junction/benchmark/generate.py',
                    ROOT / 'experiments/junction/benchmark/metrics.py', ROOT / 'experiments/junction/measure.py',
                    ROOT / 'experiments/junction/scenario.py', ROOT / 'run_lifelong.py']
    provenance = {'image_id': image_id, 'compiled_native_source_sha256': compiled,
                  'compiled_sources_match_checkout': True,
                  'checkout_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in source_paths},
                  'recipe': {'geometry': 't', 'num_agents': 24, 'core_width': 1,
                             'arm_length': 22, 'seed': args.seed, 'duration_s': args.duration_s,
                             'safety': SAFETY},
                  'notes': ['Same finite tasks, map, seed, and safety geometry in both variants.',
                            'Stationary approach counts are a queue proxy, not measured connected queue tails.',
                            'Physical completion is distinct from task counters; goal acknowledgements are retained.']}
    write_json(output / 'provenance.json', provenance)
    summary = {'valid': False, 'cases': {}}
    write_json(output / 'summary.json', summary)
    try:
        for trigger in TRIGGERS:
            summary['cases'][trigger] = record(docker, image_id, output, trigger, args.seed,
                                               args.duration_s, args.image, provenance)
            write_json(output / 'summary.json', summary)
        summary['valid'] = True
    except Exception as exc:
        summary['error'] = str(exc)
        write_json(output / 'summary.json', summary)
        raise
    write_json(output / 'summary.json', summary)
    print(f'Recordings ready: {output}')


if __name__ == '__main__':
    main()
