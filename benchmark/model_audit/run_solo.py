#!/usr/bin/env python3
"""Run three fixed original tasks alone to measure isolated native trip time.

Selection is the shortest, median and longest geometric transport lower bound
in the 81-robot seed-8401 assignment, with ties ordered by original robot ID.
These representative trips do not estimate the fleet-average free-flow time.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

REPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPORT/'code'))
from features import REPO
from runtime import validate_native_image
from fleet_execution import digest, write_json
from run_route_search import run_case, build_proof_artifacts
from run_safety_capacity import CONDITIONS, inputs_for
from experiments.junction.scenario import load_scenario
from experiments.zones.run import docker_prefix


def selections():
    base = inputs_for(dict(workload_seed=8401, horizon_s=240, num_agents=81,
                           safety_condition='outer_overlap'))
    ordered = sorted(base[4]['assignments'],
                     key=lambda r: (r['geometric_transport_lower_bound_s'], r['robot_id']))
    return [{'selection': label, 'rank_zero_based': rank,
             'original_robot_id': ordered[rank]['robot_id'],
             'original_assignment': deepcopy(ordered[rank])}
            for label, rank in [('shortest', 0), ('median', len(ordered)//2),
                                ('longest', len(ordered)-1)]]


def solo_inputs(selection, condition, image_id, horizon_s=240):
    values = inputs_for(dict(workload_seed=8401, horizon_s=horizon_s, num_agents=81,
                             safety_condition=condition))
    map_doc, scenario, config, tasks, assignment = deepcopy(values)
    original_id = selection['original_robot_id']
    original_agent = next(a for a in scenario['agents'] if a['id'] == original_id)
    original_task = next(t for t in tasks['tasks'] if t['robot_id'] == original_id)
    agent, task = deepcopy(original_agent), deepcopy(original_task)
    agent.update(id=0, original_robot_id=original_id)
    task.update(robot_id=0, original_robot_id=original_id)
    name = f'solo_original_{original_id}_{selection["selection"]}'
    scenario.update(name=name, agents=[agent],
                    description='One original seed-8401 task executed alone on the unchanged 81-robot floor.')
    zone_ids = [zone['id'] for zone in scenario['zones']]
    counts = {zone: int(zone == agent['zone_id']) for zone in zone_ids}
    matrix = [[0]*len(zone_ids) for _ in zone_ids]
    matrix[zone_ids.index(agent['zone_id'])][zone_ids.index(agent['destination_zone_id'])] = 1
    for zone in scenario['zones']:
        zone['bays'] = {
            'starts': [agent['start']] if zone['id'] == agent['zone_id'] else [],
            'pickups': [agent['goals'][0]] if zone['id'] == agent['zone_id'] else [],
            'dropoffs': [agent['goals'][1]] if zone['id'] == agent['destination_zone_id'] else [],
        }
    initial_matrix = [[counts[f'Z{r}{c}'] for c in range(3)] for r in range(3)]
    scenario['network'].pop('robots_per_zone', None)
    scenario['network'].update(initial_robot_matrix=initial_matrix, allocation_matrix=matrix)
    scenario['notes'] = [
        'Exactly one selected original task is present. Other robots and their tasks are absent.',
        'Map, shelves, starts, pickup/dropoff locations, safety geometry and native settings are retained.',
        'The original robot ID is remapped to native ID zero; original identity is retained in metadata.',
        'Task selected by geometric transport bound, not observed performance. This is not a fleet-average estimate.',
    ]
    tasks.update(scenario=name, tasks=[task], allocation_matrix=matrix,
                 initial_robot_counts=counts, initial_robot_matrix=initial_matrix)
    record = deepcopy(selection['original_assignment'])
    record.update(robot_id=0, original_robot_id=original_id)
    assignment.update(family='isolated_selected_original_task', fleet_size=1,
                      assignments=[record], allocation_matrix=matrix,
                      selection=deepcopy(selection), original_workload_seed=8401,
                      original_fleet_size=81,
                      prefix_rule='One selected original task, native ID remapped to zero.',
                      strictly_center_preferring_tasks=int(record['center_preference_gap_m'] > 0))
    assignment.pop('robots_per_source_zone', None)
    config.update(image=image_id, wall_timeout_s=5400)
    config['parameters'].update(num_agents=1, sim_window_tick=20, single_agent_solver='ASTAR')
    if (map_doc != values[0] or agent['start'] != original_agent['start'] or
            agent['goals'] != original_agent['goals'] or task['stages'] != original_task['stages'] or
            scenario['safety'] != values[1]['safety'] or agent['safety'] != original_agent['safety']):
        raise ValueError('Solo extraction changed original task or geometry')
    if len(scenario['agents']) != 1 or len(tasks['tasks']) != 1 or config['parameters']['num_agents'] != 1:
        raise ValueError('Solo fleet count differs')
    return map_doc, scenario, config, tasks, assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, help='Fresh name for the nine isolated controls')
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--docker-host', default=None)
    parser.add_argument('--build-proof', type=Path, default=REPORT/'native/build_proof.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if Path(args.study).name != args.study or args.study in ('', '.', '..'):
        parser.error('Choose a fresh simple study name')
    docker = docker_prefix(args.docker_host)
    image = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    validate_native_image(docker, image)
    compiled = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64', image,
                          'cat', '/usr/project/route-search-source-sha256.json'], text=True))
    for relative, expected in compiled.items():
        if digest(REPO/relative) != expected:
            raise ValueError(f'Compiled native source mismatch: {relative}')
    build_paths = build_proof_artifacts(args.build_proof, image, compiled)
    chosen = selections()
    specs = [dict(run_id=f'w8401_original{s["original_robot_id"]}_{s["selection"]}_{c}',
                  workload_seed=8401, num_agents=1, original_robot_id=s['original_robot_id'],
                  selection=s['selection'], method='none', safety_condition=c,
                  trigger='body_in_outer' if c == 'body_in_outer' else 'outer_overlap',
                  horizon_s=240, image_id=image, minimum_mean_eta_gain_s=.5)
             for s in chosen for c in CONDITIONS]
    random.Random(20261105).shuffle(specs)
    prepared = {spec['run_id']: solo_inputs(next(s for s in chosen if s['original_robot_id'] == spec['original_robot_id']),
                                          spec['safety_condition'], image) for spec in specs}
    paths = sorted(set((REPORT/'code').glob('*.py')) |
                   set((REPO/'experiments').rglob('*.py')) |
                   {Path(__file__).resolve(), REPO/'run_lifelong.py'} | set(build_paths))
    hashes = {str(p.relative_to(REPORT.parent)): digest(p) for p in paths}
    manifest = {
        'schema_version': 1, 'status': 'prepared', 'study': args.study,
        'source_sha256': hashes, 'compiled_source_sha256': compiled,
        'image_id': image, 'runs': specs, 'selections': chosen,
        'build_proof': str(args.build_proof.resolve().relative_to(REPORT.parent)),
        'host': {'platform': platform.platform(), 'machine': platform.machine(),
                 'python': platform.python_version(), 'threads': 1},
        'protocol': {'refresh_s': 2., 'horizon_s': 240., 'workload_seed': 8401,
                     'primary_cpu_budget_s': 1., 'route_costs': 'zero', 'planner': 'PBS',
                     'fallback': 'PIBT', 'serial_native_runs': True,
                     'selection_rule': 'Sort all 81 original tasks by geometric transport bound then original ID; take zero-based ranks 0, 40, 80.',
                     'disabled_control': 'Instrumentation enabled; lone agent safety response disabled.',
                     'accounting': 'native_time_only',
                     'scope': 'Representative isolated original trips, not a population-average free-flow estimator.'},
    }
    root = REPORT/'data/safety_capacity'/args.study
    root.mkdir(parents=True, exist_ok=True)
    prepared_root = root/'prepared_inputs'
    input_hashes = {}
    for spec in specs:
        directory = prepared_root/spec['run_id']; directory.mkdir(parents=True, exist_ok=True)
        names = ('map.json', 'scenario.json', 'config.json', 'tasks.json', 'assignment.json')
        for name, value in zip(names, prepared[spec['run_id']]):
            path = directory/name
            if path.exists() and json.loads(path.read_text()) != value:
                raise ValueError(f'Prepared input changed: {path}')
            if not path.exists(): write_json(path, value)
        load_scenario(directory/'scenario.json', prepared[spec['run_id']][0]['layout'], 1)
        input_hashes[spec['run_id']] = {name: digest(directory/name) for name in names}
    manifest['prepared_input_sha256'] = input_hashes
    path = root/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('source_sha256', 'compiled_source_sha256', 'image_id', 'runs', 'protocol', 'prepared_input_sha256', 'build_proof'):
            if old.get(key) != manifest[key]:
                raise ValueError(f'Resume mismatch: {key}')
        # Preparing existing complete inputs must not mark a finished study
        # incomplete. A normal resumed run still revalidates cached cases.
        if args.prepare_only:
            manifest['status'] = old['status']
    archive = root/'source_snapshot'
    sources = {**hashes, **{str(Path('simulator')/p): h for p, h in compiled.items()}}
    for relative, expected in sources.items():
        target = archive/relative
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f'Archived source mismatch: {relative}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPORT.parent/relative, target)
    if args.prepare_only:
        write_json(path, manifest)
        print('PREPARED', root)
        print(json.dumps(chosen, indent=2))
        return
    manifest['status'] = 'running'; write_json(path, manifest)
    try:
        results = []
        for spec in specs:
            result = run_case(root/spec['run_id'], spec, args.docker_host, inputs=prepared[spec['run_id']])
            # The native run writes the same frozen inputs; verify after it.
            for name, expected in input_hashes[spec['run_id']].items():
                if digest(root/spec['run_id']/name) != expected:
                    raise ValueError(f'Execution input differs from frozen input: {spec["run_id"]}/{name}')
            results.append(result); write_json(root/'results.json', results)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc)); write_json(path, manifest); raise
    manifest['status'] = 'complete'; write_json(path, manifest)
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
