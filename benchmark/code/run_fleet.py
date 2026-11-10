#!/usr/bin/env python3
"""Prepare or locally record nested 27/54/81-robot scenarios on one floor."""
import argparse
import json
from pathlib import Path
import subprocess

from features import REPORT, REPO
from fleet_cases import make_fleet_workload
from fleet_execution import run, write_json, digest
from experiments.junction.scenario import load_scenario
from experiments.zones.generate import render_svg
from experiments.zones.run import docker_prefix
from runtime import validate_native_image


def prepare(study, fleets, seeds, triggers, horizon_s, image):
    root = REPORT / 'data/fleet_pilots' / study
    if root.exists():
        raise ValueError(f'Study already exists; choose a fresh --study name: {root}')
    cases = []
    for seed in seeds:
        for count in fleets:
            for trigger in triggers:
                values = make_fleet_workload(seed, duration_s=horizon_s, fleet_size=count, trigger=trigger)
                map_doc, scenario, config, tasks, assignment = values
                config.update(image=image, wall_timeout_s=1200)
                config['parameters'].update(sim_window_tick=20, single_agent_solver='ASTAR')
                run_id = f'w{seed}_fleet{count}_{trigger}'
                cases.append((run_id, count, seed, trigger, values))
    # Complete generation first, so unsupported geometry never creates a
    # misleading partly prepared batch.
    root.mkdir(parents=True)
    rows = []
    for run_id, count, seed, trigger, values in cases:
        map_doc, scenario, config, tasks, assignment = values
        folder = root / run_id; folder.mkdir()
        for name, value in [('map.json', map_doc), ('scenario.json', scenario), ('config.json', config),
                            ('tasks.json', tasks), ('assignment.json', assignment)]:
            write_json(folder / name, value)
        load_scenario(folder / 'scenario.json', map_doc['layout'], count)
        (folder / 'layout.svg').write_text(render_svg(map_doc, scenario))
        row = {'run_id': run_id, 'run_directory': str(folder), 'relative_directory': run_id,
               'num_agents': count, 'workload_seed': seed, 'trigger': trigger, 'demand_regime': f'fleet_{count}',
               'method': 'zero', 'native_duration_s': horizon_s}
        write_json(folder / 'run_spec.json', row)
        rows.append(row)
    hashes = {str(p.relative_to(REPORT)): digest(p) for p in
              (Path(__file__), REPORT / 'code/fleet_cases.py', REPORT / 'code/fleet_execution.py')}
    manifest = {'schema_version': 1, 'pilot': True, 'purpose': 'Actual fleet-size scenarios; raw native execution, no congestion definition selected.',
                'status': 'prepared', 'study': study, 'image_id': image, 'expected_runs': len(rows),
                'source_sha256': hashes, 'runs': rows}
    write_json(root / 'manifest.json', manifest)
    return root, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--fleets', nargs='+', type=int, default=[27, 54, 81])
    parser.add_argument('--seeds', nargs='+', type=int, default=[8301])
    parser.add_argument('--triggers', nargs='+', choices=['body_in_outer', 'outer_overlap'], default=['body_in_outer', 'outer_overlap'])
    parser.add_argument('--horizon-s', type=int, default=240)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--docker-host', default=None)
    args = parser.parse_args()
    if Path(args.study).name != args.study or args.study in ('', '.', '..'):
        parser.error('Use a single fresh study directory name')
    if args.horizon_s < 20 or args.horizon_s % 2:
        parser.error('Duration must be at least 20 seconds, in whole two-second intervals')
    if len(set(args.fleets)) != len(args.fleets) or len(set(args.seeds)) != len(args.seeds) or len(set(args.triggers)) != len(args.triggers):
        parser.error('Duplicate fleet sizes, seeds or safety rules are not allowed')
    image = subprocess.check_output(docker_prefix(args.docker_host) + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    validate_native_image(docker_prefix(args.docker_host), image)
    root, manifest = prepare(args.study, args.fleets, args.seeds, args.triggers, args.horizon_s, image)
    if args.prepare_only:
        print(root / 'manifest.json'); return
    manifest['status'] = 'running'; write_json(root / 'manifest.json', manifest)
    try:
        for row in manifest['runs']:
            print('START', row['run_id'], flush=True)
            result = run(row['run_directory'], repo=REPO, docker_host=args.docker_host)
            print('DONE', row['run_id'], f'{result["recording_wall_s"]:.1f}s', flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc)); write_json(root / 'manifest.json', manifest)
        raise
    manifest['status'] = 'complete'; write_json(root / 'manifest.json', manifest)
    print('COMPLETE', root / 'manifest.json')


if __name__ == '__main__':
    main()
