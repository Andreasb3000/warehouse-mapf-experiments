#!/usr/bin/env python3
"""Matched native safety-response ablations on the shared fleet floor.

This measures the total execution effect of the polygon response, including
its downstream effect on repeated planning. Safety-off delay is not identified
as pure geometric capacity: ordinary planner, dependencies, kinematics and
service remain. No predictor or route-cost intervention is used.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess

from features import REPORT, REPO
from fleet_cases import make_fleet_workload
from fleet_execution import digest, write_json
from run_route_search import run_case, build_proof_artifacts
from experiments.zones.run import docker_prefix

CONDITIONS = ('disabled', 'body_in_outer', 'outer_overlap')


def inputs_for(spec):
    inputs = make_fleet_workload(spec['workload_seed'], duration_s=spec['horizon_s'],
                                fleet_size=spec['num_agents'], trigger='outer_overlap')
    scenario = inputs[1]
    condition = spec['safety_condition']
    if condition not in CONDITIONS:
        raise ValueError('Unknown safety condition')
    trigger = 'outer_overlap' if condition == 'disabled' else condition
    scenario['safety'].update(enabled=True, inner_stop_enabled=False, robot_trigger=trigger)
    for agent in scenario['agents']:
        # Keep the instrumentation and all bodies/obstacles. Disabling an
        # agent's response does not remove that robot from the physical floor.
        agent.setdefault('safety', {}).update(enabled=condition != 'disabled',
                                             inner_stop_enabled=False, robot_trigger=trigger)
    return inputs


def matched_digest(inputs):
    """Prove all non-treatment inputs are identical within a workload/fleet."""
    normalized = deepcopy(inputs)
    safety = normalized[1]['safety']
    safety.pop('robot_trigger', None)
    for agent in normalized[1]['agents']:
        for key in ('enabled', 'robot_trigger'):
            agent['safety'].pop(key, None)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--build-proof', type=Path, default=REPORT/'native/build_proof.json')
    parser.add_argument('--docker-host', default=None)
    parser.add_argument('--fleets', nargs='+', type=int, choices=(27, 54, 81), default=[27, 54, 81])
    parser.add_argument('--seeds', nargs='+', type=int, default=[8401, 8402, 8403])
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--horizon-s', type=int, default=240)
    args = parser.parse_args()
    if Path(args.study).name != args.study or args.study in ('', '.', '..'):
        parser.error('Choose a fresh simple study name')
    if args.horizon_s < 20 or args.horizon_s % 2:
        parser.error('Duration must be even and at least 20 seconds')
    if any(len(v) != len(set(v)) for v in (args.fleets, args.seeds, args.conditions)):
        parser.error('Duplicate fleets, seeds or conditions')
    docker = docker_prefix(args.docker_host)
    image = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    compiled = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64', image,
                          'cat', '/usr/project/route-search-source-sha256.json'], text=True))
    for relative, expected in compiled.items():
        if digest(REPO / relative) != expected:
            raise ValueError(f'Compiled native source mismatch: {relative}')
    build_paths = build_proof_artifacts(args.build_proof, image, compiled)
    specs = [dict(run_id=f'w{s}_fleet{n}_{c}', workload_seed=s, num_agents=n,
                  method='none', trigger='body_in_outer' if c == 'body_in_outer' else 'outer_overlap',
                  safety_condition=c, horizon_s=args.horizon_s, image_id=image,
                  minimum_mean_eta_gain_s=.5)
             for s in args.seeds for n in args.fleets for c in args.conditions]
    random.Random(20261105).shuffle(specs)
    prepared = {spec['run_id']: inputs_for(spec) for spec in specs}
    match_groups = {}
    for spec in specs:
        key = f"w{spec['workload_seed']}_fleet{spec['num_agents']}"
        value = matched_digest(prepared[spec['run_id']])
        if key in match_groups and value != match_groups[key]:
            raise ValueError(f'Non-treatment inputs differ: {key}')
        match_groups[key] = value
    # Archive local Python execution dependencies as well as every compiled
    # native source. Include analysis adapters so later metrics are traceable.
    paths = sorted(set((REPORT/'code').glob('*.py')) |
                   set((REPO/'experiments').rglob('*.py')) |
                   {REPO/'run_lifelong.py'} | set(build_paths))
    hashes = {str(p.relative_to(REPORT.parent)): digest(p) for p in paths}
    manifest = {
        'schema_version': 1, 'status': 'prepared', 'accounting': 'native_time_only',
        'study': args.study, 'image_id': image, 'runs': specs,
        'build_proof': str(args.build_proof.resolve().relative_to(REPORT.parent)),
        'source_sha256': hashes, 'compiled_source_sha256': compiled,
        'non_treatment_input_sha256': match_groups,
        'host': {'platform': platform.platform(), 'machine': platform.machine(),
                 'python': platform.python_version(), 'threads': 1},
        'protocol': {'refresh_s': 2., 'primary_cpu_budget_s': 1., 'route_costs': 'zero',
                     'planner': 'PBS', 'fallback': 'PIBT', 'single_agent_solver': 'ASTAR',
                     'planner_seed': 'uint32(workload_seed) XOR (uint32(clock)*0x9e3779b9)',
                     'disabled_control': 'Global instrumentation enabled; every agent response disabled.',
                     'geometry_and_tasks': 'Identical within workload/fleet; larger fleets are nested prefixes.',
                     'serial_native_runs': True},
        'interpretation': 'Safety response on/off is a total closed-loop treatment effect, not a decomposition into pure road capacity and avoidable planner-model mismatch.',
    }
    root = REPORT/'data/safety_capacity'/args.study
    path = root/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in ('source_sha256', 'compiled_source_sha256', 'image_id', 'runs', 'protocol', 'non_treatment_input_sha256', 'build_proof'):
            if old.get(key) != manifest[key]:
                raise ValueError(f'Resume mismatch: {key}')
    root.mkdir(parents=True, exist_ok=True)
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
    manifest['status'] = 'running'; write_json(path, manifest)
    try:
        results = []
        for spec in specs:
            result = run_case(root/spec['run_id'], spec, args.docker_host, inputs=prepared[spec['run_id']])
            results.append(result)
            write_json(root/'results.json', results)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc)); write_json(path, manifest); raise
    manifest['status'] = 'complete'; write_json(path, manifest)
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
