#!/usr/bin/env python3
"""Run difficulty-development pilots without changing the frozen predictor study.

The established native bridge is reused in an isolated worker process, with
only its input factory replaced. These runs tune workload difficulty and are
never treated as held-out evidence that a predictor improves routing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import subprocess

import run_decisions as native
from features import REPORT
from record_dataset import write_json, sha256
from congestion_cases import make_congestion_workload

ORIGINAL_INPUTS = native.inputs
VARIANTS = ('dispersed', 'center12_compact', 'center24_compact', 'center24_spread')


def case_inputs(seed, regime, trigger, horizon, image):
    if regime == 'dispersed':
        return ORIGINAL_INPUTS(seed, 'cross_zone', trigger, horizon, image)
    load, burst = regime.removeprefix('center').split('_')
    result = make_congestion_workload(seed, duration_s=horizon,
                                     center_load=int(load), burst=burst, trigger=trigger)
    _, _, config, _, _ = result
    config.update(image=image, wall_timeout_s=900)
    config['parameters'].update(sim_duration=round(horizon * 10),
                                sim_window_tick=20, single_agent_solver='ASTAR')
    return result


def execute(spec):
    # Each process runs one native case at a time; the replacement is confined
    # to that process and leaves the archived benchmark source/data untouched.
    native.inputs = case_inputs
    result = native.run_case(spec)
    directory = Path(spec['run_directory'])
    provenance = json.loads((directory / 'provenance.json').read_text())
    for path in (Path(__file__), REPORT / 'code/congestion_cases.py'):
        provenance['host_code_sha256'][str(path.relative_to(REPORT))] = sha256(path)
    write_json(directory / 'provenance.json', provenance)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, help='New pilot version name; existing manifests must match exactly.')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[8201])
    parser.add_argument('--triggers', nargs='+', choices=['body_in_outer', 'outer_overlap'],
                        default=['body_in_outer', 'outer_overlap'])
    parser.add_argument('--horizon-s', type=int, default=180)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--image', default='warehouse-congestion:local')
    parser.add_argument('--docker-host', default=None)
    args = parser.parse_args()
    if not args.study or Path(args.study).name != args.study or args.study in ('.', '..'):
        parser.error('Study must be a single directory name')
    if args.horizon_s < 40 or args.jobs < 1:
        parser.error('Use at least 40 simulated seconds and one worker')
    root = REPORT / 'data/congestion_pilots' / args.study
    root.mkdir(parents=True, exist_ok=True)
    docker = native.docker_prefix(args.docker_host)
    image = subprocess.check_output(docker + ['image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    from runtime import validate_native_image
    validate_native_image(docker, image)
    protocol = json.loads((REPORT / 'configs/decision_protocol.json').read_text())
    protocol.update(horizon_s=float(args.horizon_s))
    code_hashes = {str(p.relative_to(REPORT)): sha256(p) for p in
                  (Path(__file__), REPORT / 'code/congestion_cases.py', REPORT / 'code/run_decisions.py')}
    specs = []
    for seed in args.seeds:
        for variant in args.variants:
            for trigger in args.triggers:
                run_id = f'w{seed}_{variant}_{trigger}_zero'
                specs.append(dict(run_id=run_id, workload_seed=seed, demand_regime=variant,
                                  trigger=trigger, method='zero', run_directory=str(root / run_id),
                                  protocol=protocol, image_id=image, docker_host=args.docker_host,
                                  pilot=True, study=args.study, driver_source_sha256=code_hashes,
                                  timing_scope='Difficulty development; concurrent runs are not latency evidence.'
                                  if args.jobs > 1 else 'Serial difficulty-development pilot, not a predictor-utility trial.'))
    random.Random(8200).shuffle(specs)
    manifest = dict(schema_version=1, pilot=True, purpose='Workload difficulty development; no predictor selection',
                    status='running', image_id=image, expected_runs=len(specs), jobs=args.jobs,
                    protocol=protocol, policy=native.POLICY, runs=specs)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ('runs', 'image_id', 'jobs', 'protocol', 'policy'):
            if previous.get(key) != manifest[key]:
                raise ValueError(f'Existing study differs ({key}); choose a new --study name')
    # Validate every generated case before starting any native processes.
    for spec in specs:
        case_inputs(spec['workload_seed'], spec['demand_regime'], spec['trigger'], args.horizon_s, image)
    write_json(manifest_path, manifest)
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(execute, spec) for spec in specs]
            for future in as_completed(futures):
                future.result()
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        write_json(manifest_path, manifest)
        raise
    manifest['status'] = 'complete'
    write_json(manifest_path, manifest)
    print('COMPLETE', len(specs), manifest_path, flush=True)


if __name__ == '__main__':
    main()
