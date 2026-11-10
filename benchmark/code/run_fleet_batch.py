#!/usr/bin/env python3
"""Execute a prepared fleet batch on its current host using only the stdlib."""
import argparse
import json
import os
from pathlib import Path
import platform

from fleet_execution import run, write_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--backend', choices=['docker', 'bwrap'], required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--docker-host')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(); manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'prepared':
        raise ValueError('Execute a fresh prepared batch; existing runs must be inspected before reuse')
    manifest.update(status='running', execution={'hostname': platform.node(), 'architecture': platform.machine(),
        'backend': args.backend, 'runtime': None if args.runtime is None else str(args.runtime),
        'logical_cpus': os.cpu_count(), 'batch_runner_sha256': digest(__file__),
        'recorder_sha256': digest(Path(__file__).with_name('fleet_execution.py'))})
    write_json(manifest_path, manifest)
    try:
        for row in manifest['runs']:
            relative = Path(row['relative_directory'])
            if len(relative.parts) != 1 or relative.name in ('.', '..') or relative.is_absolute():
                raise ValueError('Run directory must be a direct child of the batch')
            directory = manifest_path.parent / relative
            print('START', row['run_id'], flush=True)
            result = run(directory, backend=args.backend, runtime=args.runtime, repo=args.repo, docker_host=args.docker_host)
            print('DONE', row['run_id'], f'{result["recording_wall_s"]:.1f}s', flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc)); write_json(manifest_path, manifest)
        raise
    manifest['status'] = 'complete'; write_json(manifest_path, manifest)
    print('COMPLETE', len(manifest['runs']), flush=True)


if __name__ == '__main__':
    main()
