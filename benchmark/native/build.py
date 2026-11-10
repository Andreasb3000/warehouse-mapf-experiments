#!/usr/bin/env python3
"""Build the native simulator and write fresh source/image provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from native_manifest import manifests

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(__file__).resolve().parent
SIMULATOR = ROOT / 'simulator'
DEFAULT_IMAGE = 'warehouse-congestion:local'
CHECKS = ['safety_geometry', 'finite_tasks', 'entry_gate', 'admission_bridge',
          'route_preview_purity_primary_fallback', 'pibt_terminal_parking',
          'routing_penalties', 'route_candidate_selection',
          'pbs_path_export_service_markers', 'pbs_terminal_parking']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default=DEFAULT_IMAGE)
    parser.add_argument('--docker-host', help='Optional Docker daemon; otherwise use DOCKER_HOST/current context')
    args = parser.parse_args()
    docker = ['docker'] + (['--host', args.docker_host] if args.docker_host else [])
    subprocess.run(docker + ['info', '--format', '{{.ServerVersion}}'], check=True)
    subprocess.run([sys.executable, str(SIMULATOR / 'experiments/zones/fetch_argos.py')], check=True)
    with tempfile.TemporaryDirectory(prefix='warehouse-native-build-') as directory:
        context = Path(directory)
        ignore = shutil.ignore_patterns('.git', 'build', '__pycache__', '.DS_Store', '*.pyc', '*.log', 'cmake-build-*')
        for name in ('client', 'server', 'planner/PBS', 'ArgosConfig'):
            shutil.copytree(SIMULATOR / name, context / name, ignore=ignore)
        for name in ('run_lifelong.py', 'run_lifelong_kwargs.py', 'experiments/junction/scenario.py', '.local/argos3-beta59-ubuntu20.deb'):
            destination = context / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SIMULATOR / name, destination)
        for name in ('Dockerfile', 'native_manifest.py'):
            shutil.copy2(NATIVE / name, context / name)
        subprocess.run(docker + ['build', '--platform', 'linux/amd64', '-t', args.tag, str(context)], check=True)
    image_id = subprocess.check_output(docker + ['image', 'inspect', args.tag, '--format', '{{.Id}}'], text=True).strip()
    expected = manifests(SIMULATOR)
    for name, value in expected.items():
        actual = json.loads(subprocess.check_output(docker + ['run', '--rm', '--platform', 'linux/amd64',
                           image_id, 'cat', '/usr/project/' + name], text=True))
        if actual != value:
            raise RuntimeError(f'Built image source manifest differs: {name}')
    proof = {'schema_version': 1, 'image': args.tag, 'image_id': image_id,
             'built_at_utc': datetime.now(timezone.utc).isoformat(),
             'compiled_source_sha256': expected['route-search-source-sha256.json'],
             'compiled_native_source_sha256': expected['server-source-sha256.json'],
             'build_artifacts_sha256': {name: digest(NATIVE / name) for name in ('Dockerfile', 'build.py', 'native_manifest.py')},
             'native_checks': CHECKS, 'source_hashes_match_workspace': True,
             'scope': 'Fresh build of current checkout, including terminal-parking correction. Historical pre-correction results are not reproduced bit-for-bit.'}
    write_json(NATIVE / 'build_proof.json', proof)
    write_json(ROOT / 'benchmark/results/routing_native_provenance.json', proof)
    print(json.dumps({'image_id': image_id, 'proof': str(NATIVE / 'build_proof.json'), 'native_checks': CHECKS}, indent=2))


if __name__ == '__main__':
    main()
