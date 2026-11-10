"""Portable host metadata and source checks for fresh experiment runs."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess

BENCHMARK = Path(__file__).resolve().parents[1]
SIMULATOR = BENCHMARK.parent / 'simulator'


def evaluation_origins(snapshots, frames, requested=(300, 700, 1100)):
    """Keep published origins when available; support short functional recordings."""
    eligible = sorted(clock for clock in snapshots if clock >= 120
                      and all(k in frames for k in range(clock - 120, clock + 201)))
    chosen = [clock for clock in requested if clock in eligible]
    if chosen:
        return chosen
    if not eligible:
        raise ValueError('No origins with 12 s history and 20 s future; record at least 40 s')
    return sorted({eligible[0], eligible[len(eligible) // 2], eligible[-1]})


def host_info():
    result = {'platform': platform.platform(), 'architecture': platform.machine(),
              'python': platform.python_version(), 'logical_cpus': os.cpu_count(),
              'cpu': platform.processor() or platform.machine()}
    try:
        result['memory_bytes'] = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        result['memory_bytes'] = None
    return result


def python_source_hashes():
    paths = list((BENCHMARK / 'code').glob('*.py'))
    paths += list((SIMULATOR / 'experiments').rglob('*.py'))
    paths += [SIMULATOR / 'run_lifelong.py']
    return {os.path.relpath(p, BENCHMARK): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def validate_native_image(docker, image_id, proof_path=None):
    """Bind the selected image to the current source and recorded build inputs."""
    path = Path(proof_path or BENCHMARK / 'native/build_proof.json')
    proof = json.loads(path.read_text())
    if proof.get('image_id') != image_id:
        raise ValueError('Image differs from native/build_proof.json; rebuild or select the matching image')
    compiled = json.loads(subprocess.check_output(
        docker + ['run', '--rm', '--platform', 'linux/amd64', image_id, 'cat',
                  '/usr/project/route-search-source-sha256.json'], text=True))
    if proof.get('compiled_source_sha256') != compiled or not compiled:
        raise ValueError('Native image source manifest differs from its build proof')
    for relative, expected in compiled.items():
        source = SIMULATOR / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Compiled native source differs from checkout: {relative}')
    artifacts = proof.get('build_artifacts_sha256', {})
    if not artifacts:
        raise ValueError('Native build proof has no build artifact hashes')
    for name, expected in artifacts.items():
        if Path(name).name != name:
            raise ValueError('Build artifact names must be local filenames')
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Native build artifact changed: {name}')
    return proof
