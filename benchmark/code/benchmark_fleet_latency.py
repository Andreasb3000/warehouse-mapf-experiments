#!/usr/bin/env python3
"""Measure complete frozen-predictor queries after simulation/extraction finish.

Inputs and models are preloaded. The timer includes causal raster features,
plan projection when required, prediction, transfers and materialized CPU
output. It excludes disk I/O, simulation, routing, and model loading.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from evaluate_fleet_predictors import (
    METHODS, MODEL_PATHS, REPORT, digest, load_models, load_recording, model_hashes,
    prepare_query, query, source_hashes,
)
from fleet_execution import write_json


def prepare_queries(manifests, origin_s):
    """Use fixed origin times, never choose examples from prediction errors."""
    queries = defaultdict(list)
    seen = set()
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'Native batch is incomplete: {path}')
        for entry in manifest['runs']:
            if entry['trigger'] != 'outer_overlap':
                continue
            if entry['run_id'] in seen:
                raise ValueError('Duplicate run in timing manifests')
            seen.add(entry['run_id'])
            directory = path.parent / entry['relative_directory']
            recording = load_recording(directory)
            for origin in origin_s:
                clock = round(origin * 10)
                q = prepare_query(recording, recording['snapshots'][clock])
                q['workload_seed'] = entry['workload_seed']
                queries[recording['num_agents']].append(q)
    if not queries:
        raise ValueError('Timing requires at least one Safety rule 2 recording')
    return dict(queries)


def summary(rows):
    seconds = np.asarray([r['query_s'] for r in rows])
    return {
        'n': len(rows), 'p50_s': float(np.median(seconds)),
        'p95_s': float(np.percentile(seconds, 95)),
        'mean_s': float(seconds.mean()), 'min_s': float(seconds.min()),
        'max_s': float(seconds.max()), 'samples_s': seconds.tolist(),
        'samples': rows,
    }


def hardware():
    result = {'platform': platform.platform(), 'machine': platform.machine(),
              'python': platform.python_version(), 'torch': torch.__version__}
    if platform.system() == 'Darwin':
        for name, key in [('cpu', 'machdep.cpu.brand_string'),
                          ('logical_cpus', 'hw.logicalcpu'),
                          ('memory_bytes', 'hw.memsize')]:
            value = subprocess.check_output(['sysctl', '-n', key], text=True).strip()
            result[name] = value if name == 'cpu' else int(value)
    return result


def make_runtime_protocol(extraction_path, manifest_paths):
    extraction = json.loads(extraction_path.read_text())
    settings = extraction['protocol']
    origins = list(range(settings['first_origin_s'], settings['last_origin_s'] + 1,
                         settings['origin_stride_s']))
    preferred = [t for t in (32, 92, 152) if t in origins]
    selected = preferred if len(preferred) == 3 else sorted({origins[0], origins[len(origins)//2], origins[-1]})
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    protocol = {
        'schema_version': 2,
        'model_hash_key': 'name',
        'selection': 'Fixed representative origin clocks from the extraction interval, without inspecting errors.',
        'workload_seeds': sorted({r['workload_seed'] for m in manifests for r in m['runs']}),
        'model_sha256': extraction['identity']['model_sha256'],
        'extraction_protocol_sha256': digest(extraction_path),
        'recording_manifest_sha256': {str(p.resolve()): digest(p) for p in manifest_paths},
        'runtime': {'origin_s': selected, 'repeats': 3,
                    'warmup_queries_per_fleet_method': 3, 'cpu_threads': 2,
                    'triggers': ['outer_overlap'],
                    'includes': 'Causal features, required plan projection, inference, transfers, usable CPU output.',
                    'excludes': 'Model loading, disk I/O, simulation, routing, training.'}}
    return protocol


def validate_model_protocol(protocol):
    """Compare named model digests; accept only the exact legacy path schema."""
    expected = protocol['model_sha256']
    actual = model_hashes()
    if protocol.get('model_hash_key') == 'name' or set(expected) == set(actual):
        if expected != actual:
            raise ValueError('Frozen model changed since protocol creation')
        return
    legacy = {str(path.relative_to(REPORT)): actual[name] for name, path in MODEL_PATHS.items()}
    if expected != legacy:
        raise ValueError('Frozen model changed or model-hash schema is unsupported')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path,
                        default=REPORT / 'results/fleet_prediction/runtime.json')
    parser.add_argument('--protocol', type=Path,
                        default=REPORT / 'results/fleet_prediction/runtime_protocol.json')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Timing output already exists; use a fresh --output')
    if not args.protocol.exists():
        extraction_path = args.output.parent / 'extraction_protocol.json'
        protocol = make_runtime_protocol(extraction_path, args.manifests)
        args.protocol.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.protocol, protocol)
    protocol = json.loads(args.protocol.read_text())
    for filename, expected in protocol.get('recording_manifest_sha256', {}).items():
        if digest(filename) != expected:
            raise ValueError('Timing recording manifest differs from frozen protocol')
    spec = protocol['runtime']
    validate_model_protocol(protocol)
    torch.set_num_threads(spec['cpu_threads'])
    queries = prepare_queries([p.resolve() for p in args.manifests], spec['origin_s'])
    for count, values in queries.items():
        if len(values) != len(spec['origin_s']) * len(protocol['workload_seeds']):
            raise ValueError(f'Incomplete workload/origin set for {count} robots')
        actual = {(q['workload_seed'], q['origin_s']) for q in values}
        expected = {(seed, origin) for seed in protocol['workload_seeds']
                    for origin in spec['origin_s']}
        if actual != expected:
            raise ValueError(f'Wrong workload/origin identities for {count} robots')
    models, load_s = {}, {}
    devices = ['cpu']
    if not args.cpu_only and torch.backends.mps.is_available():
        devices.append('mps')
    for device in devices:
        started = time.perf_counter()
        models[device] = load_models(device)
        load_s[device] = time.perf_counter() - started
    identity = {'source_sha256': source_hashes(), 'model_sha256': model_hashes(),
                'benchmark_sha256': digest(__file__),
                'protocol_sha256': digest(args.protocol)}
    result = {
        'status': 'running', 'hardware': hardware(), 'cpu_threads': spec['cpu_threads'],
        'protocol': spec, 'identity': identity, 'model_load_s_excluded': load_s,
        'scope': 'Serial warmed full-fleet in-memory forecast queries. No native '
                 'recording, training, or offline extraction is run concurrently '
                 'by this experiment. Other operating-system activity is uncontrolled.',
        'precomputed_inputs_excluded': 'Reading and packing the bounded native pose history '
                                      'and static map/service masks. History rasterization '
                                      'and plan features are recomputed within each query.',
        'groups': [], 'cpu_mps_parity': [],
    }
    started = time.perf_counter()
    with threadpool_limits(limits=spec['cpu_threads']):
        for count in sorted(queries):
            values = queries[count]
            for device in devices:
                methods = METHODS if device == 'cpu' else ('convlstm', 'residual_convlstm')
                group = {'num_agents': count, 'backend': device, 'methods': {}}
                for method in methods:
                    model = models[device].get(method)
                    for index in range(spec['warmup_queries_per_fleet_method']):
                        query(values[index % len(values)], method, model, device)
                    rows = []
                    order = [(repeat, index) for repeat in range(spec['repeats'])
                             for index in range(len(values))]
                    random.Random(2026).shuffle(order)
                    for repeat, index in order:
                        q = values[index]
                        prediction, elapsed, components, _ = query(q, method, model, device)
                        rows.append({'run_id': q['run_id'], 'workload_seed': q['workload_seed'],
                                     'origin_s': q['origin_s'], 'repeat': repeat,
                                     'query_s': elapsed, 'components_s': components,
                                     'output_checksum': float(prediction.sum())})
                    group['methods'][method] = summary(rows)
                    print(json.dumps({'robots': count, 'backend': device, 'method': method,
                                      'p50_s': group['methods'][method]['p50_s'],
                                      'p95_s': group['methods'][method]['p95_s']}), flush=True)
                result['groups'].append(group)
        if 'mps' in devices:
            # Parity is outside every timed sample and uses no future labels.
            for count in sorted(queries):
                q = queries[count][0]
                for method in ('convlstm', 'residual_convlstm'):
                    cpu, _, _, _ = query(q, method, models['cpu'][method], 'cpu')
                    gpu, _, _, _ = query(q, method, models['mps'][method], 'mps')
                    delta = np.abs(cpu - gpu)
                    agrees = bool(np.allclose(cpu, gpu, rtol=1e-3, atol=1e-4))
                    result['cpu_mps_parity'].append({
                        'num_agents': count, 'method': method, 'run_id': q['run_id'],
                        'origin_s': q['origin_s'], 'max_absolute': float(delta.max()),
                        'mean_absolute': float(delta.mean()), 'within_tolerance': agrees,
                        'rtol': 1e-3, 'atol': 1e-4,
                    })
                    if not agrees:
                        raise ValueError('CPU/MPS prediction parity failed')
    if identity['source_sha256'] != source_hashes() or identity['model_sha256'] != model_hashes():
        raise ValueError('Prediction sources/models changed during timing')
    result.update(status='complete', benchmark_wall_s=time.perf_counter() - started)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    write_json(args.output, result)
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
