#!/usr/bin/env python3
"""Frozen predictor transfer to matched 27/54/81-robot native recordings.

This is an offline accuracy screen, not a new congestion definition or a
forecast-guided routing experiment. Each query receives only the origin's
accepted plan/controller state and the preceding twelve seconds of poses.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from features import (REPORT, DT, HISTORY_OFFSETS, WINDOWS, masks,
                      observation_clock, pose_array, rasterize,
                      predicted_rasters, window_mean)
from fleet_execution import objects, write_json
from evaluate import measure

METHODS = ('zero', 'persistence', 'kinematic', 'joint', 'local_ridge',
           'convlstm', 'residual_convlstm')
MODEL_PATHS = {
    'local_ridge': REPORT / 'results/local/model.joblib',
    'convlstm': REPORT / 'results/learned/convlstm/model.pt',
    'residual_convlstm': REPORT / 'results/learned/residual_convlstm/model.pt',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = [Path(__file__), REPORT / 'code/features.py',
             REPORT / 'code/local_model.py', REPORT / 'code/learned.py',
             REPORT / 'code/evaluate.py', REPORT / 'code/fleet_execution.py']
    from features import REPO
    files += list((REPO / 'experiments/prediction').glob('*.py'))
    return {str(p.relative_to(REPORT.parent)): digest(p) for p in sorted(files)}


def model_hashes():
    return {name: digest(path) for name, path in MODEL_PATHS.items()}


def validate_ids(agents, count, label):
    ids = [a['id'] for a in agents]
    if len(ids) != count or set(ids) != set(range(count)):
        raise ValueError(f'{label}: robot IDs differ from actual fleet size {count}')


def load_recording(directory):
    """Read archived evidence; do not pass this full object to a predictor."""
    directory = Path(directory)
    config, scenario, map_doc = [json.loads((directory / name).read_text())
                                 for name in ('config.json', 'scenario.json', 'map.json')]
    count = config['parameters']['num_agents']
    if config['parameters']['ticks_per_second'] != 10:
        raise ValueError('This adapter requires native 0.1-second truth')
    validate_ids(scenario['agents'], count, 'scenario')
    snapshots, frames = {}, {}
    for event in objects(directory / 'simulator.log'):
        if event.get('type') == 'tick':
            clock = event['clock']
            if clock in frames:
                raise ValueError(f'Duplicate native frame {clock}')
            validate_ids(event['agents'], count, f'frame {clock}')
            frame = pose_array(event)
            if not np.isfinite(frame).all():
                raise ValueError('Nonfinite native pose')
            frames[clock] = frame
        elif event.get('type') == 'prediction_snapshot':
            clock = observation_clock(event)
            if clock in snapshots:
                raise ValueError(f'Duplicate coherent snapshot {clock}')
            validate_ids(event['pose_frame']['agents'], count, f'snapshot {clock}')
            validate_ids(event['controllers'], count, f'controller {clock}')
            validate_ids(event['snapshot']['robots'], count, f'plan {clock}')
            snapshots[clock] = event
    if not frames or not snapshots:
        raise ValueError('Native frames and coherent passive snapshots are required')
    maximum = config['parameters']['sim_duration']
    if sorted(frames) != list(range(maximum)):
        raise ValueError('Native frame clocks are missing, reordered, or incomplete')
    shape = (len(map_doc['layout']), len(map_doc['layout'][0]))
    road, roi, service = masks(scenario, map_doc)
    trigger = scenario['safety']['robot_trigger']
    static = np.stack([road, service, np.full(shape, trigger == 'outer_overlap')]).astype(np.float32)
    return {'directory': directory, 'config': config, 'scenario': scenario,
            'map_doc': map_doc, 'frames': frames, 'snapshots': snapshots,
            'shape': shape, 'road': road, 'roi': roi, 'static': static,
            'num_agents': count, 'trigger': trigger}


def prepare_query(recording, event):
    """Project a recording to a causal query with no reference to future frames."""
    clock = observation_clock(event)
    if any(t not in recording['frames'] for t in range(clock - 120, clock)):
        raise ValueError(f'Missing twelve-second causal history at {clock}')
    past = np.stack([recording['frames'][t] for t in range(clock - 120, clock)]
                    + [pose_array(event['pose_frame'])])
    if past.shape != (121, recording['num_agents'], 3):
        raise ValueError('Causal history has an incorrect robot count')
    return {'run_id': recording['directory'].name, 'origin_s': clock * DT,
            'event': event, 'past': past.copy(), 'scenario': recording['scenario'],
            'map_doc': recording['map_doc'], 'shape': recording['shape'],
            'static': recording['static'], 'num_agents': recording['num_agents']}


def target_for(recording, q):
    """Label-only path. Invoke after forecasts have been constructed."""
    clock = observation_clock(q['event'])
    if any(t not in recording['frames'] for t in range(clock + 1, clock + 201)):
        raise ValueError('Incomplete future truth window')
    future = np.stack([recording['frames'][t] for t in range(clock + 1, clock + 201)])
    states = np.concatenate([q['past'][-21:], future])
    rr, _ = rasterize(states, q['scenario'], q['shape'])
    return window_mean(rr[20:]).astype(np.float32)


def load_models(device='cpu'):
    """Load frozen checkpoints only; no fit/selection path exists in this module."""
    import joblib
    import torch
    from learned import SpatialPredictor, synchronize
    result = {'local_ridge': joblib.load(MODEL_PATHS['local_ridge'])}
    for name in ('convlstm', 'residual_convlstm'):
        cp = torch.load(MODEL_PATHS[name], map_location='cpu', weights_only=True)
        model = SpatialPredictor(residual=cp['residual'], hidden=cp['hidden']).to(device)
        model.load_state_dict(cp['state_dict'])
        model.eval()
        result[name] = model
    synchronize(device)
    return result


def validate_prediction(value, q, method):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (3, 2, *q['shape']) or not np.isfinite(value).all():
        raise ValueError(f'{method}: invalid forecast shape or finite values')
    if np.any(value < -1e-6) or np.any(value[:, 1] > value[:, 0] + 1e-5):
        raise ValueError(f'{method}: forecast violates 0 <= slow <= occupancy')
    if method in ('persistence', 'kinematic', 'joint'):
        mass = value[:, 0].sum(axis=(1, 2))
        if not np.allclose(mass, q['num_agents'], rtol=1e-5, atol=1e-4):
            raise ValueError(f'{method}: projected robot mass differs from actual N')
    return value


def query(q, method, model=None, device='cpu'):
    """Independent full query: output, elapsed seconds, components, diagnostics.

    Models must already be loaded; disk I/O and routing are excluded. Every
    learned method independently constructs history and current-plan features.
    """
    from experiments.prediction.joint import rollout
    if method not in METHODS:
        raise ValueError(method)
    neural = method in ('convlstm', 'residual_convlstm')
    if neural:
        import torch
        from learned import synchronize
        synchronize(device)
    start = time.perf_counter()
    components, diagnostics = {}, {}
    event = q['event']
    if method == 'zero':
        result = np.zeros((3, 2, *q['shape']), dtype=np.float32)
    elif method == 'persistence':
        rr, _ = rasterize(q['past'][-21:], q['scenario'], q['shape'])
        result = np.broadcast_to(rr[-1], (3, *rr[-1].shape)).copy()
    elif method == 'joint':
        core = time.perf_counter()
        out = rollout(event['snapshot'], event['pose_frame'], q['scenario'],
                      q['map_doc'], controllers=event['controllers'], dt=.1,
                      horizon_s=20, speed_m_s=1, station_dwell_s=2)
        components['rollout_core'] = time.perf_counter() - core
        rasters, _ = predicted_rasters(out, q['past'], q['scenario'], q['shape'])
        result = window_mean(rasters)
        diagnostics = out['diagnostics']
    else:
        if method != 'kinematic':
            history_start = time.perf_counter()
            rr, _ = rasterize(q['past'], q['scenario'], q['shape'])
            history = np.stack([rr[120 + round(offset / DT)] for offset in HISTORY_OFFSETS])
            components['history_features'] = time.perf_counter() - history_start
        plan_start = time.perf_counter()
        out = rollout(event['snapshot'], event['pose_frame'], q['scenario'],
                      q['map_doc'], controllers=None, dt=.5, horizon_s=20,
                      safety_enabled=False, speed_m_s=1, station_dwell_s=2)
        rasters, _ = predicted_rasters(out, q['past'], q['scenario'], q['shape'])
        plan = window_mean(rasters)
        components['plan_and_raster'] = time.perf_counter() - plan_start
        if method == 'kinematic':
            result = plan
        else:
            plan[:, 1] = 0
            model_start = time.perf_counter()
            if method == 'local_ridge':
                from local_model import predict_one
                result = predict_one(model, history, plan, q['static'])
            else:
                inputs = [torch.from_numpy(np.asarray(a, dtype=np.float32)[None]).to(device)
                          for a in (history, plan, q['static'])]
                with torch.inference_mode():
                    result = model(*inputs).cpu().numpy()[0]
                synchronize(device)
            components['model_with_transfer'] = time.perf_counter() - model_start
    # Materialize a usable host result; validation is not charged as inference.
    result = np.asarray(result, dtype=np.float32)
    float(result.sum())
    elapsed = time.perf_counter() - start
    return validate_prediction(result, q, method), elapsed, components, diagnostics


def time_summary(values):
    a = np.asarray(values)
    return {'n': len(values), 'median_s': float(np.median(a)),
            'p95_s': float(np.percentile(a, 95)), 'mean_s': float(a.mean()),
            'min_s': float(a.min()), 'max_s': float(a.max())}


def origins_for(recording, protocol):
    clocks = list(range(round(protocol['first_origin_s'] / DT),
                        round(protocol['last_origin_s'] / DT) + 1,
                        round(protocol['origin_stride_s'] / DT)))
    for clock in clocks:
        if clock not in recording['snapshots']:
            raise ValueError(f'Missing coherent snapshot at requested origin {clock}')
        if clock + 200 > max(recording['frames']):
            raise ValueError('Requested origin extends beyond recorded truth')
    return clocks


def evaluate_run(entry, output, protocol, models, identity):
    directory = Path(entry['run_directory'])
    out = output / entry['run_id']
    inputs = {name: digest(directory / name) for name in
              ('map.json', 'scenario.json', 'config.json', 'tasks.json', 'simulator.log')}
    signature = {'protocol': protocol, 'identity': identity, 'input_sha256': inputs}
    if (out / 'complete.json').exists():
        done = json.loads((out / 'complete.json').read_text())
        if done['signature'] != signature:
            raise ValueError(f'Existing frozen evaluation differs: {out}')
        for name, expected in done['artifact_sha256'].items():
            if digest(out / name) != expected:
                raise ValueError(f'Existing evaluation artifact changed: {out / name}')
        print('REUSE', entry['run_id'], flush=True)
        return json.loads((out / 'metrics.json').read_text())
    if out.exists():
        raise ValueError(f'Incomplete evaluation exists; inspect before rerun: {out}')
    out.mkdir(parents=True)
    started = time.perf_counter()
    recording = load_recording(directory)
    if recording['num_agents'] != entry['num_agents'] or recording['trigger'] != entry['trigger']:
        raise ValueError('Manifest fleet or safety rule differs from native inputs')
    clocks = origins_for(recording, protocol)
    arrays = {method: [] for method in METHODS}
    arrays['truth'] = []
    samples = []
    for clock in clocks:
        q = prepare_query(recording, recording['snapshots'][clock])
        row = {'origin_s': clock * DT, 'observation_clock': clock, 'timing_s': {},
               'timing_components_s': {}, 'joint_plan_exhausted_at_s': None}
        for method in METHODS:
            pred, elapsed, components, diagnostics = query(q, method, models.get(method))
            arrays[method].append(pred)
            row['timing_s'][method] = elapsed
            row['timing_components_s'][method] = components
            if method == 'joint':
                row['joint_plan_exhausted_at_s'] = diagnostics['plan_exhausted_at_s']
        # Truth is not inspected until every forecast at this origin is frozen.
        arrays['truth'].append(target_for(recording, q))
        samples.append(row)
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    road, roi = recording['road'], recording['roi']
    metrics = {method: measure(arrays[method], arrays['truth'], road, roi) for method in METHODS}
    execution = json.loads((directory / 'execution_summary.json').read_text())
    result = {'run_id': entry['run_id'], 'num_agents': entry['num_agents'],
              'workload_seed': entry['workload_seed'], 'trigger': entry['trigger'],
              'origins': len(clocks), 'metrics': metrics, 'execution_summary': execution,
              'runtime_nonisolated': {name: time_summary([s['timing_s'][name] for s in samples])
                                     for name in METHODS},
              'road_cells': int(road.sum()), 'roi_cells': int(roi.sum()),
              'evaluation_wall_s': time.perf_counter() - started,
              'model_occupancy_mass_mean_by_window': {
                  name: arrays[name][:, :, 0].sum(axis=(2, 3)).mean(axis=0).tolist()
                  for name in METHODS},
              'truth_slow_robot_seconds_mean_by_window': [
                  float(arrays['truth'][:, k, 1, roi].sum(axis=1).mean() * (b-a))
                  for k, (a, b) in enumerate(WINDOWS)]}
    np.savez_compressed(out / 'arrays.npz', **arrays, road=road, roi=roi)
    write_json(out / 'index.json', {'samples': samples, 'protocol': protocol})
    write_json(out / 'metrics.json', result)
    write_json(out / 'complete.json', {'signature': signature, 'artifact_sha256': {
        name: digest(out / name) for name in ('arrays.npz', 'index.json', 'metrics.json')}})
    print('DONE', entry['run_id'], len(clocks), f'{result["evaluation_wall_s"]:.1f}s', flush=True)
    return result


def aggregate(output, rows, protocol, identity):
    groups = []
    for count, trigger in sorted({(r['num_agents'], r['trigger']) for r in rows}):
        selected = [r for r in rows if r['num_agents'] == count and r['trigger'] == trigger]
        values = {name: [] for name in (*METHODS, 'truth')}
        times = {name: [] for name in METHODS}
        road = roi = None
        for row in selected:
            directory = output / row['run_id']
            with np.load(directory / 'arrays.npz') as data:
                for name in values:
                    values[name].append(data[name])
                if road is None:
                    road, roi = data['road'], data['roi']
                elif not np.array_equal(road, data['road']) or not np.array_equal(roi, data['roi']):
                    raise ValueError('Group masks changed across workloads')
            samples = json.loads((directory / 'index.json').read_text())['samples']
            for name in METHODS:
                times[name] += [s['timing_s'][name] for s in samples]
        values = {name: np.concatenate(parts) for name, parts in values.items()}
        groups.append({'num_agents': count, 'trigger': trigger,
                       'run_ids': [r['run_id'] for r in selected],
                       'workload_seeds': [r['workload_seed'] for r in selected],
                       'origins': sum(r['origins'] for r in selected),
                       'metrics': {name: measure(values[name], values['truth'], road, roi)
                                   for name in METHODS},
                       'runtime_nonisolated': {name: time_summary(times[name]) for name in METHODS}})
    result = {'status': 'complete', 'protocol': protocol, 'identity': identity,
              'runs': rows, 'groups': groups,
              'interpretation': 'Frozen models trained on the original 27-robot floor. New bay geometry and fleet workloads are a distribution shift. Whole workload seeds are the independent units; overlapping origins are correlated. Offline errors do not establish forecast-guided routing benefit. Extraction timings are not an isolated hardware benchmark.'}
    write_json(output / 'evaluation.json', result)
    return result


def initialize_worker():
    import torch
    global _WORKER_MODELS
    torch.set_num_threads(2)
    _WORKER_MODELS = load_models()


def evaluate_in_worker(entry, output, protocol, identity):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        return evaluate_run(entry, output, protocol, _WORKER_MODELS, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=REPORT / 'results/fleet_prediction')
    parser.add_argument('--first-origin-s', type=int, default=12)
    parser.add_argument('--last-origin-s', type=int, default=216)
    parser.add_argument('--origin-stride-s', type=int, default=4)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=3,
                        help='Independent recordings; extraction timing is not isolated')
    args = parser.parse_args()
    if args.first_origin_s < 12 or args.origin_stride_s < 1 or args.last_origin_s < args.first_origin_s:
        parser.error('Require first>=12s, last>=first, and positive whole-second stride')
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    protocol = {'first_origin_s': args.first_origin_s, 'last_origin_s': args.last_origin_s,
                'origin_stride_s': args.origin_stride_s, 'truth_dt_s': DT,
                'windows_s': [list(w) for w in WINDOWS], 'methods': list(METHODS),
                'snapshot_boundary': 'Coherent passive native prediction_snapshot on physical observation clock.',
                'fitting': 'None. Reuse all frozen original 27-robot models without tuning.',
                'target': 'Existing occupancy and eligible slow-occupancy fields; unchanged proxy, not a newly selected congestion model.',
                'timing': 'Independent full in-memory queries during extraction; not an isolated latency benchmark.',
                'device': 'cpu', 'torch_threads_per_worker': 2, 'blas_threads_per_worker': 2,
                'extraction_workers': args.jobs}
    frozen_protocol = args.output / 'protocol.json'
    if frozen_protocol.exists():
        study = json.loads(frozen_protocol.read_text())
        expected_origins = list(range(args.first_origin_s, args.last_origin_s + 1,
                                      args.origin_stride_s))
        if (study['methods'] != list(METHODS)
                or study['origin_s'] != expected_origins
                or study['forecast_windows_s'] != protocol['windows_s']
                or study['truth_dt_s'] != DT):
            raise ValueError('CLI extraction settings differ from the frozen study protocol')
        for relative_path, expected in study['model_sha256'].items():
            if digest(REPORT / relative_path) != expected:
                raise ValueError(f'Frozen study model changed: {relative_path}')
    identity = {'source_sha256': source_hashes(), 'model_sha256': model_hashes(),
                'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                'study_protocol_sha256': digest(frozen_protocol) if frozen_protocol.exists() else None}
    entries = []
    for path in args.manifests:
        manifest = json.loads(path.read_text())
        if manifest['status'] != 'complete':
            raise ValueError(f'Recording batch is not complete: {path}')
        for row in manifest['runs']:
            directory = path.parent / row['relative_directory']
            entries.append({**row, 'run_directory': str(directory.resolve())})
    if len({r['run_id'] for r in entries}) != len(entries):
        raise ValueError('Duplicate run IDs across input manifests')
    configurations = [(r['num_agents'], r['trigger'], r['workload_seed']) for r in entries]
    if len(set(configurations)) != len(configurations):
        raise ValueError('Duplicate fleet/rule/workload configurations')
    if frozen_protocol.exists():
        fleets = [study['primary_case']['num_agents'], *study['comparison_fleets']]
        expected = {(n, rule, seed) for n in fleets for rule in study['triggers']
                    for seed in study['workload_seeds']}
        if set(configurations) != expected:
            raise ValueError('Recording manifests do not cover every frozen fleet/rule/workload case exactly once')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'extraction_protocol.json', {'protocol': protocol, 'identity': identity,
                                             'recording_manifests': {str(p): digest(p) for p in args.manifests}})
    if args.jobs == 1:
        models = load_models()
        with threadpool_limits(limits=2):
            rows = [evaluate_run(row, args.output, protocol, models, identity) for row in entries]
    else:
        completed = {}
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize_worker) as pool:
            futures = {pool.submit(evaluate_in_worker, row, args.output, protocol, identity): index
                       for index, row in enumerate(entries)}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        rows = [completed[index] for index in range(len(entries))]
    if model_hashes() != identity['model_sha256']:
        raise ValueError('A frozen model changed during evaluation')
    if source_hashes() != identity['source_sha256']:
        raise ValueError('An evaluator source dependency changed during evaluation')
    aggregate(args.output, rows, protocol, identity)
    print('COMPLETE', len(rows), 'runs', sum(r['origins'] for r in rows), 'origins', flush=True)


if __name__ == '__main__':
    main()
