#!/usr/bin/env python3
"""Native workload-difficulty diagnostics; never a predictor acceptance test.

The original offline and 250-run decision studies are read-only inputs. New
summaries/figures are written only when an explicit output directory is given.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import features
from features import REPORT, pose_array, rasterize
from decision_metrics import task_completions, charge_query_time
from experiments.baseline.analyze import _objects
from experiments.zones.analyze import validate_safety_configuration
from experiments.zones.queue_metrics import THRESHOLDS, build_approaches, measure_approach

SCREEN = {
    'warmup_s': 12.0, 'central_zone_id': 'Z11', 'queue_min_robots': 3,
    'sustained_s': 3.0, 'required_delivery_fraction': 1.0,
    'maximum_native_plan_failures': 0, 'minimum_observed_bypass_opportunities': 1,
    'scope': 'Development workload-difficulty screen, fixed before scoring these pilots; not predictor selection or held-out routing acceptance.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(mask, times, dt, start_s=0.0):
    """Maximal sample-supported intervals; membership may change within a queue."""
    mask = np.asarray(mask, dtype=bool) & (np.asarray(times) >= start_s - 1e-8)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [{'start_s': float(times[a]), 'end_s': float(times[b-1] + dt),
             'duration_s': float((b-a)*dt)} for a, b in zip(starts, stops)]


def in_conflict(positions, zone):
    bounds = zone['conflict_zone']
    return ((positions[..., 0] >= bounds['rows'][0]-.5) &
            (positions[..., 0] < bounds['rows'][1]+.5) &
            (positions[..., 1] >= bounds['cols'][0]-.5) &
            (positions[..., 1] < bounds['cols'][1]+.5))


def graph_distance(scenario, source, target, forbidden=None):
    """Metric shortest path on zone centers, not a committed native robot route."""
    zones = {z['id']: z for z in scenario['zones']}
    if source not in zones or target not in zones or source == forbidden or target == forbidden:
        return None
    graph = {z: [] for z in zones if z != forbidden}
    for a, b in scenario['network']['edges']:
        if a == forbidden or b == forbidden:
            continue
        length = math.dist(zones[a]['center'], zones[b]['center'])
        graph[a].append((b, length)); graph[b].append((a, length))
    import heapq
    queue = [(0.0, source)]; best = {source: 0.0}
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != best[node]:
            continue
        if node == target:
            return distance
        for neighbor, length in graph[node]:
            trial = distance + length
            if trial < best.get(neighbor, math.inf):
                best[neighbor] = trial; heapq.heappush(queue, (trial, neighbor))
    return None


def bypass_info(scenario, source, destination, central):
    direct = graph_distance(scenario, source, destination)
    first = graph_distance(scenario, source, central)
    last = graph_distance(scenario, central, destination)
    bypass = graph_distance(scenario, source, destination, forbidden=central)
    via = None if first is None or last is None else first + last
    return {'source_zone': source, 'destination_zone': destination,
            'shortest_zone_path_m': direct, 'via_central_zone_path_m': via,
            'avoiding_central_zone_path_m': bypass,
            'geometric_bypass_available': bypass is not None,
            'bypass_extra_vs_shortest_m': None if bypass is None or direct is None else bypass-direct,
            'central_is_on_a_shortest_zone_path': via is not None and direct is not None and abs(via-direct) < 1e-8}


def stationary_flags(states, scenario, active, dt):
    """The original stationary-queue detector, with physical-clock task masking."""
    steps = round(THRESHOLDS['stationary_window_s']/dt)
    service = np.unique(np.asarray([g for a in scenario['agents'] for g in a['goals']], float), axis=0)
    flags = np.zeros(active.shape, dtype=bool)
    for t in range(steps, len(states)):
        recent = states[t-steps:t+1]
        delta = recent[:, :, 2] - recent[0, :, 2]
        angular = np.abs(np.arctan2(np.sin(delta), np.cos(delta))).max(axis=0)
        displacement = np.linalg.norm(recent[:, :, :2] - recent[0, :, :2], axis=-1).max(axis=0)
        at_service = (np.linalg.norm(states[t, :, None, :2] - service[None, :, :], axis=-1) <=
                      THRESHOLDS['service_bay_exclusion_radius_m']).any(axis=1)
        flags[t] = active[t] & ~at_service & (displacement <= THRESHOLDS['stationary_displacement_m']) & (angular <= THRESHOLDS['stationary_heading_change_rad'])
    return flags


def queue_summary(counts, warnings, approaches, times, dt, screen):
    central = [i for i, a in enumerate(approaches) if a['target_zone'] == screen['central_zone_id']]
    selected = counts[:, central] if central else np.zeros((len(times), 0), int)
    after = times >= screen['warmup_s'] - 1e-8
    threshold = screen['queue_min_robots']
    rows = {}
    for j, approach in enumerate(approaches):
        intervals = episodes(counts[:, j] >= threshold, times, dt, screen['warmup_s'])
        rows[approach['id']] = {
            'target_zone': approach['target_zone'], 'max_queue_robots': int(counts[:, j].max()),
            'queued_robot_seconds': float(counts[:, j].sum()*dt),
            'threshold_approach_seconds_after_warmup': float(((counts[:, j] >= threshold) & after).sum()*dt),
            'longest_threshold_episode_after_warmup_s': max((e['duration_s'] for e in intervals), default=0.0),
            'threshold_episodes_after_warmup': intervals,
            'spillback_warning_seconds': float(warnings[:, j].sum()*dt),
        }
    central_rows = [row for row in rows.values() if row['target_zone'] == screen['central_zone_id']]
    return {
        'max_queue_robots': int(counts.max(initial=0)),
        'queued_robot_seconds': float(counts.sum()*dt),
        'threshold_approach_seconds_after_warmup': float(((counts >= threshold) & after[:, None]).sum()*dt),
        'any_threshold_seconds_after_warmup': float(((counts >= threshold).any(axis=1) & after).sum()*dt),
        'longest_threshold_episode_after_warmup_s': max((r['longest_threshold_episode_after_warmup_s'] for r in rows.values()), default=0.0),
        'central_max_queue_robots': int(selected.max(initial=0)),
        'central_threshold_approach_seconds_after_warmup': float(((selected >= threshold) & after[:, None]).sum()*dt),
        'central_longest_threshold_episode_after_warmup_s': max((r['longest_threshold_episode_after_warmup_s'] for r in central_rows), default=0.0),
        'spillback_warning_approach_seconds': float(warnings.sum()*dt), 'approaches': rows,
    }


def central_arrivals(states, active, ids, scenario, tasks, times, screen):
    zones = {z['id']: z for z in scenario['zones']}
    central = screen['central_zone_id']
    if central not in zones:
        raise ValueError(f'Missing central measurement zone {central}')
    inside = in_conflict(states[..., :2], zones[central]) & active
    neighbors = {b if a == central else a for a, b in scenario['network']['edges'] if central in (a, b)}
    task_by_id = {t['robot_id']: t for t in tasks}
    agent_by_id = {a['id']: a for a in scenario['agents']}
    rows = []
    for j, robot_id in enumerate(ids):
        indices = np.flatnonzero(inside[:, j])
        if not len(indices):
            continue
        first = int(indices[0]); upstream = None; last_inside = -1
        for zone_id in sorted(neighbors):
            previous = np.flatnonzero(in_conflict(states[:first, j, :2], zones[zone_id]) & active[:first, j])
            if len(previous) and previous[-1] > last_inside:
                last_inside = int(previous[-1]); upstream = zone_id
        task = task_by_id[robot_id]
        source = upstream or task.get('source_zone_id', agent_by_id[robot_id].get('zone_id'))
        destination = task.get('destination_zone_id', agent_by_id[robot_id].get('destination_zone_id'))
        bypass = bypass_info(scenario, source, destination, central)
        # Last observed upstream conflict-cell sample is a geometric timing
        # proxy, not an inferred action-commitment or legal-rerouting deadline.
        exit_s = None if upstream is None else float(times[last_inside])
        opportunity = exit_s is not None and exit_s >= screen['warmup_s'] and bypass['geometric_bypass_available']
        rows.append({'robot_id': robot_id, 'first_central_entry_s': float(times[first]),
                     'last_observed_upstream_junction_s': exit_s, 'upstream_zone': upstream,
                     'post_warmup_observed_geometric_bypass_opportunity': opportunity, **bypass})
    arrival_times = np.asarray([r['first_central_entry_s'] for r in rows])
    after_times = arrival_times[arrival_times >= screen['warmup_s']]
    max_arrivals_5s = max((int(((after_times >= t) & (after_times < t+5)).sum()) for t in after_times), default=0)
    return {'actual_central_entrants': len(rows),
            'entrants_before_warmup': int((arrival_times < screen['warmup_s']).sum()),
            'entrants_after_warmup': len(after_times),
            'max_first_entries_in_5s_after_warmup': max_arrivals_5s,
            'first_entry_s': None if not len(rows) else float(arrival_times.min()),
            'median_first_entry_s': None if not len(rows) else float(np.median(arrival_times)),
            'last_entry_s': None if not len(rows) else float(arrival_times.max()),
            'max_simultaneous_active_robots_in_center': int(inside.sum(axis=1).max()),
            'post_warmup_observed_geometric_bypass_opportunities': sum(r['post_warmup_observed_geometric_bypass_opportunity'] for r in rows),
            'robot_entries': rows,
            'opportunity_scope': 'Zone-graph connectivity and a recorded upstream conflict-region sample after warmup. Does not check native commitments, reservation feasibility, available communication, or benefits of the alternative route.'}, inside.sum(axis=1)


def difficulty_screen(summary, screen):
    feasible = (summary['task_completion_fraction'] >= screen['required_delivery_fraction'] and
                summary['native_plan_failures'] is not None and
                summary['native_plan_failures'] <= screen['maximum_native_plan_failures'])
    opportunity = summary['central_arrivals']['post_warmup_observed_geometric_bypass_opportunities'] >= screen['minimum_observed_bypass_opportunities']
    stationary = summary['stationary_queue']['central_longest_threshold_episode_after_warmup_s'] >= screen['sustained_s'] - 1e-8
    slow = summary['slow_queue']['central_longest_threshold_episode_after_warmup_s'] >= screen['sustained_s'] - 1e-8
    moving = summary['moving_slow_queue']['central_longest_threshold_episode_after_warmup_s'] >= screen['sustained_s'] - 1e-8
    return {'feasible_complete_batch': feasible, 'observed_geometric_diversion_opportunity': opportunity,
            'sustained_stationary_congestion': stationary, 'sustained_slow_congestion': slow,
            'sustained_moving_slow_congestion': moving,
            'candidate_for_stationary_congestion_study': feasible and opportunity and stationary,
            'candidate_for_slow_congestion_study': feasible and opportunity and slow,
            'interpretation': 'Tuning-pilot workload characterization only; no predictor-accuracy or rerouting-utility acceptance claim.'}


def safety_exposure(events, ids, active, slow, dt):
    """Duration-weight selected native blockers; never assign causal delay."""
    count = len(active)
    mode = np.zeros(active.shape, dtype=np.int8)
    kind = np.zeros(active.shape, dtype=np.int8)
    kind_code = {'robot': 1, 'box': 2, 'obstacle': 2}
    for j, robot_id in enumerate(ids):
        changes = sorted((e for e in events if e.get('type') == 'safety_state' and e['agent_id'] == robot_id), key=lambda e: e['clock'])
        for k, event in enumerate(changes):
            start = min(count, max(0, int(event['clock'])))
            stop = min(count, int(changes[k+1]['clock'])) if k+1 < len(changes) else count
            if stop < start:
                raise ValueError('Safety-state transition clocks decrease')
            value = {'clear': 0, 'slow': 1, 'stop': 2}[event['mode']]
            mode[start:stop, j] = value
            kind[start:stop, j] = 0 if value == 0 else kind_code.get(event.get('blocker_kind'), 3)
    by_kind = {}
    for label, value in [('clear', 0), ('robot', 1), ('obstacle', 2), ('unknown', 3)]:
        chosen = kind == value
        by_kind[label] = {'active_robot_seconds': float((chosen & active).sum()*dt),
                          'coincident_slow_robot_seconds': float((chosen & slow).sum()*dt),
                          'whole_recording_robot_seconds': float(chosen.sum()*dt)}
    native_summary = next((e for e in reversed(events) if e.get('type') == 'safety_summary'), None)
    counters = None
    if native_summary:
        keys = ('robot_outer_trigger_ticks', 'obstacle_outer_trigger_ticks', 'polygon_only_trigger_ticks', 'slowed_move_ticks', 'stopped_command_ticks')
        counters = {key.replace('_ticks', '_robot_seconds'): float(sum(r.get(key, 0) for r in native_summary['robots'])*dt) for key in keys}
    return {'by_selected_blocker': by_kind,
            'active_slow_mode_robot_seconds': float(((mode == 1) & active).sum()*dt),
            'active_stop_mode_robot_seconds': float(((mode == 2) & active).sum()*dt),
            'native_whole_recording_counters': counters,
            'scope': 'Selected-blocker intervals use physical safety_state transition clocks over the recorded pose horizon. Native selection visits robots before obstacles at equal severity, so a selected robot may coexist with obstacle overlap. Region exposure does not imply an executed MOVE was slowed. Whole-recording robot/obstacle trigger counters are nonexclusive; slowed_move counters count actual limited MOVE updates but do not identify their cause. Coincidence with the historical speed detector is not causal delay attribution.'}, kind


def summarize_congestion_run(run_directory, metadata=None, *, output_directory=None,
                             warmup_s=12.0, central_zone_id='Z11', sustained_s=3.0):
    """Return JSON-safe difficulty metrics; optional output directory stores traces."""
    run = Path(run_directory); metadata = metadata or {}
    config = json.loads((run/'config.json').read_text())
    scenario = json.loads((run/config['scenario']).read_text())
    map_doc = json.loads((run/scenario['map_file']).read_text())
    tasks = json.loads((run/scenario['tasks_file']).read_text())['tasks']
    events = list(_objects((run/'simulator.log').read_text(errors='replace')))
    _, safety_errors = validate_safety_configuration(events, scenario)
    if safety_errors:
        raise ValueError('Native safety configuration mismatch: ' + '; '.join(safety_errors))
    frames = {e['clock']: e for e in events if e.get('type') == 'tick'}
    if (run/'trajectories.jsonl').exists():
        frames.update({e['clock']: e for line in (run/'trajectories.jsonl').read_text().splitlines() if line.strip() for e in [json.loads(line)]})
    tick_count = config['parameters']['sim_duration']; ticks = config['parameters']['ticks_per_second']; dt = 1/ticks
    if set(frames) != set(range(tick_count)):
        raise ValueError('Difficulty scoring requires the complete configured trajectory, including unfinished tasks')
    frames = [frames[t] for t in range(tick_count)]; times = np.arange(tick_count)*dt; horizon = tick_count*dt
    ids = sorted(a['id'] for a in scenario['agents'])
    if len(set(ids)) != len(ids) or any(sorted(a['id'] for a in f['agents']) != ids for f in frames):
        raise ValueError('Robot identity set changes in native trajectory')
    if sorted(t['robot_id'] for t in tasks) != ids:
        raise ValueError('Expected one finite task per robot')
    states = np.stack([pose_array(f) for f in frames])
    outcomes = task_completions(events, tasks, ticks, horizon)
    by_id = {row['robot_id']: row for row in outcomes}
    active = np.column_stack([(times >= by_id[i]['release_s']) &
        (times < (horizon if by_id[i]['delivery_completed_s'] is None else by_id[i]['delivery_completed_s']+dt)) for i in ids])
    shape = (len(map_doc['layout']), len(map_doc['layout'][0]))
    raster, slow = rasterize(states, scenario, shape, dt)
    del raster
    slow &= active
    stationary = stationary_flags(states, scenario, active, dt)
    approaches = build_approaches(scenario)
    radii = {a['id']: a.get('safety', {}).get('robot_radius_m', scenario['safety']['robot_radius_m']) for a in scenario['agents']}
    counts = {key: np.zeros((tick_count, len(approaches)), dtype=np.int16) for key in ('stationary', 'slow', 'moving_slow')}
    warnings = {key: np.zeros((tick_count, len(approaches)), dtype=bool) for key in counts}
    for t, frame in enumerate(frames):
        stationary_ids = {ids[j] for j in np.flatnonzero(stationary[t])}
        slow_ids = {ids[j] for j in np.flatnonzero(slow[t])}
        for j, approach in enumerate(approaches):
            stopped = measure_approach(approach, frame['agents'], stationary_ids, radii)
            low = measure_approach(approach, frame['agents'], slow_ids, radii)
            counts['stationary'][t, j] = stopped['queued_robots']; warnings['stationary'][t, j] = stopped['spillback_warning']
            counts['slow'][t, j] = low['queued_robots']; warnings['slow'][t, j] = low['spillback_warning']
            if any(i not in stationary_ids for i in low['queued_robot_ids']):
                counts['moving_slow'][t, j] = low['queued_robots']; warnings['moving_slow'][t, j] = low['spillback_warning']
    screen = {**deepcopy(SCREEN), 'warmup_s': float(warmup_s), 'central_zone_id': central_zone_id, 'sustained_s': float(sustained_s)}
    arrivals, central_count = central_arrivals(states, active, ids, scenario, tasks, times, screen)
    completed = sum(row['delivery_completed_s'] is not None for row in outcomes)
    plans = [e for e in events if e.get('type') == 'routing_plan']
    query_path = run/'routing_decisions.jsonl'
    charged = None
    if query_path.exists():
        queries = [json.loads(line) for line in query_path.read_text().splitlines() if line.strip()]
        charge_query_time(outcomes, queries, horizon)
        charged = float(np.mean([row['charged_restricted_flow_s'] for row in outcomes]))
    movement = np.linalg.norm(np.diff(states[:, :, :2], axis=0), axis=-1)
    active_interval = np.column_stack([(times[1:] >= by_id[i]['release_s']) &
        (times[1:] <= (horizon if by_id[i]['delivery_completed_s'] is None else by_id[i]['delivery_completed_s']+dt)+1e-8) for i in ids])
    wait_per_robot = stationary.sum(axis=0)*dt
    after = times >= warmup_s - 1e-8
    safety_attribution, blocker_kind = safety_exposure(events, ids, active, slow, dt)
    summary = {**{k:v for k,v in metadata.items() if k not in ('protocol',)}, 'schema_version': 1, 'study': 'congestion_difficulty_development_pilot',
        'run_directory': str(run.resolve()), 'horizon_s': horizon, 'sample_dt_s': dt, 'fleet_size': len(ids),
        'safety': scenario['safety'], 'trigger': scenario['safety']['robot_trigger'], 'screen_parameters': screen,
        'deliveries_completed': completed, 'deliveries_unfinished': len(tasks)-completed,
        'task_completion_fraction': completed/len(tasks), 'deliveries_per_minute_of_horizon': completed*60/horizon,
        'native_restricted_mean_flow_s': float(np.mean([row['restricted_flow_s'] for row in outcomes])),
        'charged_restricted_mean_flow_s': charged,
        'all_deliveries_completed_s': max(row['delivery_completed_s'] for row in outcomes) if completed == len(tasks) else None,
        'native_plan_failures': sum(not e.get('success', False) for e in plans) if plans else None,
        'travel_distance_m': float((movement*active_interval).sum()),
        'stationary_wait_robot_seconds': float(wait_per_robot.sum()), 'p95_stationary_wait_s': float(np.quantile(wait_per_robot, .95)),
        'slow_robot_seconds': float(slow.sum()*dt), 'slow_robot_seconds_after_warmup': float(slow[after].sum()*dt),
        'active_robot_seconds': float(active.sum()*dt), 'fraction_active_time_slow': float(slow.sum()/active.sum()) if active.any() else None,
        'central_arrivals': arrivals, 'task_outcomes': outcomes, 'safety_exposure': safety_attribution,
        'clock_semantics': 'Task acknowledgements use server clocks; pose metrics use physical observation clocks. Native delivery is observed in pose frames one tick after its server label.',
        'definitions': {
            'slow': 'Exactly features.rasterize: preceding2s path-mean speed<=0.3m/s, heading deviation<=0.1rad, >0.25m from all service/goal locations; additionally active and not delivered.',
            'stationary': 'Existing queue detector: all preceding2s positions within0.1m of window start and heading within0.1rad, >0.2m from service/goals, active.',
            'slow_queue': 'Existing connected approach geometry with slow eligible robots rather than only stationary robots. This is a low-speed group proxy and includes fully stationary groups.',
            'moving_slow_queue': 'A slow connected group with at least one robot not classified stationary. Group count includes its stationary members; no causal congestion attribution.',
            'episodes': 'Consecutive samples above count threshold on one approach; queue membership may change. Post-warmup episodes are truncated at warmup.',
            'spillback': 'Geometric tail warning from existing queue_metrics, not proof upstream traffic is obstructed.',
        },
        'source_hashes': {name: digest(run/name) for name in ('config.json', 'scenario.json', 'map.json', 'tasks.json', 'simulator.log')},
        'analysis_code_sha256': digest(__file__), 'slow_detector_code_sha256': digest(features.__file__),
    }
    for key in counts:
        summary[key+'_queue'] = queue_summary(counts[key], warnings[key], approaches, times, dt, screen)
    summary['difficulty_screen'] = difficulty_screen(summary, screen)
    if output_directory is not None:
        output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
        trace_path = output/'congestion_timeseries.npz'
        deliveries = np.asarray([sum(r['delivery_completed_s'] is not None and t >= r['delivery_completed_s']+dt for r in outcomes) for t in times])
        np.savez_compressed(trace_path, time_s=times, states=states, robot_ids=np.asarray(ids),
            slow_robots=slow.sum(axis=1), stationary_robots=stationary.sum(axis=1), central_occupancy=central_count,
            slow_with_selected_robot=(slow & (blocker_kind == 1)).sum(axis=1),
            slow_with_selected_obstacle=(slow & (blocker_kind == 2)).sum(axis=1),
            deliveries=deliveries, approach_ids=np.asarray([a['id'] for a in approaches]),
            central_approach_mask=np.asarray([a['target_zone'] == central_zone_id for a in approaches]),
            stationary_queue_counts=counts['stationary'], slow_queue_counts=counts['slow'], moving_slow_queue_counts=counts['moving_slow'])
        summary['timeseries_path'] = str(trace_path.resolve())
        (output/'congestion_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def plot_run(summary, output):
    """Actual measured time series, not rendered predictions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.load(summary['timeseries_path']); t = data['time_s']; central = data['central_approach_mask']
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(t, data['slow_robots'], label='Slow active robots'); axes[0].plot(t, data['stationary_robots'], label='Stationary active robots')
    axes[0].set_ylabel('Robot count'); axes[0].legend(fontsize=8, ncol=2)
    for key, label in [('stationary', 'Stationary queue'), ('slow', 'Slow group'), ('moving_slow', 'Slow group with motion')]:
        axes[1].plot(t, data[key+'_queue_counts'][:, central].max(axis=1), label=label)
    axes[1].axhline(3, color='gray', ls='--', lw=.8); axes[1].set_ylabel('Largest center\napproach group'); axes[1].legend(fontsize=8, ncol=3)
    axes[2].plot(t, data['deliveries'], label='Completed deliveries'); axes[2].plot(t, data['central_occupancy'], label='Active robots inside center')
    axes[2].set_ylabel('Robot / task count'); axes[2].set_xlabel('Native physical observation time (s)'); axes[2].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.axvline(summary['screen_parameters']['warmup_s'], color='black', ls=':', lw=.8); ax.grid(alpha=.2)
    fig.suptitle(f"{summary.get('variant', Path(summary['run_directory']).name)} | {summary['fleet_size']} robots | {summary['trigger']}")
    fig.savefig(Path(output)/'congestion_timeline.pdf'); fig.savefig(Path(output)/'congestion_timeline.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='*', type=Path); parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--plot', action='store_true')
    args = parser.parse_args(); entries = [{'run_directory': str(p), 'run_id': p.name} for p in args.runs]
    if args.manifest:
        entries.extend(json.loads(args.manifest.read_text())['runs'])
    if not entries:
        parser.error('Supply run directories or --manifest')
    summaries = []
    for row in entries:
        output = args.output/row.get('run_id', Path(row['run_directory']).name)
        summary = summarize_congestion_run(row['run_directory'], row, output_directory=output)
        if args.plot: plot_run(summary, output)
        summaries.append(summary)
        print(json.dumps({'run': row.get('run_id'), 'screen': summary['difficulty_screen']}, allow_nan=False), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'difficulty_summary.json').write_text(json.dumps({'scope': SCREEN['scope'], 'runs': summaries}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
