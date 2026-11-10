"""Counterfactual search over a small set of temporary junction admissions.

At each fresh post-planner snapshot, compare doing nothing with at most eight
two-second admission interventions. Every rollout uses the same current
accepted actions, controllers and safety model; no future native truth or
future PBS plans are available. The selected intervention expires after two
seconds, although its predicted consequences are examined over five or ten.

The objective is an explicit progress proxy, not predicted deliveries. Let
w_i(k) be cumulative accepted-action work advanced by robot i at step k:
signed MOVE distance / nominal speed, reduction toward an accepted TURN
target / nominal turn rate, plus actual SERVICE time consumed. For K=H/0.1,
score = sum_{k=0}^{K-1} sum_i w_i(k) / K. Its unit is fleet nominal work-seconds.
It rewards earlier advancement of all robots, including the robots held.
Waiting earns no credit. Exhausting a short accepted prefix does not imply
task completion and earns no bonus for unknown remaining task stages.

Do nothing wins ties and any improvement not exceeding the configured gain
margin. The default is an absolute 0.5 work-second margin; a global relative
margin defaults to zero so adding distant robots does not bury a local gain.
These are exploratory fixed choices, not optimized or validated thresholds.
Consecutive requested holds are capped at six seconds followed by two seconds
of release to avoid repeatedly postponing the same promised release. This
does not guarantee progress, bounded total waiting, or freedom from deadlock.
"""
from __future__ import annotations

import math
import time

import numpy as np

from features import observation_clock, pose_array
from experiments.prediction.admission import contains, enters
from experiments.prediction.joint import rollout


POLICY = {
    'version': 'search-admission-v1',
    'center': [30., 30.],
    'half_width': 3.,
    'dt_s': .1,
    'execution_interval_s': 2.,
    'interval_ticks': 20,
    'ticks_per_second': 10,
    'candidate_remaining_move_distance_m': 5.,
    'max_candidates_including_noop': 9,
    'max_continuous_hold_ticks': 60,
    'forced_release_ticks': 20,
    'absolute_gain_s': .5,
    'relative_gain_fraction': 0.,
    'default_horizon_s': 5.,
    'allowed_horizons_s': [5., 10.],
    'protect_clearance': False,
    'objective': 'time_average_all_robot_cumulative_accepted_nominal_work_s',
    'task_completion_inferred': False,
    'future_native_planning_modeled': False,
    'native_poll_before_completion': True,
    'parameters_validated': False,
}
APPROACHES = ('north', 'east', 'south', 'west')
_EPS = 1e-9


def _inside(point):
    return contains(point, POLICY['center'], POLICY['half_width'])


def _approach(start):
    row, col = np.asarray(start, dtype=float) - POLICY['center']
    if abs(row) >= abs(col):
        return 'north' if row < 0 else 'south'
    return 'west' if col < 0 else 'east'


def eligible_entries(event):
    """Outside robots with a nearby accepted entry beyond both commitment cuts."""
    robot_list = event['snapshot']['robots']
    pose_list = event['pose_frame']['agents']
    robots = {int(r['id']): r for r in robot_list}
    poses = {int(a['id']): np.array([-a['x'], -a['y']], dtype=float) for a in pose_list}
    if len(robots) != len(robot_list) or len(poses) != len(pose_list) or robots.keys() != poses.keys():
        raise ValueError('Snapshot and pose robot IDs must be unique and identical')
    entries = {}
    excluded = {'inside': [], 'committed_or_queued_entry': [], 'no_nearby_accepted_entry': []}
    for rid in sorted(robots):
        robot, position = robots[rid], poses[rid]
        if not np.isfinite(position).all():
            raise ValueError('Nonfinite current robot position')
        if _inside(position):
            excluded['inside'].append(rid)
            continue
        cut = max(int(robot.get('committed_node_id', -1)), int(robot.get('enqueued_node_id', -1)))
        actions = sorted((a for a in robot['actions'] if int(a['node_id']) > int(robot['finished_node_id'])),
                         key=lambda a: int(a['node_id']))
        distance, first_move, found = 0., True, False
        for action in actions:
            if action['type'] != 'M':
                continue
            start, goal = np.asarray(action['start'], dtype=float), np.asarray(action['goal'], dtype=float)
            if start.shape != (2,) or goal.shape != (2,) or not np.isfinite([start, goal]).all():
                raise ValueError('Invalid accepted MOVE geometry')
            distance += float(np.linalg.norm(goal - (position if first_move else start)))
            first_move = False
            if not enters(start, goal, POLICY['center'], POLICY['half_width']):
                continue
            found = True
            if int(action['node_id']) <= cut:
                excluded['committed_or_queued_entry'].append(rid)
            elif distance <= POLICY['candidate_remaining_move_distance_m'] + _EPS:
                entries[rid] = {'entry_node_id': int(action['node_id']),
                                'remaining_move_distance_m': distance,
                                'approach': _approach(start)}
            else:
                excluded['no_nearby_accepted_entry'].append(rid)
            break
        if not found:
            excluded['no_nearby_accepted_entry'].append(rid)
    return entries, excluded


def candidate_interventions(entries, forced_release_ids=()):
    """No-op, nearest single on each approach, then each whole approach."""
    forced = set(forced_release_ids)
    groups = {side: sorted((rid for rid, detail in entries.items()
                            if detail['approach'] == side and rid not in forced),
                           key=lambda rid: (entries[rid]['remaining_move_distance_m'], rid))
              for side in APPROACHES}
    candidates = [{'name': 'noop', 'held_robot_ids': []}]
    seen = {()}
    for kind in ('single', 'approach'):
        for side in APPROACHES:
            members = groups[side][:1] if kind == 'single' else groups[side]
            ids = tuple(sorted(members))
            if not ids or ids in seen:
                continue
            seen.add(ids)
            candidates.append({'name': f'{kind}_{side}', 'held_robot_ids': list(ids)})
    if len(candidates) > POLICY['max_candidates_including_noop']:
        raise AssertionError('Candidate enumeration exceeded the fixed search budget')
    return candidates


def progress_score(result, robot_ids, horizon_s):
    """Score all IDs on the declared half-open grid; never infer completed tasks."""
    frames = result['frames']
    offsets = np.array([frame['offset_s'] for frame in frames], dtype=float)
    expected = np.arange(round(horizon_s / POLICY['dt_s']) + 1) * POLICY['dt_s']
    if offsets.shape != expected.shape or not np.allclose(offsets, expected, atol=1e-9, rtol=0):
        raise ValueError('Counterfactual rollout returned an unexpected time grid')
    values = []
    ids = sorted(robot_ids)
    for frame in frames:
        agents = sorted(frame['agents'], key=lambda agent: agent['id'])
        if [agent['id'] for agent in agents] != ids:
            raise ValueError('Every counterfactual frame must include every robot exactly once')
        values.append([agent['accepted_work_done_s'] for agent in agents])
    work = np.asarray(values, dtype=float)
    if not np.isfinite(work).all() or not np.allclose(work[0], 0., atol=1e-9, rtol=0):
        raise ValueError('Accepted-work progress must be finite and start at zero')
    return {'score_work_s': float(work[:-1].sum(axis=1).mean()),
            'terminal_work_s': float(work[-1].sum()),
            'terminal_work_s_by_robot': {str(rid): float(value) for rid, value in zip(ids, work[-1])}}


class SearchAdmissionPolicy:
    """Stateful receding-horizon search; create one instance per native run."""

    def __init__(self, horizon_s=5., absolute_gain_s=.5, relative_gain_fraction=0., tasks=None):
        if horizon_s not in POLICY['allowed_horizons_s']:
            raise ValueError('Use the frozen five- or ten-second prediction horizon')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in (absolute_gain_s, relative_gain_fraction)):
            raise ValueError('Gain margins must be finite nonnegative numbers')
        self.horizon_s = float(horizon_s)
        self.absolute_gain_s, self.relative_gain_fraction = float(absolute_gain_s), float(relative_gain_fraction)
        # Original tasks may be retained by the runner, but this proxy does not
        # claim to reconstruct completed stages from locations or plan endings.
        self.tasks = tasks
        self.last_clock = None
        self.held_since = {}
        self.release_until = {}
        self.decision_count = 0
        self.intervention_count = 0

    def decide(self, event, past, scenario, map_doc, method='search'):
        started = time.perf_counter()
        accepted_methods = ('search', f'search_h{int(self.horizon_s)}')
        if method not in accepted_methods:
            raise ValueError('Search method must match the configured prediction horizon')
        if event.get('boundary') != 'post_planner_before_dispatch':
            raise ValueError('Search requires the fresh post-planner, pre-dispatch snapshot')
        observation = observation_clock(event)
        if not event.get('controllers'):
            raise ValueError('Search counterfactuals require current native controllers')
        clock = event['clock']
        if type(clock) is not int or clock < 0 or (self.last_clock is not None and clock <= self.last_clock):
            raise ValueError('Decision clocks must be strictly increasing nonnegative ticks')
        if past is not None:
            history = np.asarray(past)
            if history.ndim != 3 or not len(history) or not np.allclose(history[-1], pose_array(event['pose_frame']), atol=1e-7, rtol=0):
                raise ValueError('Optional history must end at the current observation')
        held_since = dict(self.held_since)
        if self.last_clock is not None and clock - self.last_clock > POLICY['interval_ticks']:
            held_since.clear()
        release_until = {rid: until for rid, until in self.release_until.items() if until > clock}
        for rid, since in held_since.items():
            if clock - since >= POLICY['max_continuous_hold_ticks']:
                release_until[rid] = clock + POLICY['forced_release_ticks']
        forced = set(release_until)
        entries, exclusions = eligible_entries(event)
        candidates = candidate_interventions(entries, forced)
        ids = [int(robot['id']) for robot in event['snapshot']['robots']]
        evaluations = []
        for candidate in candidates:
            admission = None
            if candidate['held_robot_ids']:
                admission = {'center': list(POLICY['center']), 'half_width': POLICY['half_width'],
                             'held_robot_ids': candidate['held_robot_ids'],
                             'until_offset_s': POLICY['execution_interval_s'], 'protect_clearance': False}
            prediction_started = time.perf_counter()
            prediction = rollout(event['snapshot'], event['pose_frame'], scenario, map_doc,
                                 horizon_s=self.horizon_s, dt=POLICY['dt_s'],
                                 controllers=event['controllers'], admission=admission, record_progress=True,
                                 native_poll_before_completion=POLICY['native_poll_before_completion'])
            score = progress_score(prediction, ids, self.horizon_s)
            gate = prediction.get('diagnostics', {}).get('admission', {})
            evaluations.append({**candidate, **score,
                                'rollout_wall_s': time.perf_counter() - prediction_started,
                                'predicted_dispatch_block_events': int(gate.get('dispatch_block_events', 0)),
                                'predicted_empty_queue_wait_robot_s': float(gate.get('empty_queue_wait_robot_seconds', 0.)),
                                'predicted_commitment_bypasses': int(gate.get('commitment_prerequisite_bypasses', 0)),
                                'predicted_dispatch_denials': gate.get('events', [])})
        baseline = evaluations[0]['score_work_s']
        margin = max(self.absolute_gain_s, self.relative_gain_fraction * abs(baseline))
        best = evaluations[0]
        for candidate in evaluations:
            gain = candidate['score_work_s'] - baseline
            candidate['gain_over_noop_s'] = gain
            candidate['relative_gain_fraction'] = gain / abs(baseline) if abs(baseline) > _EPS else None
            if candidate['score_work_s'] > best['score_work_s'] + _EPS:
                best = candidate
        best_gain = best['score_work_s'] - baseline
        selected = best if best_gain > margin + _EPS else evaluations[0]
        held = selected['held_robot_ids']
        self.held_since = {rid: held_since.get(rid, clock) for rid in held}
        self.release_until = release_until
        self.last_clock = clock
        self.decision_count += 1
        self.intervention_count += bool(held)
        payload = {'center': list(POLICY['center']), 'half_width': POLICY['half_width'],
                   'held_robot_ids': list(held), 'until_clock': clock + POLICY['interval_ticks'],
                   'protect_clearance': False}
        diagnostics = {
            'policy_version': POLICY['version'], 'method': method, 'clock': clock,
            'observation_clock': observation, 'horizon_s': self.horizon_s,
            'native_poll_before_completion': POLICY['native_poll_before_completion'],
            'execution_interval_s': POLICY['execution_interval_s'], 'objective': POLICY['objective'],
            'task_completion_inferred': False, 'candidate_count': len(evaluations),
            'candidate_ids': sorted(entries), 'candidate_entries': {str(rid): detail for rid, detail in entries.items()},
            'excluded_robot_ids': exclusions, 'forced_release_robot_ids': sorted(forced),
            'applied_hold_since_clock': {str(rid): since for rid, since in sorted(self.held_since.items())},
            'release_until_clock': {str(rid): until for rid, until in sorted(release_until.items())},
            'candidates': evaluations, 'noop_score_work_s': baseline,
            'absolute_gain_margin_s': self.absolute_gain_s, 'relative_gain_margin_fraction': self.relative_gain_fraction,
            'required_gain_s': margin, 'best_candidate': best['name'], 'best_gain_s': best_gain,
            'selected_candidate': selected['name'], 'selected_gain_s': selected['gain_over_noop_s'],
            'held_robot_ids': list(held), 'intervened': bool(held),
            'selection_reason': 'gain_exceeds_margin' if held else 'noop_best_tied_or_below_margin',
            'decision_count': self.decision_count, 'intervention_count': self.intervention_count,
            'choice_fraction': self.intervention_count / self.decision_count,
            'noop_fraction': 1. - self.intervention_count / self.decision_count,
            'policy_wall_s': time.perf_counter() - started,
            'assumptions': ['Current accepted plans remain fixed inside each rollout; future PBS replanning is not modeled.',
                            'Only native-compatible admission changes; controller queues and commitments are respected.',
                            'Polling precedes completion publication; exact interleaved native robot callback order remains approximate.',
                            'Progress is known-action work, not a predicted delivery or full remaining-task cost.',
                            'Six-second request cap prevents indefinite consecutive requests, not all indirect waiting.'],
        }
        return payload, diagnostics
