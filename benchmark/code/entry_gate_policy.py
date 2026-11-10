"""Provisional central-junction admission baseline; no routes are changed.

Inspired by virtual traffic lights (https://arxiv.org/html/2511.07811v1)
and intersection reservations (Dresner & Stone, 2008,
https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/JAIR08-dresner.pdf).
This is a deliberately simpler adaptation, not a reproduction or a safety
guarantee. Thresholds below are fixed before experiments and are unvalidated.

Reactive and forecast modes use exactly the same admission rule. Only the
pressure estimate differs: current slow robots versus their mean number over
the next five seconds in a joint kinematic rollout. Both use the report's
causal slow-robot detector. For control pressure, robot centers must actually
lie inside the square: raster rounding would count a robot held just outside
as inside, creating feedback from the gate's own queue. Evaluation retains
the common raster-based slow-occupancy metric without this geometric change.

The native controller enforces a selected robot's gate on outside,
uncommitted MOVEs that decrease L1 distance to the square or enter it. This
can hold dispatch upstream, before the boundary entry has become committed.
Committed/queued actions and their prerequisites must still be allowed even
if commitment advances after this decision. The default bounded mode does
not exempt prerequisites of an interior robot's uncommitted future exit:
that robot may therefore be delayed indirectly. Strict clearance protection
is retained as an explicit diagnostic option. Neither mode guarantees
progress or freedom from deadlock. Requested holds need not become actual
holds; the six-second cap bounds requests, not total delay. A decision
expires after one two-second planning interval.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from features import DT, observation_clock, pose_array, rasterize, rollout_array
from experiments.prediction.joint import rollout


POLICY = {
    'version': 'central-entry-gate-v3',
    'center': [30.0, 30.0],
    'half_width': 3.0,
    'interval_ticks': 20,
    'ticks_per_second': 10,
    'forecast_horizon_s': 5.0,
    'forecast_dt_s': 0.1,
    'candidate_remaining_move_distance_m': 5.0,
    'slow_pressure_threshold_robots': 0.5,
    'max_continuous_hold_ticks': 60,
    'forced_release_ticks': 20,
    'protect_uncommitted_clearance': False,
    'dispatch_scope': 'outside_uncommitted_move_decreasing_L1_distance_or_entering_square',
    'selection': 'oldest_candidate_then_shortest_remaining_distance_then_id',
    'admission': 'clear_inside_and_committed_entrants_first_otherwise_admit_oldest_candidate',
    'pressure_geometry': 'continuous_robot_centers_inside_inclusive_square',
    'parameters_validated': False,
}
METHODS = ('none', 'reactive', 'forecast')


def _inside(points):
    points = np.asarray(points, dtype=float)
    return np.all(np.abs(points - POLICY['center']) <= POLICY['half_width'] + 1e-9, axis=-1)


def _pending_actions(robot):
    return sorted((a for a in robot['actions'] if int(a['node_id']) > int(robot['finished_node_id'])),
                  key=lambda a: int(a['node_id']))


def _entry(actions, position):
    """First accepted outside-to-inside MOVE and remaining MOVE distance.

    Native accepted plans consist of short segments, so entry is a MOVE with
    its goal inside and its start outside. The first segment's distance uses
    the observed position, not the already executed part of its planned start.
    Turns and service do not add distance; this is a proximity filter, not ETA.
    """
    distance = 0.0
    first_move = True
    for action in actions:
        if action['type'] != 'M':
            continue
        start = np.asarray(action['start'], dtype=float)
        goal = np.asarray(action['goal'], dtype=float)
        distance += float(np.linalg.norm(goal - (position if first_move else start)))
        first_move = False
        if not _inside(start) and _inside(goal):
            return action, distance
    return None, None


def _clearing_actions(actions, currently_inside):
    """Actions needed through the first exit, including a committed entry."""
    entered = bool(currently_inside)
    selected = []
    for action in actions:
        selected.append(action)
        if action['type'] != 'M':
            continue
        start_inside, goal_inside = bool(_inside(action['start'])), bool(_inside(action['goal']))
        entered = entered or start_inside or goal_inside
        if entered and not goal_inside:
            break
    return selected


def _protected_robots(robots, actions_by_id, inside_ids, committed_entry_ids):
    """Protect clearers and transitive ADG prerequisites through their exit.

    A prerequisite node also needs its own robot's preceding unfinished nodes.
    Following that prefix protects observed dependencies that a direct-only
    check would miss. Missing unfinished dependency nodes trigger
    a conservative release of all robots, rather than guessing that it is safe.
    """
    protected = set(inside_ids) | set(committed_entry_ids)
    roots = set(protected)
    seen = set()
    stack = []
    unknown = []
    for rid in roots:
        stack.extend((rid, int(a['node_id'])) for a in
                     _clearing_actions(actions_by_id[rid], rid in inside_ids))
    while stack:
        rid, node_id = stack.pop()
        if (rid, node_id) in seen:
            continue
        seen.add((rid, node_id))
        if rid not in robots:
            unknown.append([rid, node_id])
            continue
        if node_id <= int(robots[rid]['finished_node_id']):
            continue
        prefix = [a for a in actions_by_id[rid] if int(a['node_id']) <= node_id]
        if not any(int(a['node_id']) == node_id for a in prefix):
            unknown.append([rid, node_id])
            continue
        protected.add(rid)
        for action in prefix:
            stack.extend((int(dep[0]), int(dep[1])) for dep in action.get('dependencies', []))
    if unknown:
        protected.update(robots)
    return protected, sorted(unknown)


class EntryGatePolicy:
    """One instance per run; state records age and bounded hold requests.

    ``protect_uncommitted_clearance=True`` opts into the old strict host and
    native future-exit protection. It is not the v3 comparison condition.
    """

    def __init__(self, *, protect_uncommitted_clearance=None):
        if protect_uncommitted_clearance is None:
            protect_uncommitted_clearance = POLICY['protect_uncommitted_clearance']
        if type(protect_uncommitted_clearance) is not bool:
            raise ValueError('Clearance protection must be a boolean')
        self.protect_uncommitted_clearance = protect_uncommitted_clearance
        self.first_candidate_clock = {}
        self.held_since = {}
        self.release_until = {}
        self.last_clock = None
        self.method = None

    def decide(self, event, past, scenario, map_doc, method):
        """Return a native ``entry_gate`` object and JSON-serializable audit.

        The runner owns the 12-second warmup. ``past`` ends at the current pose
        and contains at least 21 frames at 0.1 seconds. No future truth is read.
        """
        started = time.perf_counter()
        if method not in METHODS:
            raise ValueError(f'Unknown entry-gate method: {method!r}')
        clock = int(event['clock'])
        if self.last_clock is not None and clock <= self.last_clock:
            raise ValueError('Gate decisions must have strictly increasing server clocks')
        if self.method is not None and method != self.method:
            raise ValueError('Use a separate policy instance for each method/run')
        # A missed decision expires in native code; it breaks continuous holding.
        if self.last_clock is not None and clock - self.last_clock > POLICY['interval_ticks']:
            self.held_since.clear()
        self.last_clock, self.method = clock, method
        payload = {'center': list(POLICY['center']), 'half_width': POLICY['half_width'],
                   'held_robot_ids': [], 'until_clock': clock + POLICY['interval_ticks'],
                   'protect_clearance': self.protect_uncommitted_clearance}
        diagnostics = {'policy_version': POLICY['version'], 'method': method, 'clock': clock,
                       'protect_uncommitted_clearance': self.protect_uncommitted_clearance,
                       'observation_clock': observation_clock(event), 'candidate_ids': [],
                       'held_robot_ids': [], 'admitted_robot_ids': [], 'protected_robot_ids': [],
                       'forced_release_robot_ids': [], 'pressure_robots': None,
                       'current_pressure_robots': None, 'pressure_triggered': False,
                       'forecast_wall_s': 0.0, 'unknown_dependencies': []}
        if method == 'none':
            diagnostics['policy_wall_s'] = time.perf_counter() - started
            return payload, diagnostics

        robots = {int(r['id']): r for r in event['snapshot']['robots']}
        ids = sorted(robots)
        poses = pose_array(event['pose_frame'])
        pose_ids = sorted(int(a['id']) for a in event['pose_frame']['agents'])
        states = np.asarray(past, dtype=float)
        if (pose_ids != ids or states.ndim != 3 or states.shape[1:] != (len(ids), 3)
                or len(states) < 21 or not np.isfinite(states).all()
                or not np.allclose(states[-1], poses, atol=1e-7, rtol=0)):
            raise ValueError('History must end at the current sorted robot poses and include two seconds')
        shape = (len(map_doc['layout']), len(map_doc['layout'][0]))
        _, eligible = rasterize(states[-21:], scenario, shape, dt=DT)
        current_pressure = float(np.sum(eligible[-1] & _inside(states[-1, :, :2])))
        diagnostics['current_pressure_robots'] = current_pressure
        pressure = current_pressure
        if method == 'forecast':
            if not event.get('controllers'):
                raise ValueError('Joint forecast requires current native controller state')
            forecast_start = time.perf_counter()
            result = rollout(event['snapshot'], event['pose_frame'], scenario, map_doc,
                             horizon_s=POLICY['forecast_horizon_s'], dt=DT,
                             controllers=event['controllers'])
            offsets, future = rollout_array(result)
            expected = np.arange(round(POLICY['forecast_horizon_s'] / DT) + 1) * DT
            if offsets.shape != expected.shape or not np.allclose(offsets, expected):
                raise ValueError('Joint forecast returned an unexpected time grid')
            if future.shape != (len(expected), len(ids), 3) or not np.isfinite(future).all():
                raise ValueError('Joint forecast returned invalid robot states')
            combined = np.concatenate([states[-21:-1], future])
            _, future_eligible = rasterize(combined, scenario, shape, dt=DT)
            # Half-open [0,5), same convention used for report target windows.
            counts = np.sum(future_eligible[20:-1] & _inside(future[:-1, :, :2]), axis=1)
            pressure = float(np.mean(counts))
            diagnostics['forecast_wall_s'] = time.perf_counter() - forecast_start
            diagnostics['forecast_pressure_by_second'] = [float(v) for v in counts.reshape(5, 10).mean(axis=1)]
        diagnostics['pressure_robots'] = pressure
        triggered = pressure >= POLICY['slow_pressure_threshold_robots']
        diagnostics['pressure_triggered'] = bool(triggered)

        actions_by_id = {rid: _pending_actions(robots[rid]) for rid in ids}
        inside_ids = {rid for rid, pose in zip(ids, poses) if _inside(pose[:2])}
        committed_entry_ids = set()
        candidates = {}
        for rid, pose in zip(ids, poses):
            if rid in inside_ids:
                continue
            action, distance = _entry(actions_by_id[rid], pose[:2])
            if action is None:
                continue
            cut = max(int(robots[rid].get('committed_node_id', -1)),
                      int(robots[rid].get('enqueued_node_id', -1)))
            if int(action['node_id']) <= cut:
                committed_entry_ids.add(rid)
            elif distance <= POLICY['candidate_remaining_move_distance_m'] + 1e-9:
                candidates[rid] = {'entry_node_id': int(action['node_id']),
                                   'remaining_move_distance_m': float(distance)}
        if self.protect_uncommitted_clearance:
            protected, unknown = _protected_robots(robots, actions_by_id, inside_ids, committed_entry_ids)
        else:
            # Native dispatch separately protects the latest queued/committed
            # requirements. Uncommitted future-exit ancestors are deliberately
            # controllable in the bounded experimental mode.
            protected, unknown = inside_ids | committed_entry_ids, []
        diagnostics['inside_robot_ids'] = sorted(inside_ids)
        diagnostics['committed_entry_robot_ids'] = sorted(committed_entry_ids)
        diagnostics['protected_robot_ids'] = sorted(protected)
        diagnostics['unknown_dependencies'] = unknown
        diagnostics['candidate_ids'] = sorted(candidates)
        diagnostics['candidate_details'] = {str(rid): detail for rid, detail in sorted(candidates.items())}
        controllable = set(candidates) - protected
        self.first_candidate_clock = {rid: self.first_candidate_clock.get(rid, clock)
                                      for rid in controllable}
        self.release_until = {rid: until for rid, until in self.release_until.items() if until > clock}
        forced = {rid for rid in controllable if self.release_until.get(rid, -1) > clock}
        for rid in controllable:
            if rid in self.held_since and clock - self.held_since[rid] >= POLICY['max_continuous_hold_ticks']:
                self.release_until[rid] = clock + POLICY['forced_release_ticks']
                forced.add(rid)
        ordered = sorted(controllable, key=lambda rid: (self.first_candidate_clock[rid],
                         candidates[rid]['remaining_move_distance_m'], rid))
        # An existing clearing group already has admission. Prioritize it over
        # more entrants, without promising it can move: its uncommitted future
        # actions may depend on a held robot. If the square is currently empty
        # and only the forecast predicts pressure, admit one candidate.
        admitted = set(forced)
        if ordered and not (inside_ids or committed_entry_ids):
            admitted.add(ordered[0])
        held = (controllable - admitted) if triggered else set()
        self.held_since = {rid: self.held_since.get(rid, clock) for rid in held}
        diagnostics['forced_release_robot_ids'] = sorted(forced)
        diagnostics['admitted_robot_ids'] = sorted(set(candidates) - held)
        diagnostics['held_robot_ids'] = sorted(held)
        diagnostics['continuous_hold_s'] = {str(rid): (clock - since) / POLICY['ticks_per_second']
                                             for rid, since in sorted(self.held_since.items())}
        payload['held_robot_ids'] = sorted(held)
        diagnostics['policy_wall_s'] = time.perf_counter() - started
        return deepcopy(payload), diagnostics
