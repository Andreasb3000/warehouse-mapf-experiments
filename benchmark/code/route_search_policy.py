"""Choose among actual PBS preview plans using a common, forecast-proposed menu.

At each review, zero/mild/strong destination-cell costs produce alternative
joint routes from the same native commitment cut. Both nontrivial selectors
use the same 20-second slow-traffic forecast to propose costs. ``nominal``
ranks the resulting routes by free-flow remaining service ETA; ``rollout``
uses a 30-second joint forecast with a free-flow tail. Thus this comparison
compares joint safety/dependency ETA with nominal free-flow ETA; both rank
actual candidate plans and both already use a forecast to propose costs.

For candidate k, J_k is fleet-mean remaining ETA to the identical pending
terminal service action. Select argmin J_k only when J_0 - J_k > 0.5 s;
otherwise retain the zero-cost plan. Completion and route suffix validity are
checked by route_eta. A failed zero-cost solve retains native fallback.

This is a small receding-horizon search, not an optimal-control guarantee.
Rollouts keep candidate plans fixed, omit future PBS solves, and use optimistic
free-flow work after their horizon. Native final outcomes use simulator time;
computation is measured separately and is not added to that clock.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
from scipy.ndimage import maximum_filter

from features import observation_clock, pose_array, predicted_rasters, window_mean
from experiments.prediction.joint import rollout
from route_eta import score_route_eta, service_targets


METHODS = ('none', 'nominal', 'rollout')
POLICY = {
    'version': 'route-cost-search-v1',
    'refresh_s': 2.,
    'history_warmup_s': 12.,
    'forecast_horizon_s': 20.,
    'eta_horizon_s': 30.,
    'rollout_dt_s': .1,
    'cost_weights': [0., 2., 8.],
    'window_weights': [.5, .3, .2],
    'spatial_max_filter_cells': 3,
    'max_cell_penalty': 1000.,
    'minimum_mean_eta_gain_s': .5,
    'gain_comparison': 'strictly greater; zero wins ties and insufficient gains',
    'scope': 'all traversable map cells; no manually selected corridor',
    'objective': 'fleet-mean remaining ETA to common terminal service',
    'compute_in_native_outcome': False,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _method(method):
    if method not in METHODS:
        raise ValueError(f'Unknown route-search method: {method}')


def road_mask(map_doc):
    layout = map_doc['layout']
    if not layout or not layout[0] or any(len(row) != len(layout[0]) for row in layout):
        raise ValueError('Map layout must be nonempty and rectangular')
    return np.array([[cell not in '@T' for cell in row] for row in layout], dtype=bool)


def forecast_cost_menu(prediction, road):
    """The existing three-window, 3x3 slow-field adapter, at three fixed weights.

PBS adds these nonnegative values to MOVE destination costs; WAIT retains its
normal base cost. Duplicate fields are retained to keep the three-plan menu
identical for nominal and rollout selection, and are explicitly diagnosed.
"""
    road = np.asarray(road, dtype=bool)
    prediction = np.asarray(prediction, dtype=float)
    if (road.ndim != 2 or prediction.shape != (3, 2, *road.shape)
            or not np.isfinite(prediction).all() or np.any(prediction < 0)):
        raise ValueError('Expected finite nonnegative three-window occupancy/slow fields')
    slow = np.tensordot(POLICY['window_weights'], prediction[:, 1], axes=(0, 0))
    expanded = maximum_filter(slow, size=POLICY['spatial_max_filter_cells'],
                              mode='constant', cval=0.)
    return [{'name': name, 'cell_penalties':
             (np.minimum(POLICY['max_cell_penalty'], weight * expanded) * road).ravel().tolist()}
            for name, weight in zip(('zero', 'mild', 'strong'), POLICY['cost_weights'])]


def _cost_metadata(candidates):
    result = []
    for index, candidate in enumerate(candidates):
        penalties = np.asarray(candidate['cell_penalties'])
        result.append(dict(index=index, name=candidate['name'],
                           nonzero_penalty_cells=int(np.count_nonzero(penalties)),
                           max_cell_penalty=float(penalties.max()),
                           sum_cell_penalties=float(penalties.sum()),
                           penalty_sha256=_digest(candidate['cell_penalties'])))
    return result


def _score_checked(score):
    eta = float(score['mean_eta_s'])
    if not math.isfinite(eta) or eta < 0:
        raise ValueError('Candidate ETA must be finite and nonnegative')
    # Enforce auditable JSON values before accepting an intervention.
    json.dumps(score, allow_nan=False)
    return eta


class RouteSearchPolicy:
    def __init__(self, minimum_mean_eta_gain_s=POLICY['minimum_mean_eta_gain_s'],
                 eta_horizon_s=POLICY['eta_horizon_s']):
        if (not math.isfinite(minimum_mean_eta_gain_s) or minimum_mean_eta_gain_s < 0
                or not math.isfinite(eta_horizon_s) or eta_horizon_s <= 0):
            raise ValueError('Margin must be finite/nonnegative and ETA horizon positive')
        self.minimum_mean_eta_gain_s = float(minimum_mean_eta_gain_s)
        self.eta_horizon_s = float(eta_horizon_s)

    def propose(self, event, past, scenario, map_doc, method='rollout'):
        """Return the pre-planner bridge payload and causal forecast diagnostics.

The runner calls method='none' during warmup and after all deliveries. For
active comparisons it supplies history ending at the current observation.
"""
        started = time.perf_counter()
        _method(method)
        road = road_mask(map_doc)
        forecast_wall_s = 0.
        if method == 'none':
            candidates = [{'name': 'zero', 'cell_penalties': np.zeros(road.size).tolist()}]
        else:
            observation_clock(event)
            past = np.asarray(past, dtype=float)
            current = pose_array(event['pose_frame'])
            if (past.ndim != 3 or past.shape[0] < 21 or past.shape[1:] != current.shape
                    or not np.isfinite(past).all()
                    or not np.allclose(past[-1], current, rtol=0., atol=1e-8)):
                raise ValueError('Forecast history must end at the current observed poses')
            forecast_started = time.perf_counter()
            result = rollout(event['snapshot'], event['pose_frame'], scenario, map_doc,
                             controllers=event['controllers'], dt=POLICY['rollout_dt_s'],
                             horizon_s=POLICY['forecast_horizon_s'],
                             native_poll_before_completion=True)
            fields, _ = predicted_rasters(result, past, scenario, road.shape)
            prediction = window_mean(fields)
            candidates = forecast_cost_menu(prediction, road)
            forecast_wall_s = time.perf_counter() - forecast_started
        metadata = _cost_metadata(candidates)
        diagnostics = dict(policy_version=POLICY['version'], method=method,
                           candidate_count=len(candidates), candidates=metadata,
                           distinct_penalty_fields=len({c['penalty_sha256'] for c in metadata}),
                           forecast_wall_s=forecast_wall_s,
                           query_wall_s=time.perf_counter() - started,
                           forecast_is_shared_by_selectors=True)
        payload = dict(clock=event['clock'], cell_penalties=candidates[0]['cell_penalties'],
                       route_candidates=candidates)
        return payload, diagnostics

    def select(self, event, scenario, map_doc, method='rollout', completed_robot_ids=()):
        """Select an exact successful preview; never resubmit PBS for the winner.

completed_robot_ids must come from native final-service acknowledgements, not
empty routes or task-assigner state. Invalid alternate suffixes are ineligible;
invalid baseline scoring safely keeps the zero-cost native result.
"""
        started = time.perf_counter()
        _method(method)
        candidates = event['candidates']
        if (not candidates or [c['index'] for c in candidates] != list(range(len(candidates)))
                or len(candidates) > 3):
            raise ValueError('Expected one to three ordered candidates beginning at index zero')
        diagnostics = dict(policy_version=POLICY['version'], method=method,
                           selected_index=0, selected_name=candidates[0]['name'],
                           selection_reason='zero_only', candidate_count=len(candidates),
                           minimum_mean_eta_gain_s=self.minimum_mean_eta_gain_s,
                           eta_horizon_s=self.eta_horizon_s,
                           baseline_mean_eta_s=None, best_mean_eta_gain_s=None,
                           selected_mean_eta_gain_s=0.,
                           completed_robot_ids=sorted(int(rid) for rid in completed_robot_ids),
                           candidates=[])
        for candidate in candidates:
            diagnostics['candidates'].append(dict(
                index=candidate['index'], name=candidate['name'],
                evaluated=bool(candidate['evaluated']), success=bool(candidate['success']),
                eligible=False, mean_eta_s=None, mean_eta_gain_s=None,
                mean_nominal_eta_s=None, nominal_detour_mean_s=None,
                common_service_targets_validated=False,
                score_wall_s=0., rejection_reason=None,
                planner_wall_s=candidate.get('planner_wall_s'),
                nonzero_penalty_cells=candidate.get('nonzero_penalty_cells'),
                max_cell_penalty=candidate.get('max_cell_penalty'),
                native_plan_sha256=_digest(candidate.get('plan'))))

        def finish(reason, selected=0):
            diagnostics.update(selected_index=selected, selected_name=candidates[selected]['name'],
                               selection_reason=reason, query_wall_s=time.perf_counter() - started)
            return dict(clock=event['clock'], selected_index=selected), diagnostics

        # Do not inspect empty/incomplete failed-plan suffixes: native fallback
        # remains responsible for this case, as in the existing simulator.
        if not candidates[0]['evaluated'] or not candidates[0]['success']:
            return finish('zero_planner_failed_native_fallback')
        if method == 'none':
            return finish('zero_only')
        if event.get('boundary') != 'post_commit_before_plan_selection':
            raise ValueError('Route selection requires the post-commit preview boundary')

        try:
            targets = service_targets(candidates[0]['preview_snapshot'],
                                      completed_robot_ids=diagnostics['completed_robot_ids'])
        except ValueError as error:
            diagnostics['candidates'][0]['rejection_reason'] = str(error)
            return finish('zero_target_invalid')
        diagnostics['service_targets_sha256'] = _digest(targets)

        for candidate, row in zip(candidates, diagnostics['candidates']):
            if not candidate['evaluated'] or not candidate['success']:
                row['rejection_reason'] = 'planner_failed_or_skipped'
                continue
            score_started = time.perf_counter()
            try:
                snapshot = candidate['preview_snapshot']
                observed = event.get('observation_clock', event['pose_frame']['clock'])
                if (snapshot['clock'] != observed or event['pose_frame']['clock'] != observed
                        or snapshot.get('server_clock', event['clock']) != event['clock']
                        or any(c['clock'] != observed for c in event['controllers'])):
                    raise ValueError('Preview, pose, controller, or server clocks differ')
                score = score_route_eta(snapshot, event['pose_frame'], event['controllers'],
                                        scenario, map_doc, targets, mode=method,
                                        horizon_s=self.eta_horizon_s)
                row['mean_eta_s'] = _score_checked(score)
                row['eligible'] = True
                row['common_service_targets_validated'] = True
                row['mean_nominal_eta_s'] = score.get('mean_nominal_eta_s')
                row['eta'] = score
            except ValueError as error:
                row['rejection_reason'] = str(error)
            finally:
                row['score_wall_s'] = time.perf_counter() - score_started
            if candidate['index'] == 0 and not row['eligible']:
                return finish('zero_score_invalid')

        baseline = diagnostics['candidates'][0]['mean_eta_s']
        diagnostics['baseline_mean_eta_s'] = baseline
        eligible = [row for row in diagnostics['candidates'] if row['eligible']]
        for row in eligible:
            row['mean_eta_gain_s'] = baseline - row['mean_eta_s']
            nominal_baseline = diagnostics['candidates'][0]['mean_nominal_eta_s']
            if row['mean_nominal_eta_s'] is not None and nominal_baseline is not None:
                row['nominal_detour_mean_s'] = row['mean_nominal_eta_s'] - nominal_baseline
        # Stable index tie-break gives zero first, then the milder cost field.
        best = min(eligible, key=lambda row: (row['mean_eta_s'], row['index']))
        gain = baseline - best['mean_eta_s']
        diagnostics['best_mean_eta_gain_s'] = gain
        if best['index'] and gain > self.minimum_mean_eta_gain_s:
            diagnostics['selected_mean_eta_gain_s'] = gain
            return finish('predicted_mean_eta_improvement', best['index'])
        return finish('zero_wins_tie_or_margin')
