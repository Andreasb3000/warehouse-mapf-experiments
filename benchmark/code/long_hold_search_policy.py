"""Exploratory H10 search with a selected six-second admission episode.

This separate policy reuses the short-search candidate geometry and progress
objective. A new search models a six-second hold, chooses only a sufficient
all-fleet progress gain, and retains the chosen IDs for three two-second
reviews without searching again. Each native response still expires after
two seconds. One two-second release follows before another search is allowed.

The rollout keeps current accepted plans fixed. Future PBS plans and later
commitment promotions are not modeled. Native queued/committed-action guards
remain authoritative and may bypass a requested hold; retaining an ID does
not authorize stopping its committed motion. No task completion is inferred
from the accepted-work proxy, and no native benefit is claimed by this module.
"""
from __future__ import annotations

import math
import time

import numpy as np

from features import observation_clock, pose_array
from experiments.prediction.joint import rollout
from search_admission_policy import (POLICY as SHORT_POLICY, candidate_interventions,
                                     eligible_entries, progress_score)


POLICY = {
    **SHORT_POLICY,
    'version': 'long-hold-search-v1',
    'method': 'search_h10_hold6',
    'default_horizon_s': 10.,
    'allowed_horizons_s': [10.],
    'intended_hold_s': 6.,
    'intended_hold_ticks': 60,
    'search_during_active_hold': False,
    'future_commitment_promotions_modeled': False,
}
_EPS = 1e-9


class LongHoldSearchPolicy:
    """One instance per run; a hold episode is followed by mandatory release."""

    def __init__(self, horizon_s=10., absolute_gain_s=.5, relative_gain_fraction=0., tasks=None):
        if horizon_s != 10.:
            raise ValueError('The exploratory long-hold policy uses a ten-second horizon')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in (absolute_gain_s, relative_gain_fraction)):
            raise ValueError('Gain margins must be finite nonnegative numbers')
        self.horizon_s = 10.
        self.absolute_gain_s, self.relative_gain_fraction = float(absolute_gain_s), float(relative_gain_fraction)
        self.tasks = tasks
        self.last_clock = None
        self.active_ids = []
        self.active_until_clock = None
        self.release_until_clock = None
        self.selection_origin_clock = None
        self.source_choice = None
        self.decision_count = self.intervention_count = self.search_call_count = self.new_hold_count = 0

    def _search(self, event, scenario, map_doc, entries):
        candidates = candidate_interventions(entries)
        ids = [int(robot['id']) for robot in event['snapshot']['robots']]
        evaluations = []
        for candidate in candidates:
            admission = None
            if candidate['held_robot_ids']:
                admission = {'center': list(POLICY['center']), 'half_width': POLICY['half_width'],
                             'held_robot_ids': candidate['held_robot_ids'],
                             'until_offset_s': POLICY['intended_hold_s'], 'protect_clearance': False}
            started = time.perf_counter()
            prediction = rollout(event['snapshot'], event['pose_frame'], scenario, map_doc,
                                 horizon_s=self.horizon_s, dt=POLICY['dt_s'], controllers=event['controllers'],
                                 admission=admission, record_progress=True, native_poll_before_completion=True)
            score = progress_score(prediction, ids, self.horizon_s)
            gate = prediction.get('diagnostics', {}).get('admission', {})
            evaluations.append({**candidate, **score, 'rollout_wall_s': time.perf_counter() - started,
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
        chosen = best if best['score_work_s'] - baseline > margin + _EPS else evaluations[0]
        return evaluations, chosen, best, baseline, margin

    def decide(self, event, past, scenario, map_doc, method='search_h10_hold6'):
        started = time.perf_counter()
        if method != POLICY['method']:
            raise ValueError('Use method=search_h10_hold6 for the long-hold policy')
        if event.get('boundary') != 'post_planner_before_dispatch':
            raise ValueError('Long-hold search requires a fresh post-planner snapshot')
        observation = observation_clock(event)
        if not event.get('controllers'):
            raise ValueError('Counterfactuals require current native controller state')
        clock = event['clock']
        if type(clock) is not int or clock < 0 or (self.last_clock is not None and clock <= self.last_clock):
            raise ValueError('Decision clocks must be strictly increasing nonnegative ticks')
        if past is not None:
            history = np.asarray(past)
            if history.ndim != 3 or not len(history) or not np.allclose(history[-1], pose_array(event['pose_frame']), atol=1e-7, rtol=0):
                raise ValueError('Optional history must end at the current observation')
        entries, exclusions = eligible_entries(event)
        live_ids = {int(robot['id']) for robot in event['snapshot']['robots']}
        if not set(self.active_ids) <= live_ids:
            raise ValueError('An active held robot disappeared from the native snapshot')
        continuing = bool(self.active_ids) and clock < self.active_until_clock
        releasing = (not continuing and self.release_until_clock is not None
                     and clock < self.release_until_clock)
        evaluations, new_hold = [], False
        baseline = margin = best_gain = selected_gain = None
        best_name = None
        if continuing:
            held = list(self.active_ids)
            selected_name = self.source_choice['name']
            reason = 'continuing_selected_hold'
        elif releasing:
            held, selected_name = [], 'noop'
            reason = 'forced_release_after_hold'
        else:
            evaluations, chosen, best, baseline, margin = self._search(event, scenario, map_doc, entries)
            held, selected_name = list(chosen['held_robot_ids']), chosen['name']
            best_name, best_gain, selected_gain = best['name'], best['gain_over_noop_s'], chosen['gain_over_noop_s']
            self.search_call_count += 1
            new_hold = bool(held)
            if new_hold:
                self.active_ids = list(held)
                self.selection_origin_clock = clock
                self.active_until_clock = clock + POLICY['intended_hold_ticks']
                self.release_until_clock = self.active_until_clock + POLICY['forced_release_ticks']
                self.source_choice = {'name': selected_name, 'selected_gain_s': selected_gain}
                self.new_hold_count += 1
                reason = 'gain_exceeds_margin_start_hold'
            else:
                self.active_ids = []
                self.active_until_clock = self.release_until_clock = self.selection_origin_clock = None
                self.source_choice = None
                reason = 'noop_best_tied_or_below_margin'
        self.last_clock = clock
        self.decision_count += 1
        self.intervention_count += bool(held)
        payload = {'center': list(POLICY['center']), 'half_width': POLICY['half_width'],
                   'held_robot_ids': held, 'until_clock': clock + POLICY['interval_ticks'],
                   'protect_clearance': False}
        exempt = set(exclusions['inside']) | set(exclusions['committed_or_queued_entry'])
        diagnostics = {
            'policy_version': POLICY['version'], 'method': method, 'clock': clock,
            'observation_clock': observation, 'horizon_s': self.horizon_s,
            'execution_interval_s': POLICY['execution_interval_s'], 'intended_hold_s': POLICY['intended_hold_s'],
            'native_poll_before_completion': True, 'objective': POLICY['objective'], 'task_completion_inferred': False,
            'candidate_count': len(evaluations), 'candidates': evaluations,
            'candidate_ids': sorted(entries), 'candidate_entries': {str(rid): detail for rid, detail in entries.items()},
            'excluded_robot_ids': exclusions, 'new_hold_started': new_hold, 'continuing_hold': continuing,
            'active_until_clock': self.active_until_clock, 'forced_release_until_clock': self.release_until_clock,
            'selection_origin_clock': self.selection_origin_clock,
            'source_selected_gain_s': self.source_choice['selected_gain_s'] if self.source_choice else None,
            'selected_ids_now_inside_or_committed': sorted(set(held) & exempt),
            'forced_release_robot_ids': list(self.active_ids) if releasing else [],
            'applied_hold_since_clock': {str(rid): self.selection_origin_clock for rid in held},
            'noop_score_work_s': baseline, 'absolute_gain_margin_s': self.absolute_gain_s,
            'relative_gain_margin_fraction': self.relative_gain_fraction, 'required_gain_s': margin,
            'best_candidate': best_name, 'best_gain_s': best_gain,
            'selected_candidate': selected_name, 'selected_gain_s': selected_gain,
            'held_robot_ids': list(held), 'intervened': bool(held), 'selection_reason': reason,
            'decision_count': self.decision_count, 'intervention_count': self.intervention_count,
            'search_call_count': self.search_call_count, 'new_hold_count': self.new_hold_count,
            'choice_fraction': self.intervention_count / self.decision_count,
            'noop_fraction': 1. - self.intervention_count / self.decision_count,
            'new_hold_per_search_fraction': self.new_hold_count / self.search_call_count if self.search_call_count else 0.,
            'policy_wall_s': time.perf_counter() - started,
            'warning': 'Future PBS replanning and commitment promotions are not modeled. Native guards may bypass retained hold requests.',
            'assumptions': ['The selected IDs are retained without new search during the six-second episode.',
                            'Only each two-second native request is live; missed refreshes can create gaps in actual gating.',
                            'Current accepted plans and commitments remain fixed inside every counterfactual.',
                            'Accepted work is a progress proxy, never a task-completion oracle.'],
        }
        return payload, diagnostics
