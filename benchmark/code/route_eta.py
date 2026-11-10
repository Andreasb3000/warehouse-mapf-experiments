"""Compare alternative accepted routes to the same known terminal services.

The fixed fleet denominator includes independently confirmed parked robots.
For an active robot, nominal ETA is remaining free-flow route work. Rollout ETA
is its terminal SERVICE completion offset, or H plus remaining nominal route
work if that service has not completed by H. Longer detours therefore cost
time instead of receiving extra progress credit. The tail is optimistic:
future safety/dependency delays and future native replanning are not modeled.
This predicts the common accepted terminal service, not unplanned deliveries.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2] / 'simulator'
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.prediction.joint import remaining_nominal_work, rollout


def _robots(snapshot):
    robots = {int(r['id']): r for r in snapshot['robots']}
    if not robots or len(robots) != len(snapshot['robots']):
        raise ValueError('ETA requires a nonempty fleet with unique robot IDs')
    return robots


def _remaining(robot):
    actions = sorted((a for a in robot['actions'] if a['node_id'] > robot['finished_node_id']),
                     key=lambda a: a['node_id'])
    if len({a['node_id'] for a in actions}) != len(actions):
        raise ValueError('Duplicate remaining action node')
    if any(a['type'] not in ('M', 'T', 'S') for a in actions):
        raise ValueError('Unsupported accepted action in route ETA')
    return actions


def _services(actions):
    services = []
    for action in actions:
        if action['type'] != 'S':
            continue
        goal = action['goal']
        task_id = action.get('task_id')
        if type(task_id) is not int or task_id < 0 or len(goal) != 2 or not all(math.isfinite(v) for v in goal):
            raise ValueError('Terminal service requires a real task ID and finite goal')
        services.append({'task_id': task_id, 'goal': [float(v) for v in goal]})
    if len({s['task_id'] for s in services}) != len(services):
        raise ValueError('A remaining service task appears more than once')
    return services


def service_targets(snapshot, *, completed_robot_ids=()):
    """Freeze the comparison target from the successful zero-penalty preview.

    ``completed_robot_ids`` must come from actual native final-delivery stage
    acknowledgements. An empty accepted prefix alone is never completion proof.
    A plan may contain a committed pickup SERVICE and a later delivery SERVICE;
    all remaining services, not merely the last goal, are kept common.
    """
    robots = _robots(snapshot)
    completed = set(completed_robot_ids)
    if not completed <= set(robots):
        raise ValueError('Completed IDs are not in the common fleet')
    targets = {}
    for rid, robot in sorted(robots.items()):
        actions = _remaining(robot)
        if rid in completed:
            if actions:
                raise ValueError(f'Confirmed completed robot {rid} still has accepted actions')
            targets[rid] = {'parked': True, 'services': []}
        else:
            if not actions or actions[-1]['type'] != 'S':
                raise ValueError(f'Robot {rid} has no complete route to a terminal service')
            services = _services(actions)
            if not services:
                raise ValueError(f'Robot {rid} has no remaining service task')
            targets[rid] = {'parked': False, 'services': services}
    return targets


def _validate_candidate(snapshot, targets):
    robots = _robots(snapshot)
    targets = {int(rid): value for rid, value in targets.items()}
    if set(robots) != set(targets):
        raise ValueError('Candidate changes the common fleet denominator')
    known = {(rid, a['node_id']) for rid, robot in robots.items() for a in _remaining(robot)}
    for robot in robots.values():
        for action in _remaining(robot):
            for other, node in action.get('dependencies', []):
                if other not in robots or (node > robots[other]['finished_node_id'] and (other, node) not in known):
                    raise ValueError('Candidate preview has unresolved missing ADG dependencies')
    terminal_nodes = {}
    for rid, robot in robots.items():
        actions, target = _remaining(robot), targets[rid]
        if target['parked']:
            if actions or target['services']:
                raise ValueError(f'Parked robot {rid} acquired a route/task')
            continue
        if not actions or actions[-1]['type'] != 'S':
            raise ValueError(f'Robot {rid} candidate ends before the common terminal service')
        services = _services(actions)
        if services != target['services']:
            raise ValueError(f'Robot {rid} candidate changes remaining service IDs/goals')
        terminal_nodes[rid] = actions[-1]['node_id']
    return robots, targets, terminal_nodes


def _initial_work(snapshot, pose_frame, controllers, robots, scenario, station_dwell_s, turn_rate_deg_s):
    poses = {int(p['id']): p for p in pose_frame['agents']}
    controls = {int(c['id']): c for c in controllers}
    if len(poses) != len(pose_frame['agents']) or len(controls) != len(controllers) or set(poses) != set(robots) or set(controls) != set(robots):
        raise ValueError('ETA needs one current pose and controller per common robot')
    if pose_frame['clock'] > snapshot['clock'] or any(c['clock'] > snapshot['clock'] for c in controllers):
        raise ValueError('ETA cannot use state from after its origin')
    if not math.isclose(float(scenario.get('cell_size_m', 1)), 1):
        raise ValueError('Route ETA requires one-metre native map coordinates')
    dt = 1 / float(scenario.get('ticks_per_second', 10))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid native time step')
    work = {}
    for rid, robot in robots.items():
        controller, pose = controls[rid], poses[rid]
        if controller.get('poll_interval_ticks', 10) != 10:
            raise ValueError('Route ETA requires native ten-tick action polling')
        actions = _remaining(robot)
        known = {a['node_id']: a for a in actions}
        queue, queued = [], set()
        for item in controller['queue']:
            nodes = [n for n in item['node_ids'] if n > robot['finished_node_id']]
            if not nodes:
                continue
            if any(n not in known or n in queued for n in nodes):
                raise ValueError('Native controller queue is absent/duplicated in preview actions')
            if (any(known[n]['type'] != item['type'] for n in nodes) or
                    list(item['goal']) != list(known[nodes[-1]]['goal'])):
                raise ValueError('Preview changed an already dispatched queue action')
            queued.update(nodes)
            queue.append(dict(type=item['type'], node_ids=list(nodes), goal=list(item['goal']),
                              yaw=float(item['orientation_rad']),
                              remaining_s=max(0., float(item.get('remaining_ticks', 0)) * dt)))
        pending = [a for a in actions if a['node_id'] not in queued]
        if any(a['node_id'] <= robot['enqueued_node_id'] for a in pending):
            raise ValueError('Preview loses an already dispatched action')
        state = dict(row=-float(pose['x']), col=-float(pose['y']), yaw=float(pose.get('rz', 0)),
                     nominal_speed=float(controller['nominal_speed_m_s']), queue=queue)
        if not all(math.isfinite(state[k]) for k in ('row', 'col', 'yaw', 'nominal_speed')):
            raise ValueError('Nonfinite initial route state')
        work[rid] = remaining_nominal_work(state, pending, station_dwell_s=station_dwell_s,
                                          turn_rate_deg_s=turn_rate_deg_s)
    return work


def score_route_eta(snapshot, pose_frame, controllers, scenario, map_doc, targets, *,
                    mode='rollout', horizon_s=30., station_dwell_s=2., turn_rate_deg_s=90.):
    """Return comparable mean/per-robot ETA in seconds; lower is better.

    Invalid/partial/different-target candidates raise ``ValueError``. Successful
    callers should mark that branch ineligible; an invalid zero candidate means
    there is no valid common comparison. No fallback route is silently repaired.
    """
    if mode not in ('nominal', 'rollout'):
        raise ValueError('ETA mode must be nominal or rollout')
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('ETA horizon must be positive and finite')
    robots, targets, terminals = _validate_candidate(snapshot, targets)
    nominal = _initial_work(snapshot, pose_frame, controllers, robots, scenario, station_dwell_s, turn_rate_deg_s)
    parked = sorted(rid for rid, target in targets.items() if target['parked'])
    if any(nominal[rid] != 0 for rid in parked):
        raise ValueError('Parked robot has unfinished queued work')
    completed, tails = [], []
    prediction = None
    if mode == 'nominal':
        estimates = dict(nominal)
        tails = sorted(terminals)
    else:
        prediction = rollout(snapshot, pose_frame, scenario, map_doc, horizon_s=horizon_s,
                             dt=1 / float(scenario.get('ticks_per_second', 10)), controllers=controllers,
                             station_dwell_s=station_dwell_s, turn_rate_deg_s=turn_rate_deg_s,
                             native_poll_before_completion=True, record_remaining_work=True)
        if prediction['diagnostics']['missing_dependencies']:
            raise ValueError('Candidate preview has unresolved missing ADG dependencies')
        remaining = prediction['diagnostics']['nominal_remaining_work_s']
        if any(not math.isclose(nominal[rid], remaining['initial'][str(rid)], rel_tol=0, abs_tol=1e-9) for rid in robots):
            raise ValueError('Initial route-work reconstruction differs from rollout')
        estimates = {rid: 0. for rid in parked}
        for rid, node in terminals.items():
            matches = [e for e in prediction['completions'] if e['id'] == rid and e['node_id'] == node]
            if matches:
                if len(matches) != 1 or matches[0]['type'] != 'S' or not 0 <= matches[0]['offset_s'] <= horizon_s:
                    raise ValueError('Invalid terminal service completion record')
                estimates[rid] = float(matches[0]['offset_s'])
                completed.append(rid)
            else:
                tail = float(remaining['terminal'][str(rid)])
                if not math.isfinite(tail) or tail < 0:
                    raise ValueError('Invalid remaining route-work tail')
                estimates[rid] = horizon_s + tail
                tails.append(rid)
    ids = sorted(robots)
    return dict(mean_eta_s=sum(estimates.values()) / len(ids),
                per_robot_eta_s={str(rid): estimates[rid] for rid in ids},
                mean_nominal_eta_s=sum(nominal.values()) / len(ids),
                nominal_per_robot_eta_s={str(rid): nominal[rid] for rid in ids},
                completed_terminal_ids=sorted(completed), optimistic_tail_ids=sorted(tails),
                parked_robot_ids=parked,
                diagnostics=dict(mode=mode, horizon_s=horizon_s, robot_count=len(ids),
                    targets={str(rid): targets[rid] for rid in ids},
                    terminal_node_ids={str(rid): node for rid, node in terminals.items()},
                    target_scope='Common accepted terminal service, including earlier committed services; not unplanned future tasks.',
                    objective='Mean predicted terminal SERVICE completion offset; unfinished at H contributes H plus nominal remaining route time.',
                    tail_assumptions='Future safety/dependency, acceleration, polling delays and native replanning omitted from nominal tail.',
                    native_poll_before_completion=True,
                    rollout_diagnostics=prediction['diagnostics'] if prediction else None))
