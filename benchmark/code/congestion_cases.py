#!/usr/bin/env python3
"""Demand-only junction stress fixtures on the unchanged nine-zone map.

All 27 robots start north-facing at time zero. ``compact`` and ``spread``
change geometric staging distances; they do not invent delayed-release support.
Nominal arrival estimates omit turns, acceleration, planning, and interactions.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy.optimize import linear_sum_assignment

from record_dataset import make_workload, write_json
from experiments.zones.generate import build_case, render_svg
from experiments.prediction.joint import _Safety, _fields
from experiments.junction.scenario import load_scenario

ARMS = {"north": "Z01", "east": "Z12", "south": "Z21", "west": "Z10"}
OPPOSITE = {"north": "Z21", "east": "Z10", "south": "Z01", "west": "Z12"}
CORNERS = ("Z00", "Z02", "Z20", "Z22")
CORNER_DESTINATIONS = {"north": ("Z20", "Z22"), "east": ("Z00", "Z20"),
                       "south": ("Z00", "Z02"), "west": ("Z02", "Z22")}
CENTER = frozenset((r, c) for r in range(29, 32) for c in range(29, 32))


def neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def distances(start, free):
    seen, queue = {start: 0}, deque([start])
    while queue:
        cell = queue.popleft()
        for nxt in neighbors(cell):
            if nxt in free and nxt not in seen:
                seen[nxt] = seen[cell] + 1
                queue.append(nxt)
    return seen


@lru_cache(maxsize=1)
def _geometry():
    map_doc, scenario, _, _ = build_case(3)
    free = frozenset((r, c) for r, row in enumerate(map_doc['layout'])
                     for c, value in enumerate(row) if value not in '@T')
    bays = {z['id']: sorted(tuple(cell) for cells in z['bays'].values() for cell in cells)
            for z in scenario['zones']}
    all_bays = sorted(cell for cells in bays.values() for cell in cells)
    normal = {cell: distances(cell, free) for cell in all_bays}
    bypass = {cell: distances(cell, free - CENTER) for cell in all_bays}
    owners = {cell: zone for zone, cells in bays.items() for cell in cells}
    return free, bays, normal, bypass, owners


def _initial_safety(scenario, map_doc, starts, trigger):
    temporary = deepcopy(scenario)
    temporary['safety']['robot_trigger'] = trigger
    states = {i: {'row': cell[0], 'col': cell[1], 'yaw': 0.0}
              for i, cell in enumerate(starts)}
    return _Safety(_fields(temporary, list(states), True), map_doc).evaluate(states)


def _place_starts(records, scenario, map_doc, normal, free, goals, rng, burst, arrival_s):
    zones = {z['id']: z for z in scenario['zones']}
    # North-facing fields should initially be clear of shelves as well as
    # robots. Candidate body cells are aisles, never somebody's service bay.
    bay_mouths = {neighbor for goal in goals for neighbor in neighbors(goal) if neighbor in free}
    candidate_cells = [cell for cell in sorted(free - goals - bay_mouths)
                       if sum(n in free for n in neighbors(cell)) >= 2]
    candidates = {}
    one = deepcopy(scenario); one['agents'] = [deepcopy(scenario['agents'][0])]
    geometry = _Safety(_fields(one, [0], True), map_doc)
    for zone_id, zone in zones.items():
        rr, cc = zone['bounds']['rows'], zone['bounds']['cols']
        candidates[zone_id] = [cell for cell in candidate_cells
                              if rr[0] <= cell[0] <= rr[1] and cc[0] <= cell[1] <= cc[1]
                              and geometry.evaluate({0: {'row': cell[0], 'col': cell[1], 'yaw': 0.0}})[0][0] == 'clear']
    starts = []
    for item in records:
        pickup = item['pickup']; approach_distance = min(normal[pickup][c] for c in CENTER)
        if item['role'] == 'center_crossing':
            fraction = item['arm_index'] / max(1, item['robots_per_arm'] - 1)
            target = arrival_s + (16 * (fraction - .5) if burst == 'spread' else 0)
            desired_pickup_distance = max(2.0, target - 2.0 - approach_distance)
        else:
            target = None
            desired_pickup_distance = 4.0
        options = []
        for cell in candidates[item['source_zone']]:
            if any(math.dist(cell, old) < 2.0 - 1e-12 for old in starts):
                continue
            # Initial north-facing rectangles have length 1.4m/width 0.6m;
            # the two-metre centre spacing is conservative for either trigger.
            path_distance = normal[pickup][cell]
            options.append((abs(path_distance - desired_pickup_distance),
                            path_distance, rng.random(), cell))
        if not options:
            raise ValueError(f"No clear staging cell remains for {item['source_zone']}")
        _, length, _, cell = min(options)
        starts.append(cell)
        item.update(start=cell, start_to_pickup_distance_m=length,
                    pickup_to_center_distance_m=approach_distance,
                    nominal_center_arrival_s=length + 2.0 + approach_distance,
                    target_nominal_center_arrival_s=target)
    return starts


def make_congestion_workload(seed, duration_s=240, center_load=12, burst='compact',
                             trigger='body_in_outer', nominal_arrival_s=26.0):
    """Return map, scenario, config, tasks, assignment; never change floor cells.

    ``center_load=0`` returns the original dispersed workload. Loads 12 and 24
    give 3 and 6 pickup streams per cardinal arm. Terminal bays remain private
    leaves. Active pairs strictly prefer the center under ordinary spatial
    distance, but retain a finite route when the center is removed.
    """
    if type(center_load) is not int or center_load not in (0, 12, 24):
        raise ValueError('center_load must be 0, 12, or 24')
    if burst not in ('compact', 'spread') or trigger not in ('body_in_outer', 'outer_overlap'):
        raise ValueError('Unsupported burst pattern or safety trigger')
    if not math.isfinite(duration_s) or duration_s <= 0 or duration_s * 10 != round(duration_s * 10):
        raise ValueError('Duration must be a positive multiple of 0.1s')
    if not math.isfinite(nominal_arrival_s) or not 20 <= nominal_arrival_s <= 40:
        raise ValueError('Nominal geometric arrival must be between 20 and 40s')
    if center_load == 0:
        result = make_workload(seed, duration_s)
        result[1]['safety']['robot_trigger'] = trigger
        result[2]['parameters']['sim_duration'] = round(duration_s * 10)
        return result
    map_doc, scenario, config, tasks = build_case(3)
    free, bays, normal, bypass, owners = _geometry()
    rng = random.Random(seed)
    per_arm = center_load // 4
    records = []
    for arm, zone in ARMS.items():
        ranked = [(min(normal[cell][c] for c in CENTER), rng.random(), cell) for cell in bays[zone]]
        selected = [cell for _, _, cell in sorted(ranked)[:per_arm]]
        rng.shuffle(selected)
        for index, pickup in enumerate(selected):
            records.append({'role': 'center_crossing', 'approach': arm, 'source_zone': zone,
                            'pickup': pickup, 'arm_index': index, 'robots_per_arm': per_arm})
    pickups = {r['pickup'] for r in records}
    delivery_candidates = sorted(cell for cells in bays.values() for cell in cells if cell not in pickups)
    costs = np.full((center_load, len(delivery_candidates)), 1e8, dtype=float)
    for i, item in enumerate(records):
        arm, pickup = item['approach'], item['pickup']
        for j, goal in enumerate(delivery_candidates):
            zone = owners[goal]
            if zone != OPPOSITE[arm] and zone not in CORNER_DESTINATIONS[arm]:
                continue
            shortest = normal[pickup][goal]
            alternative = bypass[pickup].get(goal)
            if alternative is None or alternative <= shortest:
                continue
            costs[i, j] = (0 if zone == OPPOSITE[arm] else 100) + shortest - .25 * (alternative - shortest) + rng.random() * .1
    chosen_rows, chosen_cols = linear_sum_assignment(costs)
    if len(chosen_rows) != center_load or any(costs[i, j] >= 1e8 for i, j in zip(chosen_rows, chosen_cols)):
        raise ValueError('Private bay capacity cannot satisfy requested center-preferring traffic')
    for i, j in zip(chosen_rows, chosen_cols):
        item = records[int(i)]; goal = delivery_candidates[int(j)]
        item.update(dropoff=goal, destination_zone=owners[goal],
                    shortest_delivery_distance_m=normal[item['pickup']][goal],
                    center_avoiding_distance_m=bypass[item['pickup']][goal])
    used = pickups | {r['dropoff'] for r in records}
    background_counts = Counter()
    while len(records) < 27:
        options = []
        for zone in CORNERS:
            available = [cell for cell in bays[zone] if cell not in used]
            if len(available) >= 2:
                options.append((background_counts[zone], rng.random(), zone, available))
        if not options:
            raise ValueError('No private bays remain for background tasks')
        _, _, zone, available = min(options)
        pickup = rng.choice(available)
        dropoff = min((normal[pickup][cell], rng.random(), cell) for cell in available if cell != pickup)[2]
        records.append({'role': 'local_background', 'approach': zone, 'source_zone': zone,
                        'pickup': pickup, 'dropoff': dropoff, 'destination_zone': zone,
                        'shortest_delivery_distance_m': normal[pickup][dropoff],
                        'center_avoiding_distance_m': bypass[pickup][dropoff]})
        used.update((pickup, dropoff)); background_counts[zone] += 1
    starts = _place_starts(records, scenario, map_doc, normal, free, used, rng, burst, nominal_arrival_s)
    zone_ids = [z['id'] for z in scenario['zones']]
    zone_index = {z: i for i, z in enumerate(zone_ids)}
    allocation = [[0] * 9 for _ in range(9)]
    populations = Counter()
    zone_records = {z['id']: z for z in scenario['zones']}
    for zone in scenario['zones']:
        zone['bays'] = {'starts': [], 'pickups': [], 'dropoffs': []}
    for robot_id, (item, agent, task) in enumerate(zip(records, scenario['agents'], tasks['tasks'])):
        source, destination = item['source_zone'], item['destination_zone']
        goals = [list(item['pickup']), list(item['dropoff'])]
        agent.update(zone_id=source, approach=source, center_approach=item['approach'], pickup_zone_id=source,
                     destination_zone_id=destination, start=list(item['start']), goals=goals,
                     release_at_s=0, color=zone_records[source]['color'], demand_role=item['role'])
        task.update(source_zone_id=source, destination_zone_id=destination)
        task['stages'][0].update(zone_id=source, location=goals[0])
        task['stages'][1].update(zone_id=destination, location=goals[1])
        allocation[zone_index[source]][zone_index[destination]] += 1
        populations[source] += 1
        zone_records[source]['bays']['starts'].append(list(item['start']))
        zone_records[source]['bays']['pickups'].append(goals[0])
        zone_records[destination]['bays']['dropoffs'].append(goals[1])
        item['robot_id'] = robot_id
        item['center_preference_gap_m'] = item['center_avoiding_distance_m'] - item['shortest_delivery_distance_m']
    initial_matrix = [[populations[f'Z{r}{c}'] for c in range(3)] for r in range(3)]
    name = f'center_demand_{center_load}_{burst}_27_robots'
    scenario.update(name=name, description=f'{center_load} center-preferring delivery tasks, {per_arm} per cardinal approach, with geometric {burst} staging and {27-center_load} local corner tasks. The floor, fleet size, safety fields, and zero-time releases are unchanged.')
    scenario['safety']['robot_trigger'] = trigger
    scenario['network'].update(robots_per_zone=None, initial_robot_matrix=initial_matrix, allocation_matrix=allocation)
    scenario['notes'] = [
        'The complete map document is identical to the original nine-zone 27-robot fixture.',
        'All robots face north and are released at zero. Burst labels describe staging geometry, not delayed task releases.',
        'All 54 service goals are distinct existing degree-one bays; starts are clear existing aisle cells with at least 2m center spacing.',
        'Center-preferring delivery pairs have strictly longer but feasible center-avoiding routes. No center waypoint is imposed.',
        'Nominal arrival equals shortest start-to-pickup distance + 2s service + shortest pickup-to-center distance at 1m/s; it omits turns, acceleration, planning and interactions.',
        'Payload identity and mass remain metadata. Native station actions implement service timing.',
    ]
    tasks.update(scenario=name, allocation_matrix=allocation,
                 initial_robot_counts={z: populations[z] for z in zone_ids}, initial_robot_matrix=initial_matrix)
    config['parameters'].update(seed=seed, sim_duration=round(duration_s * 10))
    assignment = {'schema_version': 1, 'workload_seed': seed, 'family': 'concentrated_center_demand',
                  'center_load': center_load, 'robots_per_arm': per_arm, 'burst': burst,
                  'nominal_target_arrival_s': nominal_arrival_s,
                  'arrival_estimate_scope': 'Geometric distance plus nominal service only; actual native arrival must be measured.',
                  'initial_robot_matrix': initial_matrix, 'allocation_matrix': allocation,
                  'assignments': records,
                  'canonical_map_sha256': hashlib.sha256(json.dumps(map_doc, sort_keys=True).encode()).hexdigest()}
    validate_case(map_doc, scenario, tasks, assignment)
    return map_doc, scenario, config, tasks, assignment


def validate_case(map_doc, scenario, tasks, assignment):
    free, _, _, _, _ = _geometry()
    agents = scenario['agents']
    starts = {tuple(a['start']) for a in agents}
    goals = {tuple(g) for a in agents for g in a['goals']}
    assert map_doc == build_case(3)[0], 'Floor map changed'
    assert len(agents) == len(starts) == 27 and len(goals) == 54 and not starts & goals
    assert (starts | goals) <= free
    assert all(sum(n in free for n in neighbors(g)) == 1 for g in goals)
    assert all(a['release_at_s'] == 0 and len(a['goals']) == 2 for a in agents)
    assert scenario['initial_orientation'] == 'north'
    for trigger in ('body_in_outer', 'outer_overlap'):
        decisions = _initial_safety(scenario, map_doc, [tuple(a['start']) for a in agents], trigger)
        assert all(value[0] == 'clear' for value in decisions.values()), (trigger, decisions)
    for agent, task in zip(agents, tasks['tasks']):
        own = {tuple(agent['start']), *(tuple(g) for g in agent['goals'])}
        # Even if all other robots are initially parked or finish at their
        # terminals, every robot has an individual sequential route.
        available = free - ((starts | goals) - own)
        assert own <= distances(tuple(agent['start']), available).keys(), agent['id']
        assert [stage['location'] for stage in task['stages']] == agent['goals']
    active = [x for x in assignment['assignments'] if x['role'] == 'center_crossing']
    assert len(active) == assignment['center_load']
    assert Counter(x['approach'] for x in active) == Counter({arm: assignment['robots_per_arm'] for arm in ARMS})
    assert all(x['center_preference_gap_m'] > 0 and x['nominal_center_arrival_s'] > 12 for x in active)
    return {'valid': True, 'goals_are_private_leaves': True,
            'initial_fields_clear_for_both_triggers': True,
            'individual_routes_feasible_with_other_starts_and_goals_blocked': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=8201)
    parser.add_argument('--center-load', type=int, choices=(0, 12, 24), default=12)
    parser.add_argument('--burst', choices=('compact', 'spread'), default='compact')
    parser.add_argument('--trigger', choices=('body_in_outer', 'outer_overlap'), default='body_in_outer')
    parser.add_argument('--duration-s', type=int, default=240)
    args = parser.parse_args()
    values = make_congestion_workload(args.seed, args.duration_s, args.center_load, args.burst, args.trigger)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, value in zip(('map.json', 'scenario.json', 'config.json', 'tasks.json', 'assignment.json'), values):
        write_json(args.output / filename, value)
    (args.output / 'layout.svg').write_text(render_svg(values[0], values[1]))
    load_scenario(args.output / 'scenario.json', values[0]['layout'], 27)
    print(args.output.resolve())
