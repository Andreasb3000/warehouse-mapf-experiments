#!/usr/bin/env python3
"""Nested 27/54/81-robot workloads on one shared warehouse floor.

The original aisle and junction cells are preserved. Missing private side bays
are added once to create 192 degree-one service bays for every fleet level.
All fleets are prefixes of one 81-robot assignment, with three, six, or nine
robots per source zone. No congestion classification is built into these inputs.
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

from record_dataset import write_json
from experiments.zones.generate import build_case, render_svg
from experiments.junction.scenario import load_scenario
from experiments.prediction.joint import _Safety, _fields

FLEET_SIZES = (27, 54, 81)
CENTER = frozenset((row, col) for row in range(29, 32) for col in range(29, 32))


def neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def distances(start, free):
    found, queue = {start: 0}, deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current):
            if nxt in free and nxt not in found:
                found[nxt] = found[current] + 1
                queue.append(nxt)
    return found


@lru_cache(maxsize=1)
def _shared_floor():
    original, scenario, config, tasks = build_case(3)
    map_doc = deepcopy(original)
    layout = [list(row) for row in map_doc['layout']]
    zones = {z['id']: z for z in scenario['zones']}
    bays = set()
    for source, target in scenario['network']['edges']:
        start, end = zones[source]['center'], zones[target]['center']
        dr, dc = (end[0] - start[0]) // 20, (end[1] - start[1]) // 20
        for distance in range(3, 18, 2):
            for transverse in (-2, 2):
                cell = (start[0] + distance * dr + transverse * dc,
                        start[1] + distance * dc - transverse * dr)
                bays.add(cell)
                layout[cell[0]][cell[1]] = '.'
    map_doc.update(name='nine_zones_shared_192_private_bays', layout=[''.join(row) for row in layout])
    free = frozenset((r, c) for r, row in enumerate(map_doc['layout'])
                     for c, value in enumerate(row) if value not in '@T')
    if len(bays) != 192 or any(sum(n in free for n in neighbors(cell)) != 1 for cell in bays):
        raise ValueError('Shared floor must contain 192 separate degree-one bays')
    old_free = {(r, c) for r, row in enumerate(original['layout'])
                for c, value in enumerate(row) if value not in '@T'}
    old_bays = {cell for cell in old_free if sum(n in old_free for n in neighbors(cell)) == 1}
    if free - bays != old_free - old_bays or not old_free <= free:
        raise ValueError('Shared floor changed an original aisle or junction cell')
    owners = {}
    for cell in bays:
        owner = [z['id'] for z in scenario['zones']
                 if z['bounds']['rows'][0] <= cell[0] <= z['bounds']['rows'][1]
                 and z['bounds']['cols'][0] <= cell[1] <= z['bounds']['cols'][1]]
        if len(owner) != 1:
            raise ValueError('Every service bay must have one zone owner')
        owners[cell] = owner[0]
    per_zone = {zone: sorted(cell for cell in bays if owners[cell] == zone) for zone in zones}
    normal = {cell: distances(cell, free) for cell in sorted(bays)}
    bypass = {cell: distances(cell, free - CENTER) for cell in sorted(bays)}
    description = {'floor_version': 'shared_192_private_bays_v1',
                   'rows': 61, 'columns': 61, 'free_cells': len(free),
                   'aisle_and_junction_cells': len(free - bays),
                   'private_service_bays': len(bays), 'added_bays': len(free - old_free),
                   'bay_construction': '12 links × 8 odd longitudinal positions × 2 sides; each bay is a degree-one leaf',
                   'aisles_and_junctions_match_original': True,
                   'canonical_map_sha256': hashlib.sha256(json.dumps(map_doc, sort_keys=True).encode()).hexdigest()}
    return map_doc, scenario, config, tasks, free, per_zone, owners, normal, bypass, description


@lru_cache(maxsize=16)
def _full_assignment(seed):
    map_doc, scenario, _, _, _, bays, owners, normal, bypass, _ = _shared_floor()
    rng = random.Random(seed)
    slots = {}
    for zone in scenario['zones']:
        centre = tuple(zone['center'])
        directions = list(zone['neighbors'])
        rng.shuffle(directions)
        vectors = {'north': (-1, 0), 'east': (0, 1), 'south': (1, 0), 'west': (0, -1)}
        # Farther starts enter the first, smaller fleet. Larger prefixes add
        # robots between them while preserving at least two metres of spacing.
        staging = [(centre[0] + distance * vectors[direction][0],
                    centre[1] + distance * vectors[direction][1])
                   for distance in (8, 4, 6, 2) for direction in directions] + [centre]
        slots[zone['id']] = staging[:9]
        if len(slots[zone['id']]) != 9:
            raise ValueError('Every source zone needs nine distinct staging cells')
    records = [{'robot_id': rank * 9 + zi, 'source_zone': zone['id'],
                'start': slots[zone['id']][rank], 'source_rank': rank}
               for rank in range(9) for zi, zone in enumerate(scenario['zones'])]
    # Each source owns nine pickup bays; small fleets keep the same first
    # records, rather than recomputing a different assignment at each size.
    for zone in scenario['zones']:
        selected = [record for record in records if record['source_zone'] == zone['id']]
        choices = bays[zone['id']]
        costs = np.array([[normal[bay][record['start']] + rng.random() * .1
                           for bay in choices] for record in selected])
        row_ids, col_ids = linear_sum_assignment(costs)
        for row, col in zip(row_ids, col_ids):
            selected[int(row)]['pickup'] = choices[int(col)]
    occupied = {record['pickup'] for record in records}
    candidates = sorted(cell for values in bays.values() for cell in values if cell not in occupied)
    costs = np.full((81, len(candidates)), 1e8, dtype=float)
    for i, record in enumerate(records):
        pickup, source = record['pickup'], record['source_zone']
        source_row, source_col = int(source[1]), int(source[2])
        opposite = f'Z{2-source_row}{2-source_col}'
        for j, goal in enumerate(candidates):
            destination = owners[goal]
            if destination == source:
                continue
            shortest = normal[pickup][goal]
            alternative = bypass[pickup].get(goal)
            if alternative is None:
                continue
            gap = alternative - shortest
            # A preference, not a forced waypoint: PBS can still choose any
            # collision-free route, including the complete outside network.
            costs[i, j] = shortest - 3.0 * gap + (0 if destination == opposite else 60) + rng.random() * .1
    row_ids, col_ids = linear_sum_assignment(costs)
    if len(row_ids) != 81 or any(costs[i, j] >= 1e8 for i, j in zip(row_ids, col_ids)):
        raise ValueError('No globally unique cross-zone delivery assignment exists')
    for row, col in zip(row_ids, col_ids):
        record = records[int(row)]
        goal = candidates[int(col)]
        record.update(dropoff=goal, destination_zone=owners[goal],
                      start_to_pickup_distance_m=normal[record['pickup']][record['start']],
                      shortest_delivery_distance_m=normal[record['pickup']][goal],
                      center_avoiding_distance_m=bypass[record['pickup']][goal])
        record['center_preference_gap_m'] = record['center_avoiding_distance_m'] - record['shortest_delivery_distance_m']
        record['geometric_transport_lower_bound_s'] = record['start_to_pickup_distance_m'] + record['shortest_delivery_distance_m'] + 4.0
        for field in ('start', 'pickup', 'dropoff'):
            record[field] = list(record[field])
    return records


def make_fleet_workload(seed, duration_s=240, fleet_size=54, trigger='body_in_outer'):
    """Return map, scenario, config, tasks, assignment for a real larger fleet."""
    if type(fleet_size) is not int or fleet_size not in FLEET_SIZES:
        raise ValueError('fleet_size must be 27, 54, or 81')
    if trigger not in ('body_in_outer', 'outer_overlap'):
        raise ValueError('Unsupported safety trigger')
    if not math.isfinite(duration_s) or duration_s <= 0 or duration_s * 10 != round(duration_s * 10):
        raise ValueError('Duration must be a positive multiple of 0.1s')
    map_doc, scenario, config, tasks = deepcopy(_shared_floor()[:4])
    geometry = deepcopy(_shared_floor()[-1])
    records = deepcopy(_full_assignment(seed)[:fleet_size])
    zone_ids = [z['id'] for z in scenario['zones']]
    zone_index = {zone: index for index, zone in enumerate(zone_ids)}
    zones = {z['id']: z for z in scenario['zones']}
    allocation = [[0] * 9 for _ in range(9)]
    count = fleet_size // 9
    agents, transports = [], []
    for zone in scenario['zones']:
        zone['all_private_bays'] = [list(cell) for cell in _shared_floor()[5][zone['id']]]
        zone['bays'] = {'starts': [], 'pickups': [], 'dropoffs': []}
    for record in records:
        robot_id, source, destination = record['robot_id'], record['source_zone'], record['destination_zone']
        task_id, payload_id = f'transport-{robot_id:03d}', f'payload-{robot_id:03d}'
        goals = [record['pickup'], record['dropoff']]
        agents.append({'id': robot_id, 'zone_id': source, 'approach': source,
                       'pickup_zone_id': source, 'destination_zone_id': destination,
                       'start': record['start'], 'goals': goals, 'release_at_s': 0,
                       'task_id': task_id, 'payload_id': payload_id, 'color': zones[source]['color']})
        transports.append({'id': task_id, 'robot_id': robot_id, 'payload_id': payload_id,
                           'source_zone_id': source, 'destination_zone_id': destination,
                           'release_at_s': 0, 'payload': {'type': 'tote', 'mass_kg': 5},
                           'stages': [{'index': 0, 'operation': 'pickup', 'zone_id': source,
                                       'location': goals[0], 'nominal_service_time_s': 2},
                                      {'index': 1, 'operation': 'dropoff', 'zone_id': destination,
                                       'location': goals[1], 'nominal_service_time_s': 2}],
                           'finish_behavior': 'park_in_private_dropoff_bay'})
        zones[source]['bays']['starts'].append(record['start'])
        zones[source]['bays']['pickups'].append(record['pickup'])
        zones[destination]['bays']['dropoffs'].append(record['dropoff'])
        allocation[zone_index[source]][zone_index[destination]] += 1
    name = f'nine_zones_nested_fleet_{fleet_size}'
    scenario.update(name=name, agents=agents,
                    description=f'{fleet_size} robots ({count} per zone) on the shared 192-bay floor. Each robot picks up locally, delivers across zones, then parks off the aisle. The 27/54/81 fleets are nested prefixes of one assignment.')
    scenario['safety']['robot_trigger'] = trigger
    scenario['network'].update(robots_per_zone=count, initial_robot_matrix=[[count] * 3 for _ in range(3)],
                               allocation_matrix=allocation)
    scenario['fleet_geometry'] = geometry
    scenario['notes'] = [
        'All fleet sizes use exactly the same 61x61 map, aisle widths, junction cells and safety dimensions.',
        'The shared floor adds 111 private leaf bays to the original 81. This is a new matched floor, not the unchanged original benchmark floor.',
        'All robots face north and start at time zero; larger fleets add robots to an unchanged prefix of starts and tasks.',
        'Each robot has two distinct service goals. Goals are unique across every robot and stage, and parked robots occupy private degree-one bays.',
        'Source populations are balanced. Destination populations can differ because corner zones have fewer private bays.',
        'Delivery assignment prefers central-crossing routes when possible but never imposes a center waypoint; network bypasses stay open.',
        'Payload identity and mass are metadata; the native station controller implements service visits.',
    ]
    tasks.update(scenario=name, tasks=transports, allocation_matrix=allocation,
                 initial_robot_counts={zone: count for zone in zone_ids},
                 initial_robot_matrix=[[count] * 3 for _ in range(3)])
    config['parameters'].update(num_agents=fleet_size, seed=seed, sim_duration=round(duration_s * 10))
    config['wall_timeout_s'] = 1800
    assignment = {'schema_version': 1, 'workload_seed': seed, 'family': 'nested_fleet_on_shared_floor',
                  'fleet_size': fleet_size, 'maximum_fleet_size': 81, 'robots_per_source_zone': count,
                  'prefix_rule': 'Generate all 81 records first in source-zone round-robin order; select IDs 0..N-1.',
                  'floor': geometry, 'assignments': records, 'allocation_matrix': allocation,
                  'strictly_center_preferring_tasks': sum(r['center_preference_gap_m'] > 0 for r in records),
                  'distance_estimate_scope': 'Spatial path lengths at 1m/s plus two nominal 2s service visits; excludes turns, acceleration and interactions.'}
    validate_fleet_case(map_doc, scenario, config, tasks, assignment)
    return map_doc, scenario, config, tasks, assignment


def validate_fleet_case(map_doc, scenario, config, tasks, assignment):
    free = _shared_floor()[4]
    fleet_size = assignment['fleet_size']
    agents = scenario['agents']
    starts = {tuple(a['start']) for a in agents}
    goals = {tuple(g) for a in agents for g in a['goals']}
    if (map_doc != _shared_floor()[0] or len(agents) != fleet_size or len(tasks['tasks']) != fleet_size
            or [a['id'] for a in agents] != list(range(fleet_size))
            or config['parameters']['num_agents'] != fleet_size
            or len(starts) != fleet_size or len(goals) != 2 * fleet_size
            or starts & goals or not (starts | goals) <= free):
        raise ValueError('Fleet size, IDs, map, starts or unique goals disagree')
    if any(sum(n in free for n in neighbors(goal)) != 1 for goal in goals):
        raise ValueError('A terminal goal blocks a through aisle')
    if Counter(a['zone_id'] for a in agents) != Counter({z['id']: fleet_size // 9 for z in scenario['zones']}):
        raise ValueError('Source-zone populations are not balanced')
    if any(a['release_at_s'] != 0 for a in agents) or scenario['initial_orientation'] != 'north':
        raise ValueError('Unsupported release or orientation')
    ordered_starts = [tuple(a['start']) for a in agents]
    if any(math.dist(a, b) < 2 - 1e-12 for i, a in enumerate(ordered_starts) for b in ordered_starts[i + 1:]):
        raise ValueError('Initial robots are spaced less than two metres apart')
    for trigger in ('body_in_outer', 'outer_overlap'):
        check = deepcopy(scenario); check['safety']['robot_trigger'] = trigger
        states = {a['id']: {'row': a['start'][0], 'col': a['start'][1], 'yaw': 0.0} for a in agents}
        responses = _Safety(_fields(check, list(states), True), map_doc).evaluate(states)
        if any(response[0] != 'clear' for response in responses.values()):
            raise ValueError('Initial safety field overlaps a robot or obstacle')
    for agent, task in zip(agents, tasks['tasks']):
        own = {tuple(agent['start']), *(tuple(g) for g in agent['goals'])}
        if not own <= distances(tuple(agent['start']), free - ((starts | goals) - own)).keys():
            raise ValueError('Another stationary start or terminal blocks an individual route')
        if agent['id'] != task['robot_id'] or agent['goals'] != [s['location'] for s in task['stages']]:
            raise ValueError('Task records differ from the native scenario goals')
    return {'valid': True, 'fleet_size': fleet_size, 'initial_fields_clear': True,
            'private_unique_goals': 2 * fleet_size,
            'individual_routes_exist_with_other_starts_and_goals_blocked': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=8301)
    parser.add_argument('--fleet-size', type=int, choices=FLEET_SIZES, default=54)
    parser.add_argument('--duration-s', type=int, default=240)
    parser.add_argument('--trigger', choices=('body_in_outer', 'outer_overlap'), default='body_in_outer')
    args = parser.parse_args()
    case = make_fleet_workload(args.seed, args.duration_s, args.fleet_size, args.trigger)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, value in zip(('map.json', 'scenario.json', 'config.json', 'tasks.json', 'assignment.json'), case):
        write_json(args.output / filename, value)
    (args.output / 'layout.svg').write_text(render_svg(case[0], case[1]))
    load_scenario(args.output / 'scenario.json', case[0]['layout'], args.fleet_size)
    print(args.output.resolve())
