"""Spatially aggregate the *existing* routed slow-occupancy heat map.

9/36/81/144 denote fixed partitions of the 9 native logical warehouse zones.
Every traversable cell receives the mean heat of its corresponding region.
This is post-filter spatial averaging, NOT a new congestion predictor.
"""
from __future__ import annotations

import numpy as np

PARTITIONS = ('cells', 'regions9', 'regions36', 'regions81', 'regions144', 'junction_corridor')


def partition_labels(road, scenario, partition):
    """Return an integer region label for each cell, or -1 for obstacles."""
    if partition not in PARTITIONS or partition == 'cells':
        raise ValueError('partition_labels expects regions9, regions36, regions81 or regions144')
    road = np.asarray(road, dtype=bool)
    if partition == 'junction_corridor':
        return _junction_corridor_labels(road, scenario)
    labels = np.full(road.shape, -1, dtype=np.int32)
    factor = {'regions9': 1, 'regions36': 2, 'regions81': 3, 'regions144': 4}[partition]
    zones = scenario.get('zones', []) if isinstance(scenario, dict) else []
    if len(zones) != 9:
        raise ValueError('Expected the nine zone definitions of Jesse\'s warehouse scenario')
    n = 0
    for zone in zones:
        r0, r1 = zone['bounds']['rows']
        c0, c1 = zone['bounds']['cols']
        row_chunks = np.array_split(np.arange(r0, r1 + 1), factor)
        col_chunks = np.array_split(np.arange(c0, c1 + 1), factor)
        if any(len(chunk) == 0 for chunk in (*row_chunks, *col_chunks)):
            raise ValueError('Partition creates an empty subregion')
        for rows in row_chunks:
            for cols in col_chunks:
                patch = np.ix_(rows, cols)
                if np.any(labels[patch] != -1):
                    raise ValueError('Zone definitions overlap')
                labels[patch] = n
                n += 1
    if n != int(partition.removeprefix('regions')):
        raise ValueError('Wrong number of regions')
    if np.any(labels[road] < 0):
        raise ValueError('Some traversable cells fall outside the nine zones')
    labels[~road] = -1
    return labels


def aggregate_spatial_heat(heat, road, scenario, partition):
    """Replace each road cell's heat with its region's traversable-cell mean."""
    road = np.asarray(road, dtype=bool)
    heat = np.asarray(heat, dtype=float)
    if heat.shape != road.shape or not np.isfinite(heat).all() or np.any(heat < 0):
        raise ValueError('Heat must be finite nonnegative with the same shape as road')
    if partition == 'cells':
        return heat * road
    labels = partition_labels(road, scenario, partition)
    result = np.zeros_like(heat)
    for region in np.unique(labels[road]):
        cells = (labels == region) & road
        result[cells] = heat[cells].mean()
    return result


def _junction_corridor_labels(road, scenario):
    """Create 9 junction regions and 12 corridor regions."""
    from collections import deque

    road = np.asarray(road, dtype=bool)
    zones = scenario.get('zones', [])
    edges = scenario.get('network', {}).get('edges', [])

    if road.ndim != 2 or len(zones) != 9 or len(edges) != 12:
        raise ValueError('Expected the nine-junction warehouse')

    zone_by_id = {zone['id']: zone for zone in zones}
    if len(zone_by_id) != 9:
        raise ValueError('Duplicate junction ID')

    labels = np.full(road.shape, -1, dtype=np.int32)

    # 1. Assign the nine intersection conflict zones.
    for index, zone in enumerate(zones):
        core = zone['conflict_zone']
        r0, r1 = core['rows']
        c0, c1 = core['cols']

        rows = slice(r0, r1 + 1)
        cols = slice(c0, c1 + 1)

        patch = labels[rows, cols]
        mask = road[rows, cols]

        if np.any(patch[mask] >= 0):
            raise ValueError('Overlapping junction regions')

        patch[mask] = index

    # 2. Assign the twelve connecting corridors.
    for index, (a, b) in enumerate(edges, start=9):
        if a not in zone_by_id or b not in zone_by_id:
            raise ValueError('Unknown corridor endpoint')

        ar, ac = zone_by_id[a]['center']
        br, bc = zone_by_id[b]['center']

        if ar == br and abs(ac - bc) >= 4:
            rows = slice(ar - 1, ar + 2)
            cols = slice(min(ac, bc) + 2, max(ac, bc) - 1)

        elif ac == bc and abs(ar - br) >= 4:
            rows = slice(min(ar, br) + 2, max(ar, br) - 1)
            cols = slice(ac - 1, ac + 2)

        else:
            raise ValueError('Invalid corridor geometry')

        patch = labels[rows, cols]
        mask = road[rows, cols]

        if patch.size == 0 or np.any(patch[mask] >= 0):
            raise ValueError('Overlapping corridor regions')

        patch[mask] = index

    # Check that all 21 regions contain road cells.
    assigned = labels[road & (labels >= 0)]
    if np.any(np.bincount(assigned, minlength=21) == 0):
        raise ValueError('An intersection or corridor is empty')

    # 3. Assign remaining road cells (such as side bays)
    # to neighboring regions using road connectivity.
    queue = deque(
        (int(r), int(c))
        for r, c in np.argwhere(road & (labels >= 0))
    )

    while queue:
        r, c = queue.popleft()

        neighbors = (
            (r - 1, c), (r + 1, c),
            (r, c - 1), (r, c + 1)
        )

        for nr, nc in neighbors:
            if (
                0 <= nr < road.shape[0]
                and 0 <= nc < road.shape[1]
                and road[nr, nc]
                and labels[nr, nc] < 0
            ):
                labels[nr, nc] = labels[r, c]
                queue.append((nr, nc))

    if np.any(labels[road] < 0):
        raise ValueError('Unassigned traversable road cells')

    return labels
