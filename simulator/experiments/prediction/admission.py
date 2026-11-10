"""Bounded entry-gate geometry and commitment protection for joint rollouts.

These helpers mirror EntryGate.cpp and ADG::gateWouldBlockClearance with
protect_clearance=false. They change dispatch eligibility, never action
identities, dependency completion, or physical velocities.
"""

from collections import defaultdict
import math


def contains(point, center, half_width):
    """The native gate uses a closed square, including its boundary."""
    return all(c - half_width <= p <= c + half_width for p, c in zip(point, center))


def enters(start, goal, center, half_width):
    if contains(start, center, half_width):
        return False
    low, high = 0., 1.
    for origin, target, middle in zip(start, goal, center):
        delta = target - origin
        if delta == 0:
            if not middle - half_width <= origin <= middle + half_width:
                return False
        else:
            first = (middle - half_width - origin) / delta
            last = (middle + half_width - origin) / delta
            low, high = max(low, min(first, last)), min(high, max(first, last))
            if low > high:
                return False
    return True


def approaches(start, goal, center, half_width):
    if contains(start, center, half_width):
        return False
    distance = lambda point: sum(max(0., abs(p - c) - half_width) for p, c in zip(point, center))
    return enters(start, goal, center, half_width) or distance(goal) < distance(start)


class AdmissionGate:
    """One fixed relative-time intervention, followed by unrestricted dispatch."""

    def __init__(self, admission, robots, actions, map_document):
        required = {'center', 'half_width', 'held_robot_ids', 'until_offset_s', 'protect_clearance'}
        if not isinstance(admission, dict) or set(admission) != required:
            raise ValueError('Admission requires center, half_width, held_robot_ids, until_offset_s and protect_clearance')
        if admission['protect_clearance'] is not False:
            raise ValueError('Admission supports only bounded protect_clearance=False')
        center = admission['center']
        values = list(center) if isinstance(center, (list, tuple)) else []
        values.extend([admission['half_width'], admission['until_offset_s']])
        if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                                   not math.isfinite(v) for v in values):
            raise ValueError('Admission coordinates and expiry must be finite numbers')
        self.center = tuple(float(v) for v in center)
        self.half_width = float(admission['half_width'])
        self.until_offset_s = float(admission['until_offset_s'])
        if self.half_width <= 0 or self.until_offset_s <= 0:
            raise ValueError('Admission width and relative expiry must be positive')
        held = admission['held_robot_ids']
        if not isinstance(held, (list, tuple)) or any(type(rid) is not int or rid not in robots for rid in held):
            raise ValueError('Admission selected IDs must be known integer robot IDs')
        if len(set(held)) != len(held):
            raise ValueError('Duplicate admission robot IDs')
        self.held = set(held)
        layout = map_document.get('layout', [])
        if layout and any(c - self.half_width < 0 or c + self.half_width > size - 1
                          for c, size in zip(self.center, (len(layout), len(layout[0])))):
            raise ValueError('Admission square is outside the map')
        self.robots, self.actions = robots, actions
        self.successors = defaultdict(list)
        for key, action in actions.items():
            if (key[0], key[1] + 1) in actions:
                self.successors[key].append((key[0], key[1] + 1))
            for dep in action.get('dependencies', []):
                self.successors[tuple(dep)].append(key)
        self.events = []
        self.bypasses = 0
        self.waiting = set()
        self.wait_robot_seconds = 0.

    def protects_other_prefix(self, robot_id, node_id, states, is_done):
        """Native transitive guard, including implicit within-robot ordering."""
        protected_end = {}
        for other, robot in self.robots.items():
            if other != robot_id:
                queued = [node for item in states[other]['queue'] for node in item['node_ids']]
                protected_end[other] = max([robot.get('committed_node_id', -1), *queued])
        pending, visited = [(robot_id, node_id)], set()
        while pending:
            key = pending.pop()
            if key in visited or key not in self.actions or is_done(key):
                continue
            visited.add(key)
            if key[0] != robot_id and key[1] <= protected_end[key[0]]:
                return True
            pending.extend(self.successors[key])
        return False

    def blocks(self, robot_id, action, offset, states, is_done):
        # Rounded native-tick offsets avoid 0.9999999999999999 extending a
        # one-second intervention by an entire one-second polling interval.
        if (round(offset, 10) >= self.until_offset_s or robot_id not in self.held or action['type'] != 'M'
                or action['node_id'] <= self.robots[robot_id].get('committed_node_id', -1)
                or not approaches(action['start'], action['goal'], self.center, self.half_width)):
            return False
        if self.protects_other_prefix(robot_id, action['node_id'], states, is_done):
            self.bypasses += 1
            return False
        return True

    def diagnostics(self):
        return dict(center=list(self.center), half_width=self.half_width,
                    held_robot_ids=sorted(self.held), until_offset_s=self.until_offset_s,
                    protect_clearance=False, dispatch_block_events=len(self.events),
                    commitment_prerequisite_bypasses=self.bypasses,
                    empty_queue_wait_robot_seconds=round(self.wait_robot_seconds, 8),
                    events=self.events,
                    scope='Dispatch-only intervention on current accepted actions; no future replanning. '
                          'Queued/committed prefixes and their prerequisites are protected. '
                          'Uncommitted interior clearance may wait indirectly. '
                          'Denied prefetch can matter without an immediate empty-queue wait.')
