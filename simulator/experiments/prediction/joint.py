"""Causal, synchronous fleet rollout with the native safety-field geometry.

This is a kinematic surrogate, not an ARGoS checkpoint. Native controller
snapshots improve queue, polling, acceleration and remaining-service estimates;
the physical contact solver, angular PID and future plans are not reproduced.
Coordinates in returned frames are map row/column; yaw is native ARGoS radians.
"""

from collections import defaultdict
import math


def _angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def _box(rectangle, pose):
    """Asymmetric field -> centred oriented box (x,y,cos,sin,hx,hy)."""
    front, back, width = rectangle
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    shift = (front - back) / 2
    return x + shift * c, y + shift * s, c, s, (front + back) / 2, width


def intersects_boxes(a, b):
    """Closed oriented rectangles: touching edges/corners count as overlap."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    for ax, ay in ((a[2], a[3]), (-a[3], a[2]),
                   (b[2], b[3]), (-b[3], b[2])):
        extent_a = a[4] * abs(ax * a[2] + ay * a[3]) + a[5] * abs(-ax * a[3] + ay * a[2])
        extent_b = b[4] * abs(ax * b[2] + ay * b[3]) + b[5] * abs(-ax * b[3] + ay * b[2])
        if abs(dx * ax + dy * ay) > extent_a + extent_b + 1e-12:
            return False
    return True


def intersects_disc(box, x, y, radius):
    dx, dy = x - box[0], y - box[1]
    local_x, local_y = box[2] * dx + box[3] * dy, -box[3] * dx + box[2] * dy
    outside_x, outside_y = max(abs(local_x) - box[4], 0), max(abs(local_y) - box[5], 0)
    return outside_x ** 2 + outside_y ** 2 <= radius ** 2 + 1e-12


class _Bins:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = defaultdict(list)

    def add(self, key, x, y):
        self.cells[math.floor(x / self.cell_size), math.floor(y / self.cell_size)].append(key)

    def nearby(self, x, y, radius):
        scale = self.cell_size
        for i in range(math.floor((x - radius) / scale), math.floor((x + radius) / scale) + 1):
            for j in range(math.floor((y - radius) / scale), math.floor((y + radius) / scale) + 1):
                yield from self.cells.get((i, j), ())


def _fields(scenario, ids, enabled):
    defaults = scenario.get("safety", {})
    agents = {a["id"]: a for a in scenario.get("agents", [])}
    result = {}
    for robot_id in ids:
        settings = dict(defaults)
        settings.update(agents.get(robot_id, {}).get("safety", {}))
        slow = tuple(float(settings.get(k, v)) for k, v in (
            ("slow_front_m", 1.2), ("slow_back_m", .2), ("slow_half_width_m", .3)))
        stop = tuple(float(settings.get(k, v)) for k, v in (
            ("stop_front_m", .3), ("stop_back_m", .15), ("stop_half_width_m", .18)))
        scale, radius = float(settings.get("speed_scale", .2)), float(settings.get("robot_radius_m", .085))
        inner = bool(settings.get("inner_stop_enabled", True))
        trigger = settings.get("robot_trigger", "body_in_outer")
        if trigger not in ("body_in_outer", "outer_overlap"):
            raise ValueError(f"Unknown safety trigger {trigger!r}")
        if (not all(math.isfinite(v) and v > 0 for v in (*slow, *stop, radius, scale))
                or scale >= 1 or any(v < radius for v in slow)
                or (inner and any(a > b for a, b in zip(stop, slow)))):
            raise ValueError("Invalid safety field dimensions, nesting or speed scale")
        result[robot_id] = dict(slow=slow, stop=stop, radius=radius, scale=scale, inner=inner,
                               trigger=trigger, enabled=bool(enabled and defaults.get("enabled", False)
                                                             and settings.get("enabled", True)),
                               extent=math.hypot(max(slow[:2]), slow[2]))
    return result


class _Safety:
    def __init__(self, fields, map_document, robot_interactions=True):
        self.fields = fields
        self.robot_interactions = robot_interactions
        self.extent = max((f["extent"] for f in fields.values()), default=1)
        self.bin_size = max(1, self.extent)
        self.obstacles = []
        self.obstacle_bins = _Bins(self.bin_size)
        layout = map_document.get("layout", [])
        for row, line in enumerate(layout):
            for col, cell in enumerate(line):
                if cell in "@T":
                    self.obstacle_bins.add(len(self.obstacles), -row, -col)
                    self.obstacles.append((-row, -col, 1., 0., .4, .4))
        # ArgosConfig/ToArgos.py also emits these four 5 cm arena boundaries.
        self.boundaries = []
        if layout:
            h, w = len(layout), len(layout[0])
            self.boundaries = [(.5, -w / 2 + .5, 1., 0., .025, w / 2),
                               (-h + .5, -w / 2 + .5, 1., 0., .025, w / 2),
                               (-h / 2 + .5, .5, 1., 0., h / 2, .025),
                               (-h / 2 + .5, -w + .5, 1., 0., h / 2, .025)]

    def evaluate(self, states):
        bins = _Bins(self.bin_size)
        slow_boxes, stop_boxes = {}, {}
        for robot_id, state in states.items():
            pose = -state["row"], -state["col"], state["yaw"]
            bins.add(robot_id, pose[0], pose[1])
            slow_boxes[robot_id] = _box(self.fields[robot_id]["slow"], pose)
            stop_boxes[robot_id] = _box(self.fields[robot_id]["stop"], pose)
        decisions = {}
        for robot_id, state in states.items():
            field = self.fields[robot_id]
            level, reasons = 0, set()
            if field["enabled"]:
                outer, inner = slow_boxes[robot_id], stop_boxes[robot_id]
                x, y = -state["row"], -state["col"]
                others = bins.nearby(x, y, field["extent"] + self.extent) if self.robot_interactions else ()
                for other in others:
                    if other == robot_id:
                        continue
                    target, target_field = states[other], self.fields[other]
                    tx, ty = -target["row"], -target["col"]
                    if field["inner"] and intersects_disc(inner, tx, ty, target_field["radius"]):
                        level = 2
                        reasons.add("body_in_inner")
                    hit = (intersects_boxes(outer, slow_boxes[other]) if field["trigger"] == "outer_overlap"
                           else intersects_disc(outer, tx, ty, target_field["radius"]))
                    if hit:
                        level = max(level, 1)
                        reasons.add("outer_polygon_overlap" if field["trigger"] == "outer_overlap" else "body_in_outer")
                boxes = (self.obstacles[index] for index in self.obstacle_bins.nearby(x, y, field["extent"] + .57))
                for box in (*boxes, *self.boundaries):
                    if field["inner"] and intersects_boxes(inner, box):
                        level = 2
                        reasons.add("obstacle_in_inner")
                    if intersects_boxes(outer, box):
                        level = max(level, 1)
                        reasons.add("obstacle_in_outer")
            decisions[robot_id] = (("clear", "slow", "stop")[level],
                                   0. if level == 2 else field["scale"] if level else 1.,
                                   sorted(reasons))
        return decisions


def _passed(state, action):
    start, goal = action["start"], action["goal"]
    dr, dc = goal[0] - start[0], goal[1] - start[1]
    length = math.hypot(dr, dc)
    if length < 1e-12:
        return math.hypot(state["row"] - goal[0], state["col"] - goal[1]) < 1e-9
    along = ((state["row"] - start[0]) * dr + (state["col"] - start[1]) * dc) / length
    across = abs((state["row"] - start[0]) * dc - (state["col"] - start[1]) * dr) / length
    return along >= length - 1e-9 and across <= .15


def remaining_nominal_work(state, pending_actions, *, station_dwell_s=2, turn_rate_deg_s=90):
    """Optimistic remaining time along this route, without safety/dependency delay.

    Current merged queue groups retain their actual remaining service timers;
    future actions use the supplied nominal dwell. MOVE distance is remaining
    distance, always nonnegative, and explicit TURNs pay their remaining angle.
    Acceleration, polling, contact physics and unknown future plans are omitted.
    """
    speed = float(state['nominal_speed'])
    rate = math.radians(turn_rate_deg_s)
    if not all(math.isfinite(v) and v > 0 for v in (speed, rate)):
        raise ValueError('Nominal remaining work requires positive finite speed and turn rate')
    if not math.isfinite(station_dwell_s) or station_dwell_s < 0:
        raise ValueError('Nominal remaining work requires nonnegative finite dwell')
    row, col, yaw = state['row'], state['col'], state['yaw']
    groups = list(state['queue']) + [dict(type=a['type'], goal=a['goal'],
                yaw=-float(a.get('orientation', 0)) * math.pi / 2,
                remaining_s=station_dwell_s if a['type'] == 'S' else 0.) for a in pending_actions]
    total = 0.
    for group in groups:
        if group['type'] == 'M':
            dr, dc = group['goal'][0] - row, group['goal'][1] - col
            distance = math.hypot(dr, dc)
            total += distance / speed
            if distance > 1e-12:
                heading = math.atan2(-dc, -dr)
                yaw = heading if abs(_angle(heading - yaw)) <= abs(_angle(heading + math.pi - yaw)) else _angle(heading + math.pi)
            row, col = group['goal']
        elif group['type'] == 'T':
            total += abs(_angle(group['yaw'] - yaw)) / rate
            yaw = group['yaw']
        elif group['type'] == 'S':
            total += max(0., group['remaining_s'])
        else:
            raise ValueError('Unknown action in nominal remaining route work')
    if not math.isfinite(total):
        raise ValueError('Nonfinite nominal remaining route work')
    return total


def rollout(snapshot, pose_frame, scenario, map_document, *, horizon_s=20, dt=.1,
            speed_m_s=1, turn_rate_deg_s=90, station_dwell_s=2, controllers=None,
            safety_enabled=True, robot_interactions=True, admission=None,
            record_progress=False, native_poll_before_completion=False,
            record_remaining_work=False):
    """Advance only the actions known at the origin; never reads future truth.

    With native ``controllers`` records, reproduce ten-step action polling,
    five-node dispatch capacity, observed merged queues and service timers, and
    the source controller's longitudinal reference-speed/acceleration law.
    Without records, use constant-speed moves and immediate dependency release.
    Both modes rotate fields using predicted headings and use common pre-step
    poses for all safety decisions. Dependency completions are also published
    together, so changing input robot order cannot change the forecast.

    Optional ``admission`` selects robot IDs whose new, uncommitted approach
    MOVEs may be withheld at dispatch until ``until_offset_s``. It uses the
    native bounded gate and commitment protection, without changing actions or
    overriding velocity. Controllers are required for this counterfactual.
    ``None`` preserves the original rollout and diagnostic format exactly.
    ``record_progress=True`` adds cumulative accepted-work seconds per robot:
    signed forward route distance / nominal speed, intended TURN advancement /
    turn rate, and consumed SERVICE time. Exhausting a prefix is not delivery.
    New counterfactual comparisons can set ``native_poll_before_completion``
    for both no-op and candidates. Native controllers poll before acknowledging
    passed queue nodes; the historical default uses the opposite order. Both
    modes publish completions synchronously, unlike native callback ordering.
    ``record_remaining_work=True`` adds initial and terminal per-robot nominal
    remaining route time, for comparing different routes to the same service.
    """
    values = (horizon_s, dt, speed_m_s, turn_rate_deg_s, station_dwell_s)
    if type(record_progress) is not bool:
        raise ValueError("record_progress must be a boolean")
    if type(native_poll_before_completion) is not bool:
        raise ValueError("native_poll_before_completion must be a boolean")
    if type(record_remaining_work) is not bool:
        raise ValueError("record_remaining_work must be a boolean")
    if not all(math.isfinite(v) for v in values) or min(values[:4]) <= 0 or station_dwell_s < 0:
        raise ValueError("Positive finite horizon, time step and speeds, and nonnegative dwell required")
    if pose_frame["clock"] > snapshot["clock"]:
        raise ValueError("A forecast cannot use a pose from after its origin")
    if not math.isclose(float(scenario.get("cell_size_m", 1)), 1):
        raise ValueError("Native map geometry currently requires one-metre grid coordinates")
    native_dt = 1 / float(scenario.get("ticks_per_second", 10))
    if controllers is not None and not math.isclose(dt, native_dt, abs_tol=1e-9):
        raise ValueError("Controller-aware rollout requires the native simulation time step")
    actual = {a["id"]: a for a in pose_frame["agents"]}
    control = {c["id"]: c for c in controllers or []}
    robots = {r["id"]: r for r in snapshot["robots"]}
    if len(robots) != len(snapshot["robots"]) or len(actual) != len(pose_frame["agents"]):
        raise ValueError("Duplicate robot IDs")
    if set(robots) != set(actual):
        raise ValueError("Snapshot and origin pose robot IDs must match")
    if controllers is not None and set(control) != set(robots):
        raise ValueError("Controller records must include every origin robot")
    if admission is not None and controllers is None:
        raise ValueError("Admission counterfactual requires native controller records")
    if native_poll_before_completion and controllers is None:
        raise ValueError("Native polling order requires native controller records")
    states, actions, pending, completed, finished = {}, {}, {}, set(), {}
    missing_dependencies = set()
    dispatched_count = committed_count = 0
    for robot_id in sorted(robots):
        robot, pose, controller = robots[robot_id], actual[robot_id], control.get(robot_id)
        finished[robot_id] = robot["finished_node_id"]
        known = sorted((dict(a) for a in robot["actions"] if a["node_id"] > finished[robot_id]), key=lambda a: a["node_id"])
        for action in known:
            if action["type"] not in ("M", "T", "S"):
                raise ValueError(f"Unsupported action type {action['type']!r}")
            key = robot_id, action["node_id"]
            if key in actions:
                raise ValueError(f"Duplicate action node {key}")
            actions[key] = action
            committed_count += action["node_id"] <= robot.get("committed_node_id", -1)
        state = dict(row=-float(pose["x"]), col=-float(pose["y"]), yaw=float(pose.get("rz", 0)),
                     speed_m_s=float(controller.get("speed_m_s", 0)) if controller else 0.,
                     previous_move_speed=float(controller.get("previous_move_speed_m_s", 0)) if controller else 0.,
                     queue=[], poll_phase=int(controller.get("poll_phase", 0)) if controller else 0,
                     nominal_speed=float(controller.get("nominal_speed_m_s", speed_m_s)) if controller else speed_m_s,
                     acceleration=float(controller.get("acceleration_m_s2", 2)) if controller else None,
                     exhausted_at=None)
        if not all(math.isfinite(state[k]) for k in ("row", "col", "yaw", "speed_m_s", "previous_move_speed", "nominal_speed")):
            raise ValueError("Nonfinite origin state")
        if state["nominal_speed"] <= 0 or (controller and (not math.isfinite(state["acceleration"]) or state["acceleration"] <= 0)):
            raise ValueError("Native speed and acceleration must be positive finite values")
        states[robot_id] = state
        queued_nodes = set()
        if controller:
            if controller.get("clock", snapshot["clock"]) > snapshot["clock"]:
                raise ValueError("Controller state is from after the origin")
            for item in controller.get("queue", []):
                ids = [node for node in item["node_ids"] if node > finished[robot_id]]
                if not ids:
                    continue
                if any((robot_id, node) not in actions for node in ids):
                    raise ValueError("Controller queue contains a node absent from the action snapshot")
                queued_nodes.update(ids)
                state["queue"].append(dict(type=item["type"], node_ids=list(ids), goal=list(item["goal"]),
                                           yaw=float(item["orientation_rad"]),
                                           remaining_s=max(0., float(item.get("remaining_ticks", 0)) * native_dt)))
        else:
            for action in known:
                if action["node_id"] <= robot["enqueued_node_id"]:
                    _append_action(state, action, station_dwell_s)
                    queued_nodes.add(action["node_id"])
        dispatched_count += len(queued_nodes)
        pending[robot_id] = [a for a in known if a["node_id"] not in queued_nodes]
        # A native snapshot is coherent: dispatched-but-unrecorded nodes would
        # be a broken input, not something to silently replay a second time.
        if controller and any(a["node_id"] <= robot["enqueued_node_id"] for a in pending[robot_id]):
            raise ValueError("Dispatched action missing from native controller queue")
    fields = _fields(scenario, states, safety_enabled)
    safety = _Safety(fields, map_document, robot_interactions)
    completions, frames = [], []
    progress = {robot_id: 0. for robot_id in states} if record_progress else None
    counters = dict(slow_move_robot_seconds=0., stopped_command_robot_seconds=0., dependency_wait_robot_seconds=0.)
    native = controllers is not None
    def remaining_work():
        return {str(robot_id): remaining_nominal_work(state, pending[robot_id],
                    station_dwell_s=station_dwell_s, turn_rate_deg_s=turn_rate_deg_s)
                for robot_id, state in states.items()}
    initial_remaining_work = remaining_work() if record_remaining_work else None
    gate = None
    if admission is not None:
        try:
            from .admission import AdmissionGate
        except ImportError:
            from admission import AdmissionGate
        gate = AdmissionGate(admission, robots, actions, map_document)
        expiry_ticks = gate.until_offset_s / native_dt
        if not math.isclose(expiry_ticks, round(expiry_ticks), abs_tol=1e-9):
            raise ValueError("Admission expiry must be an integer number of native ticks")
    for robot_id, todo in pending.items():
        for action in todo:
            for other, node in action.get("dependencies", []):
                if node > finished.get(other, -1) and (other, node) not in actions:
                    missing_dependencies.add((other, node))

    def is_done(key):
        return key[1] <= finished.get(key[0], -1) or key in completed

    def record_finished(robot_id, group, nodes, offset):
        for node in nodes:
            key = robot_id, node
            if key not in completed:
                completed.add(key)
                completions.append(dict(id=robot_id, node_id=node, offset_s=round(offset, 10), type=group["type"]))

    def settle(offset):
        # Completion detection has no cross-robot reads; apply all completions
        # before anyone polls for newly released actions at this boundary.
        for robot_id, state in states.items():
            queue = state["queue"]
            while queue:
                item, done = queue[0], False
                if item["type"] == "M":
                    distance = math.hypot(state["row"] - item["goal"][0], state["col"] - item["goal"][1])
                    while len(item["node_ids"]) > 1:
                        action = actions[robot_id, item["node_ids"][0]]
                        if not (_passed(state, action) or (native and distance - .5 * (len(item["node_ids"]) - 1) < .03)):
                            break
                        record_finished(robot_id, item, [item["node_ids"].pop(0)], offset)
                        if native:
                            break
                    done = (distance < (.03 if native else 1e-8)
                            and (not native or abs(state["previous_move_speed"]) <= native_dt * state["nominal_speed"]))
                    if not native and _passed(state, actions[robot_id, item["node_ids"][-1]]):
                        done = True
                elif item["type"] == "T":
                    done = abs(_angle(item["yaw"] - state["yaw"])) < (math.radians(.5) if native else 1e-8)
                else:
                    done = item["remaining_s"] <= 1e-9
                if not done:
                    break
                record_finished(robot_id, item, list(item["node_ids"]), offset)
                queue.pop(0)

    def admit():
        for robot_id, state in states.items():
            if native and state["poll_phase"] != 0:
                continue
            outstanding = sum(len(item["node_ids"]) for item in state["queue"])
            queued_at_poll = outstanding
            if gate is not None:
                gate.waiting.discard(robot_id)
            while pending[robot_id] and (not native or outstanding < 5):
                action = pending[robot_id][0]
                if not all(is_done(tuple(dep)) for dep in action.get("dependencies", [])):
                    break
                if gate is not None and gate.blocks(robot_id, action, offset, states, is_done):
                    gate.events.append(dict(offset_s=round(offset, 10), robot_id=robot_id,
                                            node_id=action['node_id'], queued_action_count=queued_at_poll,
                                            dispatchable_before_gate=outstanding - queued_at_poll))
                    if outstanding == 0:
                        gate.waiting.add(robot_id)
                    break
                _append_action(state, action, station_dwell_s)
                pending[robot_id].pop(0)
                outstanding += 1

    def save(offset, decisions):
        agents = []
        for robot_id, state in states.items():
            kind = state["queue"][0]["type"] if state["queue"] else "WAIT" if pending[robot_id] else "EXHAUSTED"
            if kind == "EXHAUSTED" and state["exhausted_at"] is None:
                state["exhausted_at"] = round(offset, 10)
            mode, _, reasons = decisions[robot_id]
            agents.append(dict(id=robot_id, row=state["row"], col=state["col"], yaw=state["yaw"],
                               speed_m_s=state["speed_m_s"], mode=mode, action_type=kind,
                               trigger_reasons=reasons))
            if progress is not None:
                agents[-1]['accepted_work_done_s'] = progress[robot_id]
        frames.append(dict(offset_s=round(offset, 10), agents=agents))

    offset = 0.
    if native_poll_before_completion:
        admit()
        settle(offset)
    else:
        settle(offset)
        admit()
        settle(offset)
    decisions = safety.evaluate(states)
    save(offset, decisions)
    while offset < horizon_s - 1e-10:
        step = min(dt, horizon_s - offset)
        for robot_id, state in states.items():
            mode, scale, _ = decisions[robot_id]
            item = state["queue"][0] if state["queue"] else None
            state["speed_m_s"] = 0.
            if not item:
                if pending[robot_id]:
                    if gate is not None and robot_id in gate.waiting:
                        gate.wait_robot_seconds += step
                    else:
                        counters["dependency_wait_robot_seconds"] += step
            elif scale == 0 and item["type"] in ("M", "T"):
                state["previous_move_speed"] = 0.
                counters["stopped_command_robot_seconds"] += step
            elif item["type"] == "M":
                dr, dc = item["goal"][0] - state["row"], item["goal"][1] - state["col"]
                distance = math.hypot(dr, dc)
                heading = math.atan2(-dc, -dr)
                forward = abs(_angle(heading - state["yaw"])) <= abs(_angle(heading + math.pi - state["yaw"]))
                target_yaw = heading if forward else _angle(heading + math.pi)
                sign = 1 if forward else -1
                # Small heading correction is approximated; translation follows
                # the accepted straight segment, rather than a contact solver.
                change = max(-math.radians(turn_rate_deg_s) * step,
                             min(math.radians(turn_rate_deg_s) * step, _angle(target_yaw - state["yaw"])))
                state["yaw"] = _angle(state["yaw"] + change)
                nominal, cap = state["nominal_speed"], state["nominal_speed"] * scale
                if native:
                    # Exact source unit conversion: getReferenceSpeed receives
                    # metres but returns cm/s, with acceleration in cm/s².
                    reference = min(math.sqrt(2 * state["acceleration"] * distance / (100 * native_dt)), nominal)
                    wanted = sign * min(reference, cap)
                    difference = .6 * (wanted - state["previous_move_speed"])
                    limit = state["acceleration"] * step
                    velocity = state["previous_move_speed"] + max(-limit, min(limit, difference))
                    velocity = max(-cap, min(cap, velocity))
                    state["previous_move_speed"] = velocity
                    travel = min(distance, max(0., sign * velocity) * step)
                    if scale < 1 and (reference > cap or abs(velocity) >= cap - 1e-9):
                        counters["slow_move_robot_seconds"] += step
                else:
                    travel = min(distance, cap * step)
                    state["previous_move_speed"] = sign * cap
                    if scale < 1 and distance > 1e-9:
                        counters["slow_move_robot_seconds"] += step
                if distance > 1e-12:
                    state["row"] += dr * travel / distance
                    state["col"] += dc * travel / distance
                    if progress is not None:
                        # Credit signed advancement in the accepted segment's
                        # direction, not absolute motion or heading correction.
                        start = actions[robot_id, item['node_ids'][0]]['start']
                        route_dr, route_dc = item['goal'][0] - start[0], item['goal'][1] - start[1]
                        route_length = math.hypot(route_dr, route_dc)
                        if route_length > 1e-12:
                            progress[robot_id] += travel * (dr * route_dr + dc * route_dc) / (distance * route_length * nominal)
                state["speed_m_s"] = sign * travel / step
            elif item["type"] == "T":
                turn_before = abs(_angle(item["yaw"] - state["yaw"])) if progress is not None else None
                change = max(-math.radians(turn_rate_deg_s) * step,
                             min(math.radians(turn_rate_deg_s) * step, _angle(item["yaw"] - state["yaw"])))
                state["yaw"] = _angle(state["yaw"] + change)
                if progress is not None:
                    progress[robot_id] += max(0., turn_before - abs(_angle(item['yaw'] - state['yaw']))) / math.radians(turn_rate_deg_s)
            elif item["type"] == "S":
                if progress is not None:
                    progress[robot_id] += min(item['remaining_s'], step)
                item["remaining_s"] = max(0., item["remaining_s"] - step)
            if native:
                state["poll_phase"] = (state["poll_phase"] + 1) % 10
        offset = min(horizon_s, offset + step)
        if native_poll_before_completion:
            admit()
            settle(offset)
        else:
            settle(offset)
            admit()
            settle(offset)
        decisions = safety.evaluate(states)
        save(offset, decisions)
    diagnostics = dict(model="joint_safety_kinematic_rollout", controller_state_available=native,
                       motion_model="native_longitudinal_surrogate" if native else "constant_speed_fallback",
                       safety_enabled=bool(safety_enabled and scenario.get("safety", {}).get("enabled", False)),
                       robot_interactions=bool(robot_interactions),
                       dt_s=dt, horizon_s=horizon_s, robot_count=len(states),
                       unfinished_action_count=len(actions), initially_dispatched_action_count=dispatched_count,
                       committed_action_count=committed_count, completed_action_count=len(completed),
                       unresolved_action_count=len(actions) - len(completed),
                       missing_dependencies=[list(key) for key in sorted(missing_dependencies)],
                       plan_exhausted_at_s={str(i): s["exhausted_at"] for i, s in states.items()},
                       **{key: round(value, 8) for key, value in counters.items()},
                       assumptions=["Accepted actions remain fixed; no future replanning or unrecorded tasks.",
                                    "Safety fields use native geometry; touching counts, disabled peers retain geometry.",
                                    "Turn rate and MOVE heading corrections are kinematic approximations, not native angular PID.",
                                    "No physical contact solver, sensor error, network delay or random execution perturbations.",
                                    "Controller-aware polling publishes completions synchronously; native robot callback order can differ."])
    if gate is not None:
        diagnostics['admission'] = gate.diagnostics()
    if progress is not None:
        diagnostics['progress_definition'] = ('Cumulative accepted-work seconds from this origin: signed route MOVE '
                                             'distance / nominal speed, intended TURN advancement / turn rate, '
                                             'and SERVICE timer consumption. No credit after accepted-prefix '
                                             'exhaustion, and no inference of delivery completion.')
    if native_poll_before_completion:
        diagnostics['dispatch_order'] = 'native_poll_before_completion'
        diagnostics['assumptions'].append('Poll before queue completion acknowledgements; cross-robot callback ordering remains synchronous.')
    if record_remaining_work:
        diagnostics['nominal_remaining_work_s'] = dict(initial=initial_remaining_work, terminal=remaining_work(),
            definition='Remaining route MOVE length / nominal speed + explicit TURN angle / turn rate + remaining SERVICE time. '
                       'Optimistic tail omits future safety, dependency, polling and acceleration delays; prefix exhaustion is not delivery.')
    return dict(frames=frames, completions=completions, diagnostics=diagnostics)


def _append_action(state, action, station_dwell_s):
    queue = state["queue"]
    if action["type"] == "M" and queue and queue[-1]["type"] == "M":
        queue[-1]["goal"] = list(action["goal"])
        queue[-1]["node_ids"].append(action["node_id"])
        return
    queue.append(dict(type=action["type"], node_ids=[action["node_id"]], goal=list(action["goal"]),
                      yaw=-float(action.get("orientation", 0)) * math.pi / 2,
                      remaining_s=station_dwell_s if action["type"] == "S" else 0.))
