"""Behavioral tests of coupled safety, causal state and synchronous execution."""

from copy import deepcopy
import math
import unittest

try:
    from .joint import rollout, _box, intersects_boxes, intersects_disc
    from .test_predict import action, robot, snapshot, frame
except ImportError:
    from joint import rollout, _box, intersects_boxes, intersects_disc
    from test_predict import action, robot, snapshot, frame


def scenario(trigger="body_in_outer", **settings):
    return {"ticks_per_second": 10, "safety": {"enabled": True, "robot_trigger": trigger,
            "inner_stop_enabled": False, "slow_front_m": .7, "slow_back_m": .7,
            "slow_half_width_m": .3, "speed_scale": .2, **settings}}


def controller(robot_id, queue=(), **values):
    return {"id": robot_id, "clock": 0, "poll_phase": 0, "queue": list(queue),
            "speed_m_s": 0., "previous_move_speed_m_s": 0.,
            "nominal_speed_m_s": 1., "acceleration_m_s2": 2., **values}


def sample(result, seconds, robot_id=0):
    record = next(f for f in result["frames"] if math.isclose(f["offset_s"], seconds))
    return next(a for a in record["agents"] if a["id"] == robot_id)


class GeometryTests(unittest.TestCase):
    def test_asymmetric_rotated_field_matches_local_axes(self):
        field = _box((1.2, .2, .3), (0, 0, math.pi / 2))
        self.assertTrue(intersects_disc(field, 0, 1.2, .085))
        self.assertFalse(intersects_disc(field, 0, -.4, .085))
        self.assertFalse(intersects_disc(field, .5, .4, .085))

    def test_edge_contact_and_small_gap(self):
        field = _box((.5, .5, .5), (0, 0, 0))
        touching = _box((.5, .5, .5), (1, 0, 0))
        separated = _box((.5, .5, .5), (1.00001, 0, 0))
        self.assertTrue(intersects_boxes(field, touching))
        self.assertFalse(intersects_boxes(field, separated))


class JointRolloutTests(unittest.TestCase):
    def test_isolated_move_progress_and_plan_exhaustion(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 6])])])
        result = rollout(state, frame(0, [(4, 4)]), scenario(), {}, horizon_s=3)
        self.assertAlmostEqual(sample(result, 1)["col"], 5)
        self.assertAlmostEqual(sample(result, 3)["col"], 6)
        self.assertEqual(result["completions"], [{"id": 0, "node_id": 0, "offset_s": 2., "type": "M"}])
        self.assertEqual(result["diagnostics"]["plan_exhausted_at_s"], {"0": 2.})

    def test_following_fields_slow_both_without_body_entry(self):
        state = snapshot([robot(i, [action(0, [4 + i, 4], [-1 + i, 4])]) for i in range(2)])
        poses = frame(0, [(4, 4), (5, 4)])
        body = rollout(state, poses, scenario(), {}, horizon_s=1)
        overlap = rollout(state, poses, scenario("outer_overlap"), {}, horizon_s=1)
        self.assertAlmostEqual(sample(body, 1)["row"], 3)
        self.assertAlmostEqual(sample(overlap, 1)["row"], 3.8)
        self.assertEqual(sample(overlap, 0)["mode"], "slow")
        self.assertEqual(sample(body, 0)["mode"], "clear")
        self.assertAlmostEqual(overlap["diagnostics"]["slow_move_robot_seconds"], 2)

    def test_delay_propagates_through_cross_robot_dependency(self):
        state = snapshot([
            robot(0, [action(0, [2, 2], [1.5, 2])]),
            robot(1, [action(0, [4, 0], [4, 2], dependencies=[[0, 0]])]),
            robot(2, []),
        ])
        poses = frame(0, [(2, 2), (4, 0), (1, 2)])
        body = rollout(state, poses, scenario(), {}, horizon_s=2)
        overlap = rollout(state, poses, scenario("outer_overlap"), {}, horizon_s=2)
        self.assertGreater(sample(body, 2, 1)["col"], sample(overlap, 2, 1)["col"])
        self.assertEqual(sample(overlap, 2, 1)["action_type"], "WAIT")

    def test_order_independence_and_no_input_mutation(self):
        state = snapshot([robot(i, [action(0, [4 + i, 4], [-1 + i, 4])]) for i in range(2)])
        poses, settings = frame(0, [(4, 4), (5, 4)]), scenario("outer_overlap")
        original = deepcopy((state, poses, settings))
        first = rollout(state, poses, settings, {}, horizon_s=2)
        self.assertEqual((state, poses, settings), original)
        state["robots"].reverse()
        poses["agents"].reverse()
        second = rollout(state, poses, settings, {}, horizon_s=2)
        self.assertEqual(first, second)

    def test_unknown_dependency_blocks_unless_already_dispatched(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 5], dependencies=[[9, 7]])], committed=0)])
        result = rollout(state, frame(0, [(4, 4)]), scenario(), {}, horizon_s=1)
        self.assertEqual(sample(result, 1)["col"], 4)
        self.assertEqual(result["diagnostics"]["missing_dependencies"], [[9, 7]])
        state["robots"][0]["enqueued_node_id"] = 0
        result = rollout(state, frame(0, [(4, 4)]), scenario(), {}, horizon_s=1)
        self.assertAlmostEqual(sample(result, 1)["col"], 5)

    def test_partial_dispatched_move_never_reverses_to_passed_waypoint(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 4.5]), action(1, [4, 4.5], [4, 5])], enqueued=1)])
        result = rollout(state, frame(0, [(4, 4.7)]), scenario(), {}, horizon_s=.1)
        self.assertAlmostEqual(sample(result, .1)["col"], 4.8)
        self.assertEqual(result["completions"][0]["offset_s"], 0)

    def test_optional_inner_stop_blocks_move_and_turn(self):
        state = snapshot([robot(0, [action(0, [4, 4], [4, 4], kind="T", orientation=1)]), robot(1, [])])
        settings = scenario(inner_stop_enabled=True, stop_front_m=.3, stop_back_m=.15, stop_half_width_m=.18)
        result = rollout(state, frame(0, [(4, 4), (3.8, 4)]), settings, {}, horizon_s=1)
        self.assertEqual(sample(result, 1)["yaw"], 0)
        self.assertEqual(sample(result, 0)["mode"], "stop")
        settings["safety"]["inner_stop_enabled"] = False
        result = rollout(state, frame(0, [(4, 4), (3.8, 4)]), settings, {}, horizon_s=1)
        self.assertAlmostEqual(sample(result, 1)["yaw"], -math.pi / 2)

    def test_obstacle_and_arena_boundaries_still_trigger_when_robots_removed(self):
        state = snapshot([robot(0, [action(0, [3, 3], [1, 3])]), robot(1, [])])
        poses = frame(0, [(3, 3), (4, 3)])
        layout = {"layout": [".......", ".......", "...@...", ".......", ".......", "......."]}
        result = rollout(state, poses, scenario("outer_overlap", slow_front_m=1.2), layout,
                         horizon_s=.1, robot_interactions=False)
        origin = sample(result, 0)
        self.assertEqual(origin["trigger_reasons"], ["obstacle_in_outer"])
        self.assertAlmostEqual(sample(result, .1)["row"], 2.98)

    def test_disabled_robot_keeps_its_field_as_other_robots_obstacle(self):
        state = snapshot([robot(i, [action(0, [4 + i, 4], [-1 + i, 4])]) for i in range(2)])
        settings = scenario("outer_overlap")
        settings["agents"] = [{"id": 1, "safety": {"enabled": False}}]
        result = rollout(state, frame(0, [(4, 4), (5, 4)]), settings, {}, horizon_s=.1)
        self.assertEqual(sample(result, 0, 0)["mode"], "slow")
        self.assertEqual(sample(result, 0, 1)["mode"], "clear")

    def test_actual_remaining_station_timer_releases_dependency(self):
        state = snapshot([robot(0, [action(0, [3, 3], [3, 3], kind="S")], enqueued=0),
                          robot(1, [action(0, [8, 8], [8, 9], dependencies=[[0, 0]])])])
        controllers = [controller(0, [{"type": "S", "node_ids": [0], "goal": [3, 3],
                                      "orientation_rad": 0, "remaining_ticks": 3}]), controller(1)]
        result = rollout(state, frame(0, [(3, 3), (8, 8)]), scenario(), {}, horizon_s=1.5, controllers=controllers)
        self.assertEqual(result["completions"][0]["offset_s"], .3)
        self.assertEqual(sample(result, .9, 1)["col"], 8)
        self.assertGreater(sample(result, 1.5, 1)["col"], 8)
        self.assertEqual(controllers[0]["queue"][0]["remaining_ticks"], 3)

    def test_native_longitudinal_motion_starts_at_observed_speed(self):
        state = snapshot([robot(0, [action(0, [10, 10], [0, 10])], enqueued=0)])
        queue = [{"type": "M", "node_ids": [0], "goal": [0, 10], "orientation_rad": 0, "remaining_ticks": 0}]
        from_rest = rollout(state, frame(0, [(10, 10)]), scenario(), {}, horizon_s=.1,
                            controllers=[controller(0, queue)])
        moving = rollout(state, frame(0, [(10, 10)]), scenario(), {}, horizon_s=.1,
                         controllers=[controller(0, queue, speed_m_s=1., previous_move_speed_m_s=1.)])
        self.assertAlmostEqual(sample(from_rest, .1)["row"], 9.98)
        self.assertAlmostEqual(sample(moving, .1)["row"], 9.9)

    def test_native_dispatch_capacity_and_polling_are_preserved(self):
        known = [action(i, [10 - .5 * i, 10], [9.5 - .5 * i, 10]) for i in range(12)]
        state = snapshot([robot(0, known)])
        result = rollout(state, frame(0, [(10, 10)]), scenario(), {}, horizon_s=.2,
                         controllers=[controller(0, poll_phase=8)])
        self.assertEqual(sample(result, .1)["action_type"], "WAIT")
        self.assertEqual(sample(result, .2)["action_type"], "M")
        self.assertAlmostEqual(sample(result, .2)["row"], 10)

    def test_rejects_future_state_and_nonfinite_rates(self):
        state, poses = snapshot([robot(0, [])]), frame(0, [(3, 3)])
        later = deepcopy(poses)
        later["clock"] = 1
        with self.assertRaisesRegex(ValueError, "after its origin"):
            rollout(state, later, scenario(), {})
        with self.assertRaises(ValueError):
            rollout(state, poses, scenario(), {}, dt=float("nan"))
        with self.assertRaises(ValueError):
            rollout(state, poses, scenario(), {}, controllers=[])


if __name__ == "__main__":
    unittest.main()
