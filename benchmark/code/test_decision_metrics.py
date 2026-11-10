"""Tests of utility criteria that can reverse a research conclusion if wrong."""
import unittest

from decision_metrics import DEFAULT_PROTOCOL, compare_candidate, paired_cluster_gain, task_completions, charge_query_time, trajectory_metrics


class TaskOutcomeTest(unittest.TestCase):
    def task(self, robot=0):
        return {"id": f"task-{robot}", "robot_id": robot, "release_at_s": 0,
                "stages": [{"operation": "pickup", "location": [1, 2]},
                           {"operation": "dropoff", "location": [8, 9]}]}

    def event(self, clock, location, robot=0):
        return {"type": "goal_reached", "clock": clock, "agent": robot,
                "row": location[0], "col": location[1]}

    def test_unfinished_tasks_remain_in_primary_denominator(self):
        tasks = [self.task(0), self.task(1)]
        events = [self.event(10, [1, 2]), self.event(500, [8, 9])]
        rows = task_completions(events, tasks, 10, 180)
        self.assertEqual([r["restricted_flow_s"] for r in rows], [50, 180])
        self.assertEqual(sum(r["restricted_flow_s"] for r in rows) / len(rows), 115)

    def test_routing_waypoints_duplicate_pickups_and_unordered_dropoff_do_not_finish_task(self):
        events = [self.event(1, [8, 9]), self.event(10, [1, 2]), self.event(11, [1, 2]),
                  self.event(100, [5, 5]), self.event(250, [8, 9]), self.event(251, [8, 9])]
        row = task_completions(events, [self.task()], 10, 180)[0]
        self.assertEqual(row["pickup_completed_s"], 1)
        self.assertEqual(row["delivery_completed_s"], 25)

    def test_compute_pause_is_charged_through_same_tick_pre_controller_barrier(self):
        rows = task_completions([self.event(10, [1, 2]), self.event(250, [8, 9])], [self.task()], 10, 180)
        charge_query_time(rows, [{"origin_s": 12, "query_s": .3}, {"origin_s": 25, "query_s": .4},
                                 {"origin_s": 30, "query_s": .6}], 180)
        self.assertAlmostEqual(rows[0]["charged_restricted_flow_s"], 25.7)

    def test_unfinished_task_is_charged_all_queries_within_horizon(self):
        rows = task_completions([], [self.task()], 10, 180)
        charge_query_time(rows, [{"origin_s": 12, "query_s": 1}, {"origin_s": 178, "query_s": 2}], 180)
        self.assertEqual(rows[0]["charged_restricted_flow_s"], 183)

    def test_duplicate_query_timing_records_are_rejected(self):
        rows = task_completions([], [self.task()], 10, 180)
        with self.assertRaises(ValueError):
            charge_query_time(rows, [{"origin_s": 12, "query_s": 1}, {"origin_s": 12, "query_s": 2}], 180)

    def test_travel_includes_completion_pose_one_physics_tick_after_server_ack(self):
        tasks=[self.task()]
        rows=task_completions([self.event(0,[1,2]),self.event(1,[8,9])],tasks,10,1)
        frames=[{"clock":k,"agents":[{"id":0,"x":-float(k),"y":-3.,"rz":0.}]} for k in range(4)]
        result=trajectory_metrics(frames,{"agents":[{"id":0}],"zones":[]},tasks,rows,.1,DEFAULT_PROTOCOL)
        self.assertEqual(result["travel_distance_m"],2.)


TEST_PROTOCOL = {**DEFAULT_PROTOCOL, "primary_workload_seeds": list(range(12)), "control_workload_seeds": list(range(12))}


def conditions():
    result = []
    for seed in range(12):
        for trigger in DEFAULT_PROTOCOL["safety_triggers"]:
            for regime in ("cross_zone", "within_zone"):
                for method in ("zero", "joint"):
                    result.append({"workload_seed": seed, "trigger": trigger, "demand_regime": regime,
                                   "method": method, "horizon_s": 180, "tasks_total": 27,
                                   "deliveries_unfinished": 0,
                                   "restricted_mean_flow_s": 90 if method == "joint" and regime == "cross_zone" else 100,
                                   "p95_stationary_wait_s": 10,
                                   "approaches": {"a": {"severe_queue_seconds": 5}, "b": {"severe_queue_seconds": 0}}})
    return result


class AcceptanceTest(unittest.TestCase):
    def test_constant_paired_gain_is_exact_and_workloads_not_conditions_are_units(self):
        score = paired_cluster_gain({i: 100 for i in range(12)}, {i: 90 for i in range(12)})
        self.assertEqual(score["independent_workloads"], 12)
        self.assertAlmostEqual(score["relative_improvement"], .1)
        self.assertAlmostEqual(score["ci_low"], .1)
        self.assertAlmostEqual(score["ci_high"], .1)

    def test_uniform_safe_gain_passes(self):
        score = compare_candidate(conditions(), "joint", "zero", TEST_PROTOCOL)
        self.assertEqual(score["status"], "pass")
        self.assertEqual(score["primary"]["independent_workloads"], 12)

    def test_moving_a_queue_to_another_approach_fails_even_with_lower_fleet_total(self):
        rows = conditions()
        row = next(r for r in rows if r["method"] == "joint")
        row["approaches"]["a"]["severe_queue_seconds"] = 0
        row["approaches"]["b"]["severe_queue_seconds"] = 3
        score = compare_candidate(rows, "joint", "zero", TEST_PROTOCOL)
        self.assertEqual(score["status"], "not_passed")
        self.assertEqual(score["guardrail_violations"][0]["guardrail"], "severe_queue")

    def test_one_extra_unfinished_task_fails_despite_faster_mean(self):
        rows = conditions()
        next(r for r in rows if r["method"] == "joint")["deliveries_unfinished"] = 1
        score = compare_candidate(rows, "joint", "zero", TEST_PROTOCOL)
        self.assertEqual(score["status"], "not_passed")
        self.assertEqual(score["guardrail_violations"][0]["guardrail"], "extra_unfinished")

    def test_missing_control_cannot_pass(self):
        rows = [r for r in conditions() if r["demand_regime"] != "within_zone"]
        self.assertEqual(compare_candidate(rows, "joint", "zero", TEST_PROTOCOL)["status"], "incomplete")

    def test_small_workload_count_cannot_pass_on_many_replicated_conditions(self):
        rows = [r for r in conditions() if r["workload_seed"] < 3]
        score = compare_candidate(rows, "joint", "zero", {**TEST_PROTOCOL, "primary_workload_seeds": list(range(3)), "control_workload_seeds": list(range(3))})
        self.assertEqual(score["status"], "not_passed")
        self.assertFalse(score["primary"]["enough_workloads"])

    def test_false_alarms_in_local_control_fail(self):
        rows = conditions()
        for row in rows:
            if row["method"] == "joint" and row["demand_regime"] == "within_zone":
                row["restricted_mean_flow_s"] = 103
        score = compare_candidate(rows, "joint", "zero", TEST_PROTOCOL)
        self.assertEqual(score["status"], "not_passed")
        self.assertFalse(score["control"]["body_in_outer"]["pass"])

    def test_missing_paired_method_is_incomplete(self):
        rows = conditions()[:-1]
        self.assertEqual(compare_candidate(rows, "joint", "zero", TEST_PROTOCOL)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
