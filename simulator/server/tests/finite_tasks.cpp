// Focused regression checks for finite task assignment. This executable links
// against the execution server objects (excluding server.cpp's main).
#include "task_assigners/DistinctOneGoalTaskAssigner.h"
#include "backup_planners/PIBT.h"
#include "ADG.h"

#include <cstdio>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void write_tasks(const string &path, const string &text) {
    std::ofstream stream(path);
    stream << text;
}

template <typename F> void rejects(F action, const char *message) {
    try {
        action();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error(message);
}

void check_action_snapshot() {
    ADG adg(2, 0, 1);
    string first_robot = "0", second_robot = "1";
    adg.createRobotIDToStartIndexMaps(first_robot, {3, 2});
    adg.createRobotIDToStartIndexMaps(second_robot, {2, 2});
    auto empty = adg.getActionSnapshot();
    require(empty["plan_version"] == 0 && empty["robots"].size() == 2 &&
                empty["robots"][0]["actions"].empty(),
            "An initial snapshot invented planned actions.");

    auto action = [](int robot, char type, Location start, Location goal,
                     double time, int task = -1) {
        Action value;
        value.robot_id = robot;
        value.type = type;
        value.start = start;
        value.goal = goal;
        value.time = time;
        value.orientation = 2;
        value.task_id = task;
        return value;
    };
    adg.addMAPFPlan({
        {action(0, 'M', {3, 2}, {3, 2.5}, 0),
         action(0, 'M', {3, 2.5}, {3, 3}, 0),
         action(0, 'S', {3, 2.5}, {3, 3}, 0, 77)},
        {action(1, 'M', {2, 2}, {3, 2}, 1)}});
    auto snapshot = adg.getActionSnapshot();
    require(snapshot == adg.getActionSnapshot() &&
                adg.enqueue_nodes_idx[0].empty() &&
                adg.enqueue_nodes_idx[1].empty() &&
                snapshot["plan_version"] == 1,
            "Reading a snapshot changed execution state or plan version.");
    const auto& first = snapshot["robots"][0];
    require(first["finished_node_id"] == -1 &&
                first["enqueued_node_id"] == -1 &&
                first["committed_node_id"] == -1 &&
                first["actions"][0]["start"] == json({2, 3}) &&
                first["actions"][0]["goal"] == json({2.5, 3}) &&
                first["actions"][2]["task_id"] == 77,
            "Snapshot lost coordinates, task identity, or initial boundaries.");
    require(snapshot["robots"][1]["actions"][0]["dependencies"] ==
                json::array({json::array({0, 0})}),
            "Snapshot did not preserve an unresolved cross-robot dependency.");
    require(adg.getPlan(1).empty() && adg.getPlan(0).size() == 3,
            "Snapshot changed action dispatch eligibility.");

    adg.updateFinishedNode(0, 0);
    snapshot = adg.getActionSnapshot();
    require(snapshot["robots"][0]["finished_node_id"] == 0 &&
                snapshot["robots"][0]["enqueued_node_id"] == 2 &&
                snapshot["robots"][0]["actions"].size() == 2 &&
                snapshot["robots"][0]["actions"][0]["node_id"] == 1 &&
                snapshot["robots"][1]["actions"][0]["dependencies"].empty(),
            "Snapshot retained a finished action or resolved dependency.");
    require(adg.getPlan(1).size() == 1,
            "Snapshot prevented dispatch after dependency completion.");
    adg.computeCommitCut();
    adg.addMAPFPlan({{action(0, 'M', {3, 3}, {4, 3}, 2)}, {}});
    snapshot = adg.getActionSnapshot();
    require(snapshot["plan_version"] == 2 &&
                snapshot["robots"][0]["committed_node_id"] == 2 &&
                snapshot["robots"][1]["committed_node_id"] == 0 &&
                snapshot["robots"][0]["actions"].size() == 3 &&
                snapshot["robots"][0]["actions"][2]["node_id"] == 3,
            "Snapshot omitted the committed prefix or mislabeled new actions.");
}

void check_backup_conversion(const SMARTGrid &grid) {
    boost::program_options::variables_map options;
    for (const auto &option : vector<pair<string, int>>{
             {"seed", 42}, {"plan_window_timestep", 4},
             {"sim_window_timestep", 4}, {"screen", 0}, {"num_robots", 3}}) {
        options.insert({option.first, boost::program_options::variable_value(
                                          boost::any(option.second), false)});
    }
    SIPP solver;
    PIBT fallback(grid, solver, nullptr, options);
    fallback.solution.resize(3);
    for (int t = 0; t < 4; ++t) {
        fallback.solution[0].emplace_back(4, t, 0);  // already parked
        fallback.solution[1].emplace_back(t == 0 ? 20 : 21, t, 0);
        fallback.solution[2].emplace_back(9, t, 0);  // no remaining goals
    }
    vector<vector<Task>> goals = {{Task(4)}, {Task(21)}, {}};
    goals[1][0].id = 42;
    auto converted = fallback.convert_path_to_smart(goals);
    require(converted.size() == 3 && converted[0].size() == 4,
            "Fallback conversion changed the path horizon.");
    for (int t = 0; t < 4; ++t) {
        require(std::get<3>(converted[0][t]) == -1 &&
                    std::get<3>(converted[2][t]) == -1,
                "Fallback conversion created a task for an idle robot.");
        require(std::get<3>(converted[1][t]) == (t == 1 ? 42 : -1),
                "Fallback conversion duplicated a completed finite task.");
    }
}
}  // namespace

int main() {
    const string path = "/tmp/lsmart-finite-tasks-test.txt";
    SMARTGrid grid;
    grid.rows = 5;
    grid.cols = 5;
    grid.consider_rotation = false;
    grid.types.assign(25, CellType::FREE);
    grid.types[24] = CellType::OBSTACLE;
    for (int i = 0; i < 24; ++i) grid.free_locations.push_back(i);
    grid.task_locations = grid.free_locations;

    try {
        check_action_snapshot();
        check_backup_conversion(grid);
        write_tasks(path, "2\n4,-1,0,0;9,-1,0,0\n20,-1,0,0\n");
        DistinctOneGoalTaskAssigner assigner(grid, nullptr, 0, 2, 42, path);
        vector<tuple<double, double, int>> starts = {{0, 0, 0}, {4, 3, 0}};
        assigner.updateStartsAndGoals(starts, {});
        auto first = assigner.getGoalLocations();
        require(first[0][0].location == 4 && first[1][0].location == 20,
                "Scripted goals were not assigned in robot order.");
        require(!first[0][0].is_parking && first[1][0].is_parking,
                "Finite assignment did not mark only the last task as terminal.");
        int first_id = first[0][0].id;
        int second_id = first[1][0].id;
        require(first_id >= 0 && second_id >= 0 && first_id != second_id,
                "Task IDs must be unique and nonnegative.");
        require(assigner.planMakesTaskProgress(0, {{0, 4, 0, first_id}}),
                "A completion at the end of committed motion was marked blocked.");
        require(!assigner.planMakesTaskProgress(0, {{0, 0, 0, first_id}}) &&
                    !assigner.planMakesTaskProgress(0, {{0, 4, 0, second_id}}) &&
                    !assigner.planMakesTaskProgress(0, {{0, 0, 0, -1}}) &&
                    !assigner.planMakesTaskProgress(0, {}),
                "A blocked or incorrectly marked path was counted as progress.");
        require(assigner.planMakesTaskProgress(0, {{0, 0, 0, -1}, {0, 1, 1, -1}}),
                "A moving route was incorrectly marked blocked.");
        assigner.updateStartsAndGoals(starts, {});
        require(assigner.getGoalLocations()[0][0].id == first_id,
                "Replanning changed an unfinished commitment.");

        starts[1] = {4, 0, 0};
        assigner.updateStartsAndGoals(starts, {second_id});
        auto partial = assigner.getGoalLocations();
        require(partial[1][0].location == 20 && partial[1][0].id == -1,
                "A finished robot did not stay at its last destination.");
        require(partial[1][0].is_parking && !partial[1][0].movable,
                "An acknowledged final task lost its parking constraint.");
        require(assigner.hasPendingTasks(0) && !assigner.hasPendingTasks(1) &&
                    !assigner.allTasksFinished(),
                "Partial completion state is incorrect.");

        starts[0] = {0, 4, 0};
        assigner.updateStartsAndGoals(starts, {first_id});
        auto next = assigner.getGoalLocations()[0][0];
        require(next.location == 9 && next.id > std::max(first_id, second_id),
                "Next scripted task was not assigned a fresh task ID.");
        require(next.is_parking && !next.movable,
                "Final-task parking was delayed until physical acknowledgement.");
        starts[0] = {1, 4, 0};
        assigner.updateStartsAndGoals(starts, {next.id});
        for (int i = 0; i < 4; ++i) assigner.updateStartsAndGoals(starts, {});
        auto parked = assigner.getGoalLocations();
        require(assigner.allTasksFinished() && parked[0][0].location == 9 &&
                    parked[0][0].id == -1 && parked[1][0].id == -1,
                "Exhausted workloads generated new tasks or moved parking goals.");

        std::remove(path.c_str());
        rejects([&] { DistinctOneGoalTaskAssigner a(grid, nullptr, 0, 2, 42, path); },
                "Missing scripted file was accepted.");
        for (const string &invalid : {
                 "1\n4,-1,0,0\n",                    // robot count
                 "2\n4,-1,0,0\n",                   // missing robot row
                 "2\nhello,-1,0,0\n20,-1,0,0\n",   // malformed number
                 "2\n24,-1,0,0\n20,-1,0,0\n",      // obstacle goal
                 "2\n4,-1,0,0\n4,-1,0,0\n",        // shared destination
                 "2\n4,-1,1,0\n20,-1,0,0\n"}) {    // unsupported wait
            write_tasks(path, invalid);
            rejects([&] { DistinctOneGoalTaskAssigner a(grid, nullptr, 0, 2, 42, path); },
                    "Invalid scripted workload was accepted.");
        }
        DistinctOneGoalTaskAssigner random(grid, nullptr, 0, 2, 42);
        require(!random.hasFiniteTasks() && !random.allTasksFinished(),
                "The random lifelong mode was incorrectly marked finite.");
        std::remove(path.c_str());
        cout << "Finite task regression checks passed.\n";
        return 0;
    } catch (const std::exception &error) {
        std::remove(path.c_str());
        cerr << error.what() << '\n';
        return 1;
    }
}
