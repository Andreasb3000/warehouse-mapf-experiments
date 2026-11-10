#include "task_assigners/BasicTaskAssigner.h"

BasicTaskAssigner::BasicTaskAssigner(
    const SMARTGrid &G, const shared_ptr<HeuristicTableBase> heuristic_table,
    int screen, int num_of_agents, int seed, string task_file)
    : G(G),
      heuristic_table(heuristic_table),
      screen(screen),
      num_of_agents(num_of_agents),
      seed(seed),
      gen(mt19937(seed)) {
    this->task_id = 0;

    // Load tasks, if applicable
    this->load_tasks(task_file);

    // Initialize aisle usage
    if (this->G.get_grid_type() == SMARTGridType::ONE_BOT_PER_AISLE) {
        // for (const auto &entry : this->G.get_aisle_entries()) {
        //     this->aisle_usage[entry] = 0;  // Initialize to 0
        // }
        // TODO: Add support for ONE_BOT_PER_AISLE grid type
        throw std::runtime_error("BasicTaskAssigner: ONE_BOT_PER_AISLE grid "
                                 "type is not supported.");
    }

    this->starts.resize(this->num_of_agents);
    this->goal_locations.resize(this->num_of_agents);

    // Initialize workstation/endpiont distribution
    // Uniform distribution
    auto make_uniform_dist = [](size_t n) {
        return discrete_distribution<int>(n, 0.0, 1.0,
                                          [](double) { return 1.0; });
    };

    this->endpoint_dist = make_uniform_dist(this->G.endpoints.size());
    this->workstation_dist = make_uniform_dist(this->G.workstations.size());
    this->task_location_dist = make_uniform_dist(this->G.task_locations.size());
    this->free_location_dist = make_uniform_dist(this->G.free_locations.size());
}

bool BasicTaskAssigner::load_tasks(string task_file) {
    // cout << "Loading tasks from file: " << task_file << endl;
    spdlog::info("Loading tasks from file: {}", task_file);
    this->tasks = vector<list<Task>>(this->num_of_agents);
    if (task_file.empty() || task_file == "") {
        spdlog::info("No task file provided, using random tasks.");
        this->random_task = true;
        return false;  // No agent file provided, return false
    }
    // An explicitly requested workload must never silently become random.
    this->tasks = read_task_vec(task_file, this->num_of_agents);
    for (const auto &agent_tasks : tasks) {
        for (const auto &task : agent_tasks) {
            if (task.location < 0 || task.location >= G.size() ||
                G.types[task.location] == CellType::OBSTACLE) {
                throw std::runtime_error("Task file contains a non-traversable goal: " +
                                         std::to_string(task.location));
            }
        }
    }

    // Loading task is successful, we shall use the loaded tasks onward.
    this->random_task = false;
    spdlog::info("Loaded {} tasks for {} agents.", this->tasks[0].size(),
                 this->num_of_agents);

    return true;
}

bool BasicTaskAssigner::hasPendingTasks(int agent_id) const {
    if (!hasFiniteTasks() || !tasks.at(agent_id).empty()) return true;
    for (const auto &goal : goal_locations.at(agent_id)) {
        if (goal.id >= 0) return true;
    }
    return false;
}

bool BasicTaskAssigner::allTasksFinished() const {
    if (!hasFiniteTasks()) return false;
    for (int i = 0; i < num_of_agents; ++i) {
        if (hasPendingTasks(i)) return false;
    }
    return true;
}

bool BasicTaskAssigner::planMakesTaskProgress(int agent_id,
                                           const vector<UserState> &path) const {
    if (path.empty()) return false;
    for (const auto &state : path) {
        if (std::get<0>(state) != std::get<0>(path.front()) ||
            std::get<1>(state) != std::get<1>(path.front())) return true;
        // Planning starts at the end of the committed action prefix. That
        // prefix may already reach the destination, while its completion
        // action has not yet been committed. PBS then returns a one-cell path
        // carrying the goal's task ID. The parser turns it into an S action;
        // it is productive task completion, not a blocked stationary route.
        if (std::get<3>(state) < 0) continue;
        for (const auto &goal : goal_locations.at(agent_id)) {
            if (goal.id == std::get<3>(state) &&
                G.getRowCoordinate(goal.location) == std::get<0>(state) &&
                G.getColCoordinate(goal.location) == std::get<1>(state)) return true;
        }
    }
    return false;
}

json BasicTaskAssigner::getMAPFInstanceJSON() const {
    return json{
        {"starts", this->starts},
        {"goals", this->goal_locations},
    };
}

void BasicTaskAssigner::print_mapf_instance(vector<State> starts_,
                                            vector<vector<Task>> goals_) const {
    for (int i = 0; i < starts_.size(); i++) {
        cout << "Agent " << i << ": ";
        int start_x = G.getRowCoordinate(starts_[i].location);
        int start_y = G.getColCoordinate(starts_[i].location);
        cout << "(" << start_x << ", " << start_y << ", "
             << starts_[i].orientation << ", t = " << starts_[i].timestep
             << ") => ";
        for (const auto &goal : goals_[i]) {
            int goal_x = G.getRowCoordinate(goal.location);
            int goal_y = G.getColCoordinate(goal.location);
            int wait_t = goal.task_wait_time;
            cout << "(" << goal_x << ", " << goal_y << ", "
                 << this->G.types[goal.location] << ", "
                 << "wait: " << wait_t << ") -> ";
        }
        cout << endl;
    }
}

int BasicTaskAssigner::sample_workstation() {
    int idx = this->workstation_dist(this->gen);
    return this->G.workstations[idx];
}

int BasicTaskAssigner::sample_endpiont() {
    int idx = this->endpoint_dist(this->gen);
    return this->G.endpoints[idx];
}

int BasicTaskAssigner::sample_task_location() {
    int idx = this->task_location_dist(this->gen);
    return this->G.task_locations[idx];
}

int BasicTaskAssigner::sample_free_location() {
    int idx = this->free_location_dist(this->gen);
    return this->G.free_locations[idx];
}

WindowedTaskAssigner::WindowedTaskAssigner(
    const SMARTGrid &G, const shared_ptr<HeuristicTableBase> heuristic_table,
    int screen, int simulation_window, int num_of_agents, int seed,
    string task_file)
    : BasicTaskAssigner(G, heuristic_table, screen, num_of_agents, seed),
      simulation_window(simulation_window) {
}

void WindowedTaskAssigner::updateStartsAndGoals(
    vector<tuple<double, double, int>> &start_locs,
    set<int> finished_tasks_id) {
    if (start_locs.size() != this->num_of_agents) {
        spdlog::error("WindowedTaskAssigner::updateStartsAndGoals: starts size "
                      "of {} does not match num_of_agents {}.",
                      start_locs.size(), this->num_of_agents);
    }

    // Remove finished goals
    if (screen > 0) {
        string new_finished_tasks = "New finished tasks: ";
        for (const auto &_task_id : finished_tasks_id) {
            new_finished_tasks += std::to_string(_task_id) + " ";
        }
        spdlog::info(new_finished_tasks);
    }

    if (!this->goal_locations.empty()) {
        // Remove finished tasks from goal locations
        for (int i = 0; i < this->num_of_agents; i++) {
            for (int j = 0; j < (int)goal_locations[i].size(); j++) {
                if (finished_tasks_id.find(goal_locations[i][j].id) !=
                    finished_tasks_id.end()) {
                    goal_locations[i][j].id = -1;  // reset goal id
                }
            }
        }

        // Remove goals that are already finished
        for (int i = 0; i < this->num_of_agents; i++) {
            goal_locations[i].erase(
                remove_if(goal_locations[i].begin(), goal_locations[i].end(),
                          [](const Task &t) { return t.id == -1; }),
                goal_locations[i].end());
        }
    }

    // Set new starts
    for (int i = 0; i < this->num_of_agents; i++) {
        // Obtain the starts
        int row = static_cast<int>(std::get<0>(start_locs[i]));
        int col = static_cast<int>(std::get<1>(start_locs[i]));
        int ori = std::get<2>(start_locs[i]);
        this->starts[i] = State(this->G.getCellId(row, col), 0,
                                this->G.ORI_SMART_TO_RHCR.at(ori));
    }

    // Initialize next_goal_type
    if (this->next_goal_type.empty()) {
        spdlog::info("Initializing next_goal_type.");
        for (int i = 0; i < this->num_of_agents; i++) {
            int start_loc = this->starts[i].location;
            // If start from workstation, next goal is endpoint
            if (G.types[start_loc] == CellType::WORKSTATION) {
                this->next_goal_type.push_back(CellType::ENDPOINT);
            }
            // If start from endpoint, next goal is workstation
            else if (G.types[start_loc] == CellType::ENDPOINT) {
                this->next_goal_type.push_back(CellType::WORKSTATION);
            }
            // Otherwise, randomize the first goal
            else {
                // Randomly choose a goal location and infer its type
                int tmp_goal_loc =
                    G.task_locations[rand() %
                                     static_cast<int>(G.task_locations.size())];
                if (G.types[tmp_goal_loc] == CellType::WORKSTATION) {
                    this->next_goal_type.push_back(CellType::WORKSTATION);
                } else if (G.types[tmp_goal_loc] == CellType::ENDPOINT) {
                    this->next_goal_type.push_back(CellType::ENDPOINT);
                } else if (G.types[tmp_goal_loc] == CellType::FREE) {
                    // Randomly choose free location. Usually this is because no
                    // workstation or endpionts are avaialble
                    this->next_goal_type.push_back(CellType::FREE);
                } else {
                    spdlog::error("WindowedTaskAssigner::updateStartsAndGoals: "
                                  "The tile type at start={} should not "
                                  "be {}",
                                  start_loc, G.types[start_loc]);
                    exit(-1);
                }
            }
        }
    }

    // Generate new goals
    for (int k = 0; k < this->num_of_agents; k++) {
        int curr = this->starts[k].location;  // current location
        Task goal;                            // The last goal location
        if (goal_locations[k].empty()) {
            goal = Task(curr, -1, 0, 0);
        } else {
            goal = goal_locations[k].back();
        }
        double min_timesteps = 0;
        int prev_loc = curr;
        for (const auto &g : goal_locations[k]) {
            min_timesteps += G.get_Manhattan_distance(g.location, prev_loc);
            prev_loc = g.location;
        }
        // double min_timesteps = G.get_Manhattan_distance((goal.location),
        // curr);
        while (min_timesteps <= this->simulation_window ||
               goal_locations[k].size() < 2)
        // The agent might finish its tasks during the next planning
        // horizon
        {
            // assign a new task
            Task next;
            if (G.types[goal.location] == CellType::ENDPOINT ||
                G.types[goal.location] == CellType::WORKSTATION ||
                G.types[goal.location] == CellType::FREE) {
                next = Task(this->gen_next_goal(k), -1, 0, 0);
                // Avoid consecutive duplicate goals
                while (next == goal) {
                    next = Task(this->gen_next_goal(k, true), -1, 0, 0);
                }
            } else {
                spdlog::error("WindowedTaskAssigner::updateStartsAndGoals: The "
                              "tile type at curr={} should not be {}",
                              curr, G.types[curr]);
                exit(-1);
            }
            next.id = this->task_id;
            this->task_id += 1;  // Increment the global task ID
            goal_locations[k].emplace_back(next);
            min_timesteps +=
                G.get_Manhattan_distance(next.location, goal.location);
            goal = next;
        }
    }

    // Log the current start and goal locations in the format of () -> () -> ...
    if (screen > 0) {
        spdlog::info("Current start and goal locations:");
        this->print_mapf_instance(this->starts, this->goal_locations);
        if (screen > 1) {
            // Check for consecutive duplicate goals for each agent
            for (int i = 0; i < this->num_of_agents; i++) {
                if (goal_locations[i].size() > 1) {
                    auto &goals = goal_locations[i];
                    for (size_t j = 1; j < goals.size(); j++) {
                        if (goals[j].location == goals[j - 1].location) {
                            spdlog::warn(
                                "Agent {} has consecutive duplicate goals at "
                                "location ({}, {})",
                                i, G.getRowCoordinate(goals[j].location),
                                G.getColCoordinate(goals[j].location));
                        }
                    }
                }
            }
        }
    }
}

int WindowedTaskAssigner::gen_next_goal(int agent_id, bool repeat_last_goal) {
    int next = -1;
    if (!this->random_task && !this->tasks[agent_id].empty()) {
        Task next = this->tasks[agent_id].front();
        this->tasks[agent_id].pop_front();
        return next.location;  // Only take the location of the task
    }

    // Alternate goal locations between workstations and endpoints
    if (this->next_goal_type[agent_id] == CellType::WORKSTATION) {
        if (repeat_last_goal) {
            next = this->sample_endpiont();
        } else {
            next = this->sample_workstation();
            this->next_goal_type[agent_id] = CellType::ENDPOINT;
        }
    } else if (this->next_goal_type[agent_id] == CellType::ENDPOINT) {
        if (repeat_last_goal) {
            next = this->sample_workstation();
        } else {
            next = this->sample_endpiont();
            this->next_goal_type[agent_id] = CellType::WORKSTATION;
        }
    } else if (this->next_goal_type[agent_id] == CellType::FREE) {
        next = this->sample_free_location();
        // Keep the next goal type as FREE
    } else {
        spdlog::error("WindowedTaskAssigner::gen_next_goal: next goal type is "
                      "not w or e, but {}",
                      this->next_goal_type[agent_id]);
        exit(1);
    }

    return next;
}
