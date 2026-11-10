#include "trajectory_loop_functions.h"
#include <cstdlib>
#include <stdexcept>

/****************************************/
/****************************************/

/*
 * To reduce the number of waypoints stored in memory,
 * consider two robot positions distinct if they are
 * at least MIN_DISTANCE away from each other
 * This constant is expressed in meters
 */
static const Real MIN_DISTANCE = 0.05f;
/* Convenience constant to avoid calculating the square root in PostStep() */
static const Real MIN_DISTANCE_SQUARED = MIN_DISTANCE * MIN_DISTANCE;

/****************************************/
/****************************************/

void CTrajectoryLoopFunctions::Init(TConfigurationNode &t_tree) {
    // Set up logger
    auto console_logger = spdlog::default_logger()->clone("Argos Client");
    spdlog::set_default_logger(console_logger);

    /*
     * Go through all the robots in the environment
     * and create an entry in the waypoint map for each of them
     */
    /* Get the map of all foot-bots from the space */
    CSpace::TMapPerType &tFBMap = GetSpace().GetEntitiesByType("foot-bot");
    /* Go through them */
    for (CSpace::TMapPerType::iterator it = tFBMap.begin(); it != tFBMap.end();
         ++it) {
        /* Create a pointer to the current foot-bot */
        CFootBotEntity *pcFB = any_cast<CFootBotEntity *>(it->second);
        /* Create a waypoint vector */
        m_tWaypoints[pcFB] = vector<CVector3>();
        /* Add the initial position of the foot-bot */
        m_tWaypoints[pcFB].push_back(
            pcFB->GetEmbodiedEntity().GetOriginAnchor().Position);
    }

    TConfigurationNode &portParams = GetNode(t_tree, "port_number");
    GetNodeAttribute(portParams, "value", port_number);

    while (!is_port_open("127.0.0.1", port_number)) {
        spdlog::info("Waiting for server to open at port {}...", port_number);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    this->client = make_shared<rpc::client>("127.0.0.1", port_number);
    this->is_initialized = true;
    spdlog::info("Loop func: Connected to server at port {}", port_number);
    initSafety();
    const char* record = std::getenv("LSMART_RECORD_PLANS");
    record_prediction_snapshots_ = record && *record &&
        string(record) != "0" && string(record) != "false";
}

namespace {
safety::Pose poseOf(CEmbodiedEntity& body) {
    const auto& anchor = body.GetOriginAnchor();
    CRadians yaw, pitch, roll;
    anchor.Orientation.ToEulerAngles(yaw, pitch, roll);
    return {anchor.Position.GetX(), anchor.Position.GetY(), yaw.GetValue()};
}

void safetyEvent(const json& event) {
    // Loop hooks execute outside parallel controller updates.
    cout << event.dump() << endl;
}

double positiveSetting(const json& settings, const char* key, double fallback) {
    const double value = settings.value(key, fallback);
    if (!std::isfinite(value) || value <= 0)
        throw std::runtime_error(string("Invalid safety setting: ") + key);
    return value;
}
}

void CTrajectoryLoopFunctions::initSafety() {
    const char* scenario_path = std::getenv("LSMART_SCENARIO_FILE");
    if (!scenario_path || !*scenario_path) return;
    std::ifstream input(scenario_path);
    if (!input) throw std::runtime_error("Cannot read safety scenario file");
    json scenario; input >> scenario;
    const json defaults = scenario.value("safety", json::object());
    safety_enabled_ = defaults.value("enabled", false);
    if (!safety_enabled_) return;

    json configurations = json::array();
    for (auto& item : GetSpace().GetEntitiesByType("foot-bot")) {
        auto* entity = any_cast<CFootBotEntity*>(item.second);
        auto* controller = dynamic_cast<CFootBotDiffusion*>(
            &entity->GetControllableEntity().GetController());
        if (!controller) throw std::runtime_error("Safety requires footbot_diffusion controller");
        const auto id = std::stoi(entity->GetId());
        json settings = defaults;
        for (const auto& agent : scenario.at("agents")) {
            if (agent.at("id").get<int>() == id && agent.contains("safety")) {
                settings.update(agent.at("safety")); break;
            }
        }
        safety::Rectangle slow{positiveSetting(settings, "slow_front_m", 1.2),
                                positiveSetting(settings, "slow_back_m", 0.2),
                                positiveSetting(settings, "slow_half_width_m", 0.3)};
        safety::Rectangle stop{positiveSetting(settings, "stop_front_m", 0.3),
                                positiveSetting(settings, "stop_back_m", 0.15),
                                positiveSetting(settings, "stop_half_width_m", 0.18)};
        const double scale = positiveSetting(settings, "speed_scale", 0.2);
        const double radius = positiveSetting(settings, "robot_radius_m", 0.085);
        const bool inner_stop_enabled = settings.value("inner_stop_enabled", true);
        const string trigger = settings.value("robot_trigger", string("body_in_outer"));
        if (trigger != "body_in_outer" && trigger != "outer_overlap")
            throw std::runtime_error("Safety robot_trigger must be body_in_outer or outer_overlap");
        if (scale >= 1 || (inner_stop_enabled &&
            (stop.front > slow.front || stop.back > slow.back || stop.half_width > slow.half_width)))
            throw std::runtime_error("Safety requires 0 < speed_scale < 1 and nested rectangles");
        if (slow.front < radius || slow.back < radius || slow.half_width < radius)
            throw std::runtime_error("Outer safety rectangle must contain the configured robot disc");
        const bool enabled = settings.value("enabled", true);
        safety_robots_.push_back({entity, controller, slow, stop, scale, radius,
                                 enabled, inner_stop_enabled, trigger == "outer_overlap"});
        configurations.push_back({{"agent_id", id}, {"enabled", enabled},
            {"robot_trigger", trigger}, {"inner_stop_enabled", inner_stop_enabled},
            {"slow_front_m", slow.front}, {"slow_back_m", slow.back},
            {"slow_half_width_m", slow.half_width}, {"stop_front_m", stop.front},
            {"stop_back_m", stop.back}, {"stop_half_width_m", stop.half_width},
            {"speed_scale", scale}, {"robot_radius_m", radius}});
    }
    // Generated maps contain wall and shelf boxes. Their actual dimensions,
    // including the 0.8m shelf boxes, are used rather than occupied grid cells.
    for (auto& item : GetSpace().GetEntitiesByType("box"))
        safety_boxes_.push_back(any_cast<CBoxEntity*>(item.second));
    safetyEvent({{"type", "safety_config"}, {"enabled", true},
                 {"model", "ideal_local_geometry_rectangles"},
                 {"slow_applies_to", "move"}, {"stop_applies_to", "move_and_turn"},
                 {"boundary_contact_triggers", true},
                 {"obstacle_trigger", "obstacle_footprint_intersection"},
                 {"inner_robot_trigger", "body_in_inner"},
                 {"other_disabled_robot_fields", "geometry_remains_present"},
                 {"robots", configurations}});
}

void CTrajectoryLoopFunctions::updateSafety() {
    if (!safety_enabled_) return;
    vector<safety::Pose> poses;
    for (auto& robot : safety_robots_)
        poses.push_back(poseOf(robot.entity->GetEmbodiedEntity()));
    vector<safety::Box> boxes;
    for (auto* box : safety_boxes_)
        boxes.push_back({poseOf(box->GetEmbodiedEntity()),
                         box->GetSize().GetX() / 2, box->GetSize().GetY() / 2});
    for (size_t i = 0; i < safety_robots_.size(); ++i) {
        auto& robot = safety_robots_[i];
        int mode = 0;
        string blocker_id, blocker_kind, trigger_reason = "none";
        bool robot_outer = false, body_outer = false, obstacle_outer = false;
        const auto remember = [&](int candidate, const string& id, const char* kind, const char* reason) {
            if (candidate > mode) {
                mode = candidate; blocker_id = id; blocker_kind = kind; trigger_reason = reason;
            }
        };
        if (robot.enabled) {
            for (size_t j = 0; j < safety_robots_.size(); ++j) {
                if (i == j) continue;
                const safety::Point centre{poses[j].x, poses[j].y};
                const double radius = safety_robots_[j].radius;
                const bool body = safety::intersectsDisc(robot.slow, poses[i], centre, radius);
                const bool outer = robot.outer_overlap
                    ? safety::intersectsRectangles(robot.slow, poses[i], safety_robots_[j].slow, poses[j])
                    : body;
                body_outer = body_outer || body;
                robot_outer = robot_outer || outer;
                if (robot.inner_stop_enabled && safety::intersectsDisc(robot.stop, poses[i], centre, radius))
                    remember(2, safety_robots_[j].entity->GetId(), "robot", "body_in_inner");
                else if (outer)
                    remember(1, safety_robots_[j].entity->GetId(), "robot",
                             robot.outer_overlap ? "outer_polygon_overlap" : "body_in_outer");
            }
            for (size_t j = 0; j < boxes.size(); ++j) {
                // Cheap broad phase before oriented rectangle intersection.
                const double dx = boxes[j].pose.x - poses[i].x;
                const double dy = boxes[j].pose.y - poses[i].y;
                const double reach = std::hypot(std::max(robot.slow.front, robot.slow.back),
                                                robot.slow.half_width) +
                                     std::hypot(boxes[j].half_x, boxes[j].half_y);
                if (dx * dx + dy * dy > reach * reach) continue;
                const bool outer = safety::intersectsBox(robot.slow, poses[i], boxes[j]);
                obstacle_outer = obstacle_outer || outer;
                if (robot.inner_stop_enabled && safety::intersectsBox(robot.stop, poses[i], boxes[j]))
                    remember(2, safety_boxes_[j]->GetId(), "box", "obstacle_in_inner");
                else if (outer)
                    remember(1, safety_boxes_[j]->GetId(), "box", "obstacle_in_outer");
            }
        }
        const double scale = mode == 2 ? 0.0 : mode == 1 ? robot.scale : 1.0;
        robot.controller->setSafetyScale(scale);
        if (mode == 1) ++robot.slow_ticks;
        if (mode == 2) ++robot.stop_ticks;
        if (robot_outer) ++robot.robot_outer_trigger_ticks;
        if (obstacle_outer) ++robot.obstacle_outer_trigger_ticks;
        if (robot.outer_overlap && robot_outer && !body_outer && !obstacle_outer)
            ++robot.polygon_only_trigger_ticks;
        if (mode != robot.mode || blocker_id != robot.blocker_id || trigger_reason != robot.trigger_reason) {
            safetyEvent({{"type", "safety_state"}, {"clock", GetSpace().GetSimulationClock()},
                         {"agent_id", std::stoi(robot.entity->GetId())},
                         {"mode", mode == 2 ? "stop" : mode == 1 ? "slow" : "clear"},
                         {"scale", scale}, {"blocker_id", blocker_id},
                         {"blocker_kind", blocker_kind}, {"enabled", robot.enabled},
                         {"trigger_reason", trigger_reason}});
            robot.mode = mode;
            robot.blocker_id = blocker_id;
            robot.trigger_reason = trigger_reason;
        }
    }
}

void CTrajectoryLoopFunctions::reportSafety() {
    if (!safety_enabled_ || safety_reported_) return;
    json robots = json::array();
    for (const auto& robot : safety_robots_) {
        auto stats = robot.controller->safetyStats();
        stats["agent_id"] = std::stoi(robot.entity->GetId());
        stats["slow_region_ticks"] = robot.slow_ticks;
        stats["stop_region_ticks"] = robot.stop_ticks;
        stats["robot_outer_trigger_ticks"] = robot.robot_outer_trigger_ticks;
        stats["obstacle_outer_trigger_ticks"] = robot.obstacle_outer_trigger_ticks;
        stats["polygon_only_trigger_ticks"] = robot.polygon_only_trigger_ticks;
        robots.push_back(stats);
    }
    safetyEvent({{"type", "safety_summary"}, {"enabled", true},
                 {"clock", GetSpace().GetSimulationClock()}, {"robots", robots}});
    safety_reported_ = true;
}

/****************************************/
/****************************************/

void CTrajectoryLoopFunctions::Reset() {
    /*
     * Clear all the waypoint vectors
     */
    /* Get the map of all foot-bots from the space */
    CSpace::TMapPerType &tFBMap = GetSpace().GetEntitiesByType("foot-bot");
    /* Go through them */
    for (CSpace::TMapPerType::iterator it = tFBMap.begin(); it != tFBMap.end();
         ++it) {
        /* Create a pointer to the current foot-bot */
        CFootBotEntity *pcFB = any_cast<CFootBotEntity *>(it->second);
        /* Clear the waypoint vector */
        m_tWaypoints[pcFB].clear();
        /* Add the initial position of the foot-bot */
        m_tWaypoints[pcFB].push_back(
            pcFB->GetEmbodiedEntity().GetOriginAnchor().Position);
    }
}

void CTrajectoryLoopFunctions::addMobileVisualization() {
    /* Get the map of all foot-bots from the space */
    CSpace::TMapPerType &tFBMap = GetSpace().GetEntitiesByType("foot-bot");
    /* Go through them */
    task_stations.clear();
    for (CSpace::TMapPerType::iterator it = tFBMap.begin(); it != tFBMap.end();
         ++it) {
        /* Create a pointer to the current foot-bot */
        CFootBotEntity *pcFB = any_cast<CFootBotEntity *>(it->second);

        // Get the controller
        CFootBotDiffusion *pcController = dynamic_cast<CFootBotDiffusion *>(
            &(pcFB->GetControllableEntity().GetController()));

        if (pcController) {
            // Retrieve step count or any other return value
            auto tmp_station = pcController->getCurrStation();
            if (tmp_station.GetZ() != -100) {
                task_stations.push_back(tmp_station);
            }
        }
    }
}
/****************************************/

void CTrajectoryLoopFunctions::PreStep() {
    const auto routing_start = std::chrono::steady_clock::now();
    const char* routing_directory = std::getenv("LSMART_ROUTING_DIR");
    const char* routing_interval = std::getenv("LSMART_ROUTING_INTERVAL_TICKS");
    const int interval = routing_interval ? std::stoi(routing_interval) : 20;
    // ARGoS increments its clock before PreStep; the visualizer and server
    // still label this controller boundary with the preceding tick.
    const auto routing_clock = GetSpace().GetSimulationClock() - 1;
    const bool routing_boundary = routing_directory && *routing_directory && interval > 0 &&
        routing_clock % interval == 0;
    writeRoutingRequest();
    // Free simulation if necessary
    client->call("freeze_simulation_if_necessary");

    // Loop until the simulation is defrozen
    while (client->call("is_simulation_frozen").as<bool>()) {
        // spdlog::info("Simulation is frozen, waiting to defrost...");
        std::this_thread::sleep_for(std::chrono::milliseconds(routing_boundary ? 1 : 100));
    }
    if (routing_boundary) {
        const double wall_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - routing_start).count();
        safetyEvent({{"type", "routing_barrier_complete"},
                     {"clock", routing_clock}, {"wall_s", wall_s},
                     {"accounting", "client request capture through accepted plan; simulation paused"}});
    }
    updateSafety();
    recordPredictionSnapshot();
}

void CTrajectoryLoopFunctions::recordPredictionSnapshot() {
    const auto clock = GetSpace().GetSimulationClock();
    if (!record_prediction_snapshots_ || clock % 10 != 0) return;
    safetyEvent(predictionSnapshot());
}

void CTrajectoryLoopFunctions::writeRoutingRequest() {
    const char* directory = std::getenv("LSMART_ROUTING_DIR");
    if (!directory || !*directory) return;
    const char* interval_setting = std::getenv("LSMART_ROUTING_INTERVAL_TICKS");
    const int interval = interval_setting ? std::stoi(interval_setting) : 20;
    if (interval <= 0) throw std::runtime_error("Routing interval must be positive");
    const auto clock = GetSpace().GetSimulationClock() - 1;
    if (clock % interval != 0) return;
    json request = predictionSnapshot();
    request["clock"] = clock;
    // Physics has already advanced to the ARGoS pose clock before PreStep.
    // Keep observation metadata on that physical clock; only the handshake
    // uses the server clock, which increments in PostStep.
    request["observation_clock"] = GetSpace().GetSimulationClock();
    if (request["snapshot"]["server_clock"].get<unsigned long long>() != clock)
        throw std::runtime_error("Routing request/server clock mismatch");
    request["type"] = "routing_request";
    request["boundary"] = "pre_controller_before_commit_cut";
    const string path = string(directory) + "/request_" + std::to_string(clock) + ".json";
    const string temporary = path + ".tmp";
    {
        std::ofstream output(temporary);
        if (!output) throw std::runtime_error("Cannot write routing request");
        output << request.dump() << std::endl;
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        throw std::runtime_error("Cannot atomically publish routing request");
}

json CTrajectoryLoopFunctions::predictionSnapshot() {
    const auto clock = GetSpace().GetSimulationClock();
    json poses = json::array(), controllers = json::array();
    for (auto& item : GetSpace().GetEntitiesByType("foot-bot")) {
        auto* entity = any_cast<CFootBotEntity*>(item.second);
        const auto* controller = dynamic_cast<const CFootBotDiffusion*>(
            &entity->GetControllableEntity().GetController());
        if (!controller) throw std::runtime_error("Prediction snapshots require footbot_diffusion");
        const auto pose = poseOf(entity->GetEmbodiedEntity());
        poses.push_back({{"id", std::stoi(entity->GetId())},
                         {"x", pose.x}, {"y", pose.y}, {"rz", pose.yaw}});
        controllers.push_back(controller->predictionState(clock));
    }
    // No controller runs during this loop hook. The server serializes its ADG
    // read with plan acceptance under globalMutex, without dispatching actions.
    json snapshot = json::parse(client->call("prediction_action_snapshot").as<string>());
    snapshot["clock"] = clock;
    return {{"type", "prediction_snapshot"}, {"schema_version", 1},
        {"clock", clock}, {"boundary", "pre_controller_after_planner_wait"},
        {"pose_frame", {{"clock", clock}, {"agents", poses}}},
        {"snapshot", snapshot}, {"controllers", controllers},
        {"state_scope", "accepted_actions_controller_memory_and_exact_poses"},
        {"physics_state_complete", false}};
}

void CTrajectoryLoopFunctions::PostStep() {
    if (not is_initialized) {
        spdlog::error("CTrajectoryLoopFunctions::PostStep: LoopFunctions is "
                      "not initialized");
        exit(1);
    }
    addMobileVisualization();

    // Invoke the server to record per tick stats. We do it here because
    // PostStep function is invoked after each tick.
    client->call("record_stats_per_tick");

    client->call("update_sim_step");

    // Get simulation status, return true if the simulation shall end
    this->end_sim = client->call("sim_status").as<bool>();

    // TODO: Add a cmd to capture visualization frames
    // static int frame_id = 0;
    // std::stringstream filename_s;
    // filename_s << "frames/frame_" << std::setw(5) << std::setfill('0')
    //            << frame_id++ << ".png";
    // std::string filename = filename_s.str();
    // argos::CVisualization &cVisualization = GetSimulator().GetVisualization();
    // // Ensure the visualization is of the expected Qt OpenGL type
    // auto *pcQtRender = dynamic_cast<argos::CQTOpenGLRender *>(&cVisualization);
    // if (pcQtRender) {
    //     // Access the main window
    //     CQTOpenGLMainWindow &pcMainWindow = pcQtRender->GetMainWindow();

    //     // Capture the current window as a pixmap
    //     QPixmap pixmap = pcMainWindow.grab();

    //     // Save the pixmap to the specified file
    //     if (!pixmap.save(QString::fromStdString(filename))) {
    //         spdlog::error("Failed to save the visualization snapshot to {}",
    //                       filename);
    //     }
    // } else {
    //     spdlog::error("Qt OpenGL Visualization is not available.");
    // }
}

void CTrajectoryLoopFunctions::PostExperiment() {
    reportSafety();
    // We want to stop the ADG server here, instead of from the controller.
    spdlog::info("PostExperiment called, closing server at port {}",
                 port_number);
    client->async_call("close_server");
    spdlog::info("Closing argos client");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    exit(0);  // Exit the argos simulator after closing the server
}

// Additional custom logic to determine if the experiment is finished
bool CTrajectoryLoopFunctions::IsExperimentFinished() {
    if (this->end_sim) {
        spdlog::info("Simulation is either congested or time step exceeded, "
                     "stopping the simulation.");
    }

    return this->end_sim;
}

void CTrajectoryLoopFunctions::Destroy() {
    reportSafety();
    m_tWaypoints.clear();
    task_stations.clear();
}

/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CTrajectoryLoopFunctions, "trajectory_loop_functions")
