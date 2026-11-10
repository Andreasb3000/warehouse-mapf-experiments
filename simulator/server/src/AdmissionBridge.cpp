#include "AdmissionBridge.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>

namespace admission_bridge {
int timeoutSeconds(const char* setting) {
    if (!setting) return 120;
    const string value(setting);
    if (value.empty() || value.find_first_not_of("0123456789") != string::npos)
        throw std::runtime_error("Admission timeout must be an integer number of seconds");
    const auto seconds = std::stoll(value);
    if (seconds < 1 || seconds > 3600)
        throw std::runtime_error("Admission timeout must be between 1 and 3600 seconds");
    return static_cast<int>(seconds);
}

json makeRequest(const json& original, const json& accepted_snapshot,
                 int clock, const string& planner_source) {
    if (!original.is_object() || !original.contains("clock") ||
        !original["clock"].is_number_integer() || original["clock"] != clock ||
        !original.contains("observation_clock") ||
        !original["observation_clock"].is_number_integer() ||
        original["observation_clock"] != clock + 1 ||
        !original.contains("pose_frame") || !original["pose_frame"].is_object() ||
        original["pose_frame"].value("clock", -1) != clock + 1 ||
        !original.contains("controllers") || !original["controllers"].is_array() ||
        !original.contains("snapshot") || !original["snapshot"].is_object() ||
        original["snapshot"].value("server_clock", -1) != clock)
        throw std::runtime_error("Admission bridge received an incoherent original routing request");
    if (!accepted_snapshot.is_object() ||
        accepted_snapshot.value("server_clock", -1) != clock ||
        !accepted_snapshot.value("simulation_frozen", false) ||
        !accepted_snapshot.contains("plan_version") ||
        !accepted_snapshot["plan_version"].is_number_unsigned())
        throw std::runtime_error("Admission bridge requires a fresh accepted snapshot while frozen");
    if (!original["snapshot"].contains("plan_version") ||
        !original["snapshot"]["plan_version"].is_number_integer() ||
        original["snapshot"]["plan_version"] < 0 ||
        accepted_snapshot["plan_version"].get<unsigned long long>() !=
            original["snapshot"]["plan_version"].get<unsigned long long>() + 1)
        throw std::runtime_error("Admission snapshot is not the newly accepted plan version");
    for (const auto& controller : original["controllers"])
        if (!controller.is_object() || controller.value("clock", -1) != clock + 1)
            throw std::runtime_error("Admission controller observation clock mismatch");
    if (planner_source != "primary" && planner_source != "fallback")
        throw std::runtime_error("Invalid admission planner source");
    json request = original;
    request["type"] = "admission_request";
    request["boundary"] = "post_planner_before_dispatch";
    request["planner_source"] = planner_source;
    request["plan_version"] = accepted_snapshot["plan_version"];
    request["snapshot"] = accepted_snapshot;
    request["snapshot"]["clock"] = original["observation_clock"];
    return request;
}

EntryGate parseResponse(const json& response, int clock,
                        unsigned long long plan_version, int num_robots,
                        int rows, int cols, int max_until_clock) {
    if (!response.is_object() || !response.contains("clock") ||
        !response["clock"].is_number_integer() || response["clock"] != clock)
        throw std::runtime_error("Admission response clock mismatch");
    if (response.contains("plan_version") &&
        (!response["plan_version"].is_number_integer() ||
         response["plan_version"] != plan_version))
        throw std::runtime_error("Admission response plan version mismatch");
    for (const auto& item : response.items())
        if (item.key() != "clock" && item.key() != "plan_version" && item.key() != "entry_gate")
            throw std::runtime_error("Unknown admission response field");
    return response.contains("entry_gate") && !response["entry_gate"].is_null()
        ? EntryGate::parse(response["entry_gate"], clock, num_robots, rows, cols, max_until_clock)
        : EntryGate{};
}

EntryGate exchange(const string& routing_dir, const string& admission_dir,
                   int timeout_seconds, const json& accepted_snapshot,
                   int clock, const string& planner_source, int num_robots,
                   int rows, int cols, int max_until_clock) {
    namespace fs = std::filesystem;
    const auto start = std::chrono::steady_clock::now();
    const fs::path original_path = fs::path(routing_dir) / ("request_" + std::to_string(clock) + ".json");
    const fs::path request_path = fs::path(admission_dir) / original_path.filename();
    const fs::path response_path = fs::path(admission_dir) / ("response_" + std::to_string(clock) + ".json");
    if (fs::exists(response_path) || fs::exists(request_path))
        throw std::runtime_error("Stale admission files exist for this planning clock");
    std::ifstream original_file(original_path);
    if (!original_file) throw std::runtime_error("Cannot read original routing request for admission");
    json original; original_file >> original;
    const json request = makeRequest(original, accepted_snapshot, clock, planner_source);
    fs::create_directories(admission_dir);
    const auto temporary = request_path.string() + ".tmp";
    {
        std::ofstream output(temporary);
        if (!output) throw std::runtime_error("Cannot write admission request");
        output << request.dump() << std::endl;
        if (!output) throw std::runtime_error("Cannot finish admission request");
    }
    if (std::rename(temporary.c_str(), request_path.c_str()) != 0)
        throw std::runtime_error("Cannot atomically publish admission request");
    while (!fs::exists(response_path)) {
        if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(timeout_seconds))
            throw std::runtime_error("Timed out waiting for admission response at clock " + std::to_string(clock));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::ifstream response_file(response_path);
    json response; response_file >> response;
    const EntryGate gate = parseResponse(response, clock, accepted_snapshot.at("plan_version").get<unsigned long long>(),
                                          num_robots, rows, cols, max_until_clock);
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << json({{"type", "admission_barrier_complete"}, {"clock", clock},
        {"observation_clock", clock + 1}, {"plan_version", accepted_snapshot.at("plan_version")},
        {"planner_source", planner_source}, {"wall_s", wall_s},
        {"held_robot_ids", gate.held_robot_ids}, {"protect_clearance", gate.protect_clearance},
        {"accounting", "post-plan decision wall time; simulation frozen"}}).dump() << std::endl;
    return gate;
}
}
