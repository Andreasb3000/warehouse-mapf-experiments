#include "EntryGate.h"
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
bool finiteNumber(double value) {
    // Release uses -ffast-math, which can erase ordinary isfinite checks.
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
double number(const json& value) {
    if (!value.is_number()) throw std::runtime_error("Entry gate requires numeric coordinates");
    double result = value.get<double>();
    if (!finiteNumber(result)) throw std::runtime_error("Entry gate coordinates must be finite");
    return result;
}
int integer(const json& value) {
    if (!value.is_number_integer() || value < 0 || value > std::numeric_limits<int>::max())
        throw std::runtime_error("Entry gate requires a nonnegative integer");
    return value.get<int>();
}
}

EntryGate EntryGate::parse(const json& value, int clock, int num_robots,
                           int rows, int cols, int max_until_clock) {
    if (!value.is_object() || value.size() != (value.contains("protect_clearance") ? 5 : 4) || !value.contains("center") ||
        !value.contains("half_width") || !value.contains("held_robot_ids") ||
        !value.contains("until_clock")) throw std::runtime_error("Invalid entry_gate fields");
    if (!value["center"].is_array() || value["center"].size() != 2 ||
        !value["held_robot_ids"].is_array()) throw std::runtime_error("Invalid entry_gate arrays");
    EntryGate gate;
    gate.center = {number(value["center"][0]), number(value["center"][1])};
    gate.half_width = number(value["half_width"]);
    gate.until_clock = integer(value["until_clock"]);
    if (value.contains("protect_clearance")) {
        if (!value["protect_clearance"].is_boolean())
            throw std::runtime_error("Entry gate protect_clearance must be boolean");
        gate.protect_clearance = value["protect_clearance"].get<bool>();
    }
    if (gate.half_width <= 0 || gate.center.first - gate.half_width < 0 ||
        gate.center.second - gate.half_width < 0 ||
        gate.center.first + gate.half_width > rows - 1 ||
        gate.center.second + gate.half_width > cols - 1)
        throw std::runtime_error("Entry gate square is outside the map");
    if (gate.until_clock <= clock || gate.until_clock > max_until_clock)
        throw std::runtime_error("Entry gate expiry must be within the next planning interval");
    for (const auto& value_id : value["held_robot_ids"]) {
        int id = integer(value_id);
        if (id >= num_robots || !gate.held_robot_ids.insert(id).second)
            throw std::runtime_error("Invalid or duplicate entry gate robot ID");
    }
    return gate;
}

bool EntryGate::contains(Location p) const {
    return p.first >= center.first - half_width && p.first <= center.first + half_width &&
           p.second >= center.second - half_width && p.second <= center.second + half_width;
}

bool EntryGate::enters(Location start, Location goal) const {
    // Closed square: a start on its boundary is already inside and may leave.
    if (contains(start)) return false;
    double low = 0, high = 1;
    const double from[] = {start.first, start.second}, to[] = {goal.first, goal.second};
    const double c[] = {center.first, center.second};
    for (int axis = 0; axis < 2; ++axis) {
        double delta = to[axis] - from[axis];
        if (delta == 0) {
            if (from[axis] < c[axis] - half_width || from[axis] > c[axis] + half_width) return false;
        } else {
            double first = (c[axis] - half_width - from[axis]) / delta;
            double last = (c[axis] + half_width - from[axis]) / delta;
            if (first > last) std::swap(first, last);
            low = std::max(low, first);
            high = std::min(high, last);
            if (low > high) return false;
        }
    }
    return true;
}

bool EntryGate::blocks(int id, int clock, char type, Location start,
                       Location goal, bool enqueued, bool committed) const {
    return clock < until_clock && held_robot_ids.count(id) && type == 'M' &&
           !enqueued && !committed && approaches(start, goal);
}

bool EntryGate::approaches(Location start, Location goal) const {
    // Stage selected traffic upstream before its crossing enters the queue.
    // Interior, exit, and parallel/away motion remain available. Selection
    // and the upstream range belong to the host policy, not this actuator.
    if (contains(start)) return false;
    auto distance = [&](Location point) {
        return std::max(0.0, std::abs(point.first - center.first) - half_width) +
               std::max(0.0, std::abs(point.second - center.second) - half_width);
    };
    return enters(start, goal) || distance(goal) < distance(start);
}
