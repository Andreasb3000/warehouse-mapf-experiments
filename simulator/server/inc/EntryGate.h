#pragma once
#include "utils/common.h"

// Optional dispatch admission policy, in scenario [row,column] coordinates.
// This is neither an obstacle nor a planner constraint; it completes no action.
struct EntryGate {
    Location center{0, 0};
    double half_width = 0;
    int until_clock = -1;
    bool protect_clearance = true;
    set<int> held_robot_ids;
    static EntryGate parse(const json& value, int clock, int num_robots,
                           int rows, int cols, int max_until_clock);
    bool contains(Location point) const;
    bool enters(Location start, Location goal) const;
    bool approaches(Location start, Location goal) const;
    bool blocks(int robot_id, int clock, char type, Location start,
                Location goal, bool enqueued, bool committed) const;
};
