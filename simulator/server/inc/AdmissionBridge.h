#pragma once
#include "EntryGate.h"

// File-only decision barrier after a plan has been accepted, before any
// controller can dispatch from it. The caller holds the server mutex and
// keeps the simulator frozen for the entire exchange.
namespace admission_bridge {
int timeoutSeconds(const char* setting);
json makeRequest(const json& original, const json& accepted_snapshot,
                 int clock, const string& planner_source);
EntryGate parseResponse(const json& response, int clock,
                        unsigned long long plan_version, int num_robots,
                        int rows, int cols, int max_until_clock);
EntryGate exchange(const string& routing_dir, const string& admission_dir,
                   int timeout_seconds, const json& accepted_snapshot,
                   int clock, const string& planner_source, int num_robots,
                   int rows, int cols, int max_until_clock);
}
