#pragma once
#include "common.h"
#include <cstdint>

namespace route_search {
struct Candidate { string name; vector<double> penalties; };
vector<Candidate> parseCandidates(const json& response, int clock, size_t cells);
uint32_t plannerSeed(int seed, int clock);
int timeoutSeconds();
int parseSelection(const json& response, const json& request);
json makeRequest(const json& original, int clock, uint32_t seed,
                 const json& instance, const json& candidates);
int exchange(const string& directory, const json& request, int timeout_seconds);
}
