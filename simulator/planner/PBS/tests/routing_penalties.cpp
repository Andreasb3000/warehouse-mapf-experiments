#include "Graph.h"
#include "Instance.h"
#include "SpaceTimeAStar.h"
#include <cstdio>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    const char* path = "/tmp/lsmart-routing-penalties-test.json";
    json map = {{"n_row", 5}, {"n_col", 7}, {"weight", false},
                {"layout", {"@@@@@@@", "@.....@", "@.....@", "@.....@", "@@@@@@@"}}};
    { std::ofstream output(path); output << map.dump(); }
    auto graph = std::make_shared<Graph>(path, 0);
    const int start = 15, goal = 19, centre = 17;
    Instance instance(graph, 0);
    instance.loadAgents({{"starts", {{{"location", start}}}},
                         {"goals", {{{{"location", goal}, {"id", 0}}}}}});
    std::vector<Path*> paths(1, nullptr);
    SpaceTimeAStar initial(instance, 0);
    const auto direct = initial.findOptimalPath({}, paths, 0);
    require(direct.size() == 5, "Zero-cost route should be four moves");
    require(direct.back().task_id == 0, "Goal loses its finite task identifier");
    std::vector<double> penalties(35, 0.0);
    penalties[centre] = 5.0;
    graph->setRoutingPenalties(penalties);
    SpaceTimeAStar rerouted(instance, 0);
    const auto detour = rerouted.findOptimalPath({}, paths, 0);
    require(detour.size() == 7, "Penalty should select six-move detour");
    for (const auto& step : detour)
        require(step.location != centre, "Weighted path crosses penalized cell");
    penalties[centre] = 0.25;
    graph->setRoutingPenalties(penalties);
    require(graph->getWeight(centre - 1, centre) == 1.25, "Fractional cost lost");
    require(graph->getWeight(centre, centre) == 1.0, "Waiting cost changed");
    AStarNode node(centre, 1.25, 2.5, nullptr, 1);
    require(node.g_val == 1.25 && node.h_val == 2.5, "Node truncates fractional costs");
    bool rejected = false;
    penalties[centre] = -1;
    try { graph->setRoutingPenalties(penalties); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Negative forecast costs must be rejected");
    std::remove(path);
    std::cout << "Routing penalty checks passed: zero path, detour, fractional costs, wait, invalid input." << std::endl;
}
