/* Copyright (C) Jiaoyang Li
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Confidential
 * Written by Jiaoyang Li <jiaoyanl@usc.edu>, May 2020
 */

/*driver.cpp
 * Solve a MAPF instance on 2D grids.
 */
#include <rpc/client.h>
#include <rpc/rpc_error.h>
#include <cstdlib>
#include <chrono>
#include <thread>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

#include "Graph.h"
#include "Instance.h"
#include "PBS.h"
#include "common.h"
#include "RouteSearch.h"

/* Main function */
int main(int argc, char** argv) {
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    // clang-format off
	desc.add_options()
		("help", "produce help message")

		// params for the input instance and experiment settings
		("map,m", po::value<string>()->required(), "input file for map")
		("output,o", po::value<string>(), "output file for statistics")
		("outputPaths", po::value<string>(), "output file for paths")
		("agentNum,k", po::value<int>()->default_value(0), "number of agents")
		("cutoffTime,t", po::value<double>()->default_value(3), "cutoff time (seconds)")
		("screen,s", po::value<int>()->default_value(1), "screen option (0: none; 1: results; 2:all)")
		("stats", po::value<bool>()->default_value(false), "write to files some detailed statistics")
        ("portNum", po::value<int>()->default_value(8080), "port number for the server")
		("sipp", po::value<bool>()->default_value(1), "using SIPP as the low-level solver")
        ("seed", po::value<int>()->default_value(0), "random seed");
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 1;
    }

    po::notify(vm);

    int seed = vm["seed"].as<int>();
    srand(seed);  // Set the random seed for reproducibility

    // Set up logger
    auto console_logger = spdlog::default_logger()->clone("Planner");
    spdlog::set_default_logger(console_logger);

    ///////////////////////////////////////////////////////////////////////////
    // int prev_last_task_id = 0;  // last task id from the previous iteration
    // vector<Task> prev_goal_locs;
    set<int> finished_tasks_id;
    int screen = vm["screen"].as<int>();
    int num_agents = vm["agentNum"].as<int>();

    // Create a graph, heuristic will be computed only once in the graph
    auto graph = make_shared<Graph>(vm["map"].as<string>(), screen);

    // Stats
    int n_mapf_calls = 0;        // number of MAPF calls
    int n_rule_based_calls = 0;  // number of rule-based calls
    const char* plan_search_directory=std::getenv("LSMART_PLAN_SEARCH_DIR");
    const bool plan_search=plan_search_directory && *plan_search_directory;
    if(plan_search && (!std::getenv("LSMART_ROUTING_DIR") || std::getenv("LSMART_ADMISSION_DIR")))
        throw std::runtime_error("Route search requires routing bridge and excludes admission gates");
    const int search_timeout=plan_search ? route_search::timeoutSeconds() : 120;

    // We must have enough empty locations for agents to "wait in queue".
    // Since we do not consider window in this planner, it is not okay for
    // agents to have the goal. Therefore when agent j tries to go to a
    // location which is the goal of agent i, it must go to an empty space
    // first.
    if (graph->nEmptyLocations() < num_agents) {
        spdlog::error("Not enough empty locations for agents to wait in queue. "
                      "Please increase the number of empty locations.");
        exit(-1);
    }

    // We assume the server is already running at this point.
    rpc::client client("127.0.0.1", vm["portNum"].as<int>());
    // client.set_timeout(5000);  // in ms

    // Wait for the server to initialize
    int trial = 0;
    while (!client.call("is_initialized").as<bool>()) {
        if (screen > 0) {
            // printf("%d Waiting for server to initialize...\n", trial);
            trial++;
        }
    }

    while (true) {
        // Get the current simulation tick
        bool invoke_planner = client.call("invoke_planner").as<bool>();

        // Skip planning until the simulation step is a multiple of the
        // simulation window
        if (!invoke_planner) {
            const char* bridge = std::getenv("LSMART_ROUTING_DIR");
            if (bridge && *bridge)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        string result_message = client.call("get_location").as<string>();

        auto result_json = json::parse(result_message);
        if (!result_json["initialized"].get<bool>()) {
            printf("Planner not initialized! Retrying\n");
            sleep(1);
            continue;
        }

        // Obtain the MAPF instance
        if (!result_json.contains("mapf_instance")) {
            spdlog::error("PBS Driver: mapf_instance not found in the JSON "
                          "from server. Exit...");
            exit(1);
        }

        if (!result_json["mapf_instance"].contains("starts") ||
            !result_json["mapf_instance"].contains("goals")) {
            spdlog::error("PBS Driver: starts or goals not found in the "
                          "mapf_instance from server. Exit...");
            exit(1);
        }
        json mapf_instance = result_json["mapf_instance"];

        const char* routing_directory = std::getenv("LSMART_ROUTING_DIR");
        const bool routing_bridge = routing_directory && *routing_directory;
        int routing_clock = -1;
        int routing_nonzero = 0;
        double routing_max_penalty = 0.0;
        json routing_response;
        if (routing_bridge) {
            routing_clock = result_json.at("routing_clock").get<int>();
            std::ifstream input(string(routing_directory) + "/response_" +
                                std::to_string(routing_clock) + ".json");
            json response; input >> response;
            routing_response=response;
            if (response.at("clock").get<int>() != routing_clock)
                throw std::runtime_error("Routing response clock mismatch");
            vector<double> penalties = response.at("cell_penalties").get<vector<double>>();
            graph->setRoutingPenalties(penalties);
            for (double value : penalties) {
                routing_nonzero += value > 0.0;
                routing_max_penalty = std::max(routing_max_penalty, value);
            }
        }

        if(plan_search) {
            const auto definitions=route_search::parseCandidates(routing_response,routing_clock,graph->map_size);
            const auto candidate_seed=route_search::plannerSeed(seed,routing_clock);
            json candidates=json::array();
            ++n_mapf_calls;
            double total_wall=0;
            bool base_success=true;
            for(size_t i=0;i<definitions.size();++i) {
                const auto& definition=definitions[i];
                int nonzero=0; double maximum=0;
                for(double x:definition.penalties) {nonzero+=x>0; maximum=std::max(maximum,x);}
                json item={{"index",i},{"name",definition.name},{"evaluated",false},{"success",false},
                    {"planner_wall_s",0.0},{"nonzero_penalty_cells",nonzero},{"max_cell_penalty",maximum},
                    {"plan",nullptr},{"preview_snapshot",nullptr}};
                if(i==0 || base_success) {
                    graph->setRoutingPenalties(definition.penalties);
                    // Every candidate receives the same immutable starts/goals and RNG seed.
                    // Extra solves cannot perturb the next boundary's random stream.
                    srand(candidate_seed);
                    Instance candidate_instance(graph,screen); candidate_instance.loadAgents(mapf_instance);
                    PBS candidate_planner(candidate_instance,false,screen);
                    const auto start=std::chrono::steady_clock::now();
                    const bool ok=candidate_planner.solve(vm["cutoffTime"].as<double>());
                    const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                    const auto paths=candidate_planner.getPaths();
                    candidate_planner.clearSearchEngines(); candidate_planner.clear();
                    if(i==0) {base_success=ok; if(!ok) ++n_rule_based_calls;}
                    json stats={{"n_mapf_calls",n_mapf_calls},{"n_rule_based_calls",n_rule_based_calls}};
                    item["evaluated"]=true; item["success"]=ok; item["planner_wall_s"]=wall;
                    item["plan"]={{"success",ok},{"plan",paths},{"congested",congested(paths)},{"stats",stats.dump()}};
                    if(ok) item["preview_snapshot"]=json::parse(client.call("preview_plan",
                        json({{"clock",routing_clock},{"plan",paths}}).dump()).as<string>());
                    total_wall+=wall;
                } else item["skip_reason"]="base_planner_failed";
                cout<<json({{"type","route_candidate"},{"clock",routing_clock},{"index",i},
                    {"name",definition.name},{"evaluated",item["evaluated"]},{"success",item["success"]},
                    {"planner_wall_s",item["planner_wall_s"]},{"planner_seed",candidate_seed},
                    {"nonzero_penalty_cells",nonzero},{"max_cell_penalty",maximum}}).dump()<<endl;
                candidates.push_back(item);
            }
            std::ifstream original_input(string(routing_directory)+"/request_"+std::to_string(routing_clock)+".json");
            json original; original_input>>original;
            const auto request=route_search::makeRequest(original,routing_clock,candidate_seed,mapf_instance,candidates);
            const int selected=route_search::exchange(plan_search_directory,request,search_timeout);
            const auto& winner=candidates.at(selected);
            cout<<json({{"type","route_selection"},{"clock",routing_clock},{"selected_index",selected},
                {"selected_name",winner["name"]},{"planner_seed",candidate_seed},{"candidate_count",candidates.size()}}).dump()<<endl;
            json accepted=winner.at("plan");
            accepted["route_search"]={{"clock",routing_clock},{"selected_index",selected},{"selected_name",winner["name"]},
                {"planner_seed",candidate_seed},{"expected_preview_snapshot",winner["preview_snapshot"]}};
            client.call("add_plan",accepted.dump());
            cout<<json({{"type","routing_plan"},{"clock",routing_clock},{"planner_wall_s",total_wall},
                {"selected_planner_wall_s",winner["planner_wall_s"]},{"success",winner["success"]},
                {"nonzero_penalty_cells",winner["nonzero_penalty_cells"]},{"max_cell_penalty",winner["max_cell_penalty"]},
                {"selected_index",selected},{"candidate_count",candidates.size()},{"planner_seed",candidate_seed},
                {"low_level_solver","space_time_astar"},{"commitment_policy","native_ADG_commit_cut"}}).dump()<<endl;
            continue;
        }

        Instance instance(graph, screen);
        // instance.loadAgents(commit_cut, new_finished_tasks_id);
        instance.loadAgents(mapf_instance);
        // SIPP's time-based dominance does not optimize spatial penalties.
        // Every source, including zero, uses the same space-time A* here.
        PBS pbs(instance, routing_bridge ? false : vm["sipp"].as<bool>(), screen);
        // run
        double runtime = 0;
        bool success = false;
        double runtime_limit = vm["cutoffTime"].as<double>();
        int fail_count = 0;
        n_mapf_calls += 1;
        const auto routing_start = std::chrono::steady_clock::now();
        success = pbs.solve(runtime_limit);
        const double routing_wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - routing_start).count();
        vector<vector<tuple<int, int, double, int>>> new_mapf_plan =
            pbs.getPaths();
        if (!success) {
            n_rule_based_calls += 1;
        }

        pbs.clearSearchEngines();
        pbs.clear();

        // Send new plan
        json stats = {{"n_mapf_calls", n_mapf_calls},
                      {"n_rule_based_calls", n_rule_based_calls}};
        json new_plan_json = {
            {"success", success},
            {"plan", new_mapf_plan},
            {"congested", congested(new_mapf_plan)},
            {"stats", stats.dump()},
        };
        client.call("add_plan", new_plan_json.dump());
        if (routing_bridge) {
            cout << json({{"type", "routing_plan"}, {"clock", routing_clock},
                          {"planner_wall_s", routing_wall}, {"success", success},
                          {"nonzero_penalty_cells", routing_nonzero},
                          {"max_cell_penalty", routing_max_penalty},
                          {"low_level_solver", "space_time_astar"},
                          {"commitment_policy", "native_ADG_commit_cut"}}).dump() << endl;
        }
    }

    cout << "Planner finished!" << endl;
    return 0;
}
