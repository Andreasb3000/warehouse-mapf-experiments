// Parking and occupancy regressions for both fallback planners. These exercise
// real graph neighbors/heuristics and check every state through the horizon.
#include "backup_planners/PIBT.h"
#include "backup_planners/GuidedPIBT.h"
#include "heuristics/BasicHeuristicTable.h"
#include "utils/SMARTGraph.h"

namespace {
void require(bool ok, const string& why) {
    if (!ok) throw std::runtime_error(why);
}

boost::program_options::variables_map options(int count, int seed, int horizon) {
    boost::program_options::variables_map vm;
    for (const auto& [key, value] : vector<pair<string,int>>{
             {"seed",seed},{"plan_window_timestep",horizon},
             {"sim_window_timestep",horizon},{"screen",0},{"num_robots",count}})
        vm.insert({key,boost::program_options::variable_value(boost::any(value),false)});
    return vm;
}

SMARTGrid grid_for(const vector<string>& layout) {
    SMARTGrid grid;
    grid.consider_rotation=false;
    grid.hold_endpoints=false;
    grid.useDummyPaths=false;
    grid.screen=0;
    require(grid.load_map_from_jsonstr(json({{"n_row",layout.size()},
        {"n_col",layout.front().size()},{"weight",false},{"n_endpoint",0},
        {"n_agent_loc",0},{"maxtime",100},{"name","parking_regression"},
        {"layout",layout}}).dump(),1,1), "Unable to load regression grid");
    return grid;
}

Task goal(int location, bool parking, int id=-1, int wait=0) {
    return Task(location,-1,wait,0,-1,-1,parking,false,id);
}

void valid(const BasicGraph& grid, const vector<State>& starts,
           const vector<Path>& paths, int horizon) {
    require(paths.size()==starts.size(),"Planner changed robot count");
    for (size_t a=0;a<paths.size();++a) {
        require(paths[a].size()==static_cast<size_t>(horizon+1),"Wrong path horizon");
        require(paths[a][0].location==starts[a].location,"Planner changed a committed start");
        for (int t=0;t<=horizon;++t) {
            const auto& s=paths[a][t];
            require(s.location>=0 && s.location<grid.size() &&
                    grid.types[s.location]!=CellType::OBSTACLE,"Invalid path cell");
            require(s.timestep==t,"Invalid path timestamp");
            if (t>0) require(grid.get_weight(paths[a][t-1].location,s.location)<WEIGHT_MAX,
                             "Invalid movement or wait");
            for (size_t b=0;b<a;++b) {
                require(s.location!=paths[b][t].location,"Vertex collision in fallback path");
                if (t>0) require(!(paths[a][t-1].location==paths[b][t].location &&
                    paths[b][t-1].location==s.location),"Edge swap in fallback path");
            }
        }
    }
}

void stays(const Path& path, int location, int from=0) {
    for (size_t t=from;t<path.size();++t)
        require(path[t].location==location,"A parked robot left its terminal destination");
}

int arrival(const Path& path, int location) {
    for (size_t t=0;t<path.size();++t) if(path[t].location==location)return t;
    throw std::runtime_error("Regression robot failed to reach its unobstructed goal");
}

vector<Path> run(FailPolicy& planner,const BasicGraph& grid,
                 const vector<State>& starts,const vector<vector<Task>>& goals,int horizon) {
    vector<Path> guides;
    for(const auto& goals_i:goals)guides.push_back({State(goals_i.back().location,0)});
    require(planner.run(starts,goals,guides),"Fallback rejected regression instance");
    valid(grid,starts,planner.solution,horizon);
    return planner.solution;
}

template<class Planner> void checks(const SMARTGrid& grid, int seed) {
    const int horizon=12;
    auto heuristic=std::make_shared<BasicHeuristicTable>(grid,grid.free_locations,seed);
    SIPP single;
    Planner one(grid,single,heuristic,options(1,seed,horizon));

    // A zero-service goal must not create a phantom reservation that rejects
    // the robot's own WAIT, even for an ordinary unmarked lifelong goal.
    for(bool parking:{false,true}) {
        auto paths=run(one,grid,{State(7,0)},{{goal(7,parking)}},horizon);
        stays(paths[0],7);
    }

    // A pending final task is already terminal before its acknowledgement.
    vector<vector<Task>> final_goals={{goal(9,true,42)}};
    auto paths=run(one,grid,{State(6,0)},final_goals,horizon);
    stays(paths[0],9,arrival(paths[0],9));
    auto converted=one.convert_path_to_smart(final_goals);
    int markers=0;
    bool serviced=false;
    for(const auto& state:converted[0]) {
        if(std::get<3>(state)==42){++markers;serviced=true;}
        if(serviced)require(grid.getCellId(std::get<0>(state),std::get<1>(state))==9,
                            "Exported fallback departs after final service");
    }
    require(markers==1,"Final task completion marker duplicated or missing");

    // Waiting/turning already committed at the cut remains intact.
    paths=run(one,grid,{State(6,3,-1,true)},final_goals,horizon);
    for(int t=0;t<=3;++t)require(paths[0][t].location==6,"Committed wait was shortened");
    stays(paths[0],9,arrival(paths[0],9));
    paths=run(one,grid,{State(9,3,-1,true)},final_goals,horizon);
    stays(paths[0],9);
    paths=run(one,grid,{State(9,0)},{{goal(9,true,42,3)}},horizon);
    stays(paths[0],9);

    // Intermediate service is not terminal. The next task must still run.
    paths=run(one,grid,{State(7,0)},{{goal(7,false,41),goal(9,true,42)}},horizon);
    require(arrival(paths[0],9)>0,"Intermediate task did not allow continued movement");
    stays(paths[0],9,arrival(paths[0],9));

    // Reusing planner storage must copy a changed parking flag correctly.
    paths=run(one,grid,{State(7,0)},{{goal(9,false,43)}},horizon);
    require(arrival(paths[0],9)>0,"Reused planner retained an old parked state");

    // Adjacent active traffic cannot push a parked robot out of its bay.
    Planner two(grid,single,heuristic,options(2,seed,horizon));
    vector<State> starts={State(7,0),State(6,0)};
    paths=run(two,grid,starts,{{goal(7,true)},{goal(8,true,44)}},horizon);
    stays(paths[0],7);
    // PIBT is greedy and need not discover a detour that initially increases
    // distance. Its guarantee here is collision-free parking, not completion.
    paths=run(two,grid,starts,{{goal(7,true)},{goal(11,true,44)}},horizon);
    stays(paths[0],7);
    require(arrival(paths[1],11)>0,"Parking prevented unrelated active movement");

    // A robot still waiting on an earlier committed action reserves its cell
    // every timestep, including when it appears after the moving robot by ID.
    for(bool reverse:{false,true}) {
        vector<State> delayed={State(6,0),State(7,3,-1,true)};
        vector<vector<Task>> goals={{goal(8,true,45)},{goal(9,true,46)}};
        if(reverse){std::reverse(delayed.begin(),delayed.end());std::reverse(goals.begin(),goals.end());}
        paths=run(two,grid,delayed,goals,horizon);
        int waiting=reverse?0:1;
        for(int t=0;t<=3;++t)require(paths[waiting][t].location==7,"A delayed start was moved early");
    }
}

template<class Planner> void corridor_checks(int seed) {
    auto grid=grid_for({"...."});
    auto heuristic=std::make_shared<BasicHeuristicTable>(grid,grid.free_locations,seed);
    SIPP single;
    Planner planner(grid,single,heuristic,options(2,seed,8));
    vector<State> starts={State(0,0),State(1,0)};
    auto paths=run(planner,grid,starts,{{goal(3,false,10)},{goal(1,true)}},8);
    stays(paths[1],1);
    stays(paths[0],0);  // no bypass exists; do not solve by evicting parking

    // Unmarked lifelong idle goals retain the pre-existing yielding behavior.
    paths=run(planner,grid,starts,{{goal(3,false,10)},{goal(1,false)}},8);
    require(std::any_of(paths[1].begin(),paths[1].end(),[](const State& s){return s.location!=1;}),
            "Parking fix froze a nonterminal lifelong robot");
}

void task_copy_checks() {
    Task terminal=goal(7,true,1),ordinary=goal(8,false,2),assigned;
    assigned=terminal;
    require(assigned==terminal,"Task assignment dropped terminal parking metadata");
    assigned=ordinary;
    require(assigned==ordinary,"Task assignment retained obsolete parking metadata");
    vector<Task> reused={ordinary};
    reused=vector<Task>{terminal};
    require(reused[0].is_parking,"Reused vector storage dropped parking metadata");
    reused=vector<Task>{ordinary};
    require(!reused[0].is_parking,"Reused vector storage retained parking metadata");
}

// Replay an archived finite experiment's fallback request. Match each current
// goal to tasks.json's final stage, preserving nonterminal pickup goals. This
// checks discrete fallback output, not physical execution or committed ADG.
void replay(const string& map_path,const string& request_path,const string& tasks_path,
            const string& output_path) {
    std::ifstream map_file(map_path),request_file(request_path),tasks_file(tasks_path);
    json map,request,tasks;map_file>>map;request_file>>request;tasks_file>>tasks;
    SMARTGrid grid;grid.consider_rotation=false;grid.hold_endpoints=false;
    grid.useDummyPaths=false;grid.screen=0;
    require(grid.load_map_from_jsonstr(map.dump(),1,1),"Cannot load replay map");
    auto starts=request.at("mapf_instance").at("starts").get<vector<State>>();
    auto goals=request.at("mapf_instance").at("goals").get<vector<vector<Task>>>();
    vector<int> final_locations(starts.size(),-1);
    for(const auto& task:tasks.at("tasks")) {
        const int robot=task.at("robot_id").get<int>();
        require(robot>=0 && robot<static_cast<int>(starts.size()),"Invalid task robot ID");
        require(final_locations[robot]==-1,"Replay requires one transport task per robot");
        const auto& location=task.at("stages").back().at("location");
        final_locations[robot]=grid.getCellId(location.at(0).get<int>(),location.at(1).get<int>());
    }
    vector<int> locations;
    json terminal_ids=json::array();
    for(size_t a=0;a<goals.size();++a){
        auto& agent_goals=goals[a];
        require(agent_goals.size()==1,"Replay requires one current goal per robot");
        require(final_locations[a]>=0,"Missing robot's final transport stage");
        agent_goals.back().is_parking=agent_goals.back().location==final_locations[a];
        if(agent_goals.back().is_parking)terminal_ids.push_back(a);
        locations.push_back(agent_goals.back().location);
    }
    const int horizon=20;
    auto heuristic=std::make_shared<BasicHeuristicTable>(grid,locations,8403);
    SIPP single;PIBT planner(grid,single,heuristic,options(starts.size(),8403,horizon));
    auto paths=run(planner,grid,starts,goals,horizon);
    json parked=json::array(),arrived=json::array(),serialized=json::array();
    for(size_t a=0;a<paths.size();++a){
        int first=-1;
        json row=json::array();
        for(size_t t=0;t<paths[a].size();++t){
            row.push_back(paths[a][t].location);
            if(first<0 && paths[a][t].location==goals[a][0].location)first=t;
        }
        if(first>=0 && goals[a][0].is_parking){
            stays(paths[a],goals[a][0].location,first);arrived.push_back(a);
            if(first==0)parked.push_back(a);
        }
        serialized.push_back(row);
    }
    json result={{"status","passed"},{"scope","Discrete fallback request replay with corrected terminal metadata"},
        {"request_clock",request.at("clock")},{"horizon",horizon},{"robot_count",starts.size()},
        {"terminal_goal_robot_ids",terminal_ids},
        {"initially_at_terminal",parked},{"reached_terminal",arrived},{"paths",serialized}};
    std::ofstream output(output_path);output<<result.dump(2)<<'\n';
}
} // namespace

int main(int argc,char** argv) {
    try {
        spdlog::set_level(spdlog::level::warn);
        task_copy_checks();
        auto grid=grid_for({".....",".....","....."});
        for(int seed=0;seed<24;++seed){
            checks<PIBT>(grid,seed);checks<GuidedPIBT>(grid,seed);
            corridor_checks<PIBT>(seed);corridor_checks<GuidedPIBT>(seed);
        }
        if(argc==5)replay(argv[1],argv[2],argv[3],argv[4]);
        else require(argc==1,"Usage: pibt_parking [map.json request.json tasks.json output.json]");
        std::cout<<"PIBT and GuidedPIBT parking regressions passed across 24 seeds.\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
