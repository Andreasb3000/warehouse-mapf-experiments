#include "ExecutionManager.h"
#include <filesystem>
#include <chrono>
#include <cstdlib>
namespace fs=std::filesystem;
namespace {
void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
template<class F> void rejects(F f) {try{f();}catch(const std::exception&){return;}throw std::runtime_error("Invalid preview accepted");}
void write(const fs::path& p,const json& j) {std::ofstream f(p);f<<j.dump();}
template<class T> void option(boost::program_options::variables_map& vm, const string& key, T value) {
    vm.insert({key,boost::program_options::variable_value(boost::any(value),false)});
}
boost::program_options::variables_map options(const fs::path& directory) {
    boost::program_options::variables_map vm;
    for(const auto& [key,value] : vector<pair<string,int>>{{"screen",0},{"port_number",8080},
        {"total_sim_step_tick",40},{"ticks_per_second",10},{"sim_window_tick",20},{"look_ahead_tick",20},
        {"seed",0},{"num_robots",1},{"look_ahead_dist",4},{"rotation_time",1},
        {"sim_window_timestep",2},{"plan_window_timestep",5}}) option(vm,key,value);
    for(const auto& [key,value] : vector<pair<string,bool>>{{"save_stats",false},{"stop_at_congestion",false},
        {"save_heuristics_table",false}}) option(vm,key,value);
    for(const auto& [key,value] : vector<pair<string,string>>{{"output_file","unused.json"},
        {"planner_invoke_policy","default"},{"task_assigner_type","distinct_one_goal"},
        {"grid_type","regular"},{"backup_single_agent_solver","ASTAR"},
        {"heuristic_type","basic"},{"backup_planner","PIBT"},
        {"map",(directory/"map.json").string()},{"task_file",(directory/"tasks.txt").string()}}) option(vm,key,value);
    return vm;
}

Action move(int id,Location start,Location goal,double time) {
    Action a;a.robot_id=id;a.start=start;a.goal=goal;a.time=time;a.type='M';return a;
}
void detached_edges() {
    ADG live(2,0,1);string id="0";live.createRobotIDToStartIndexMaps(id,{1,1});id="1";live.createRobotIDToStartIndexMaps(id,{2,1});
    live.addMAPFPlan({{move(0,{1,1},{1,2},0)},{move(1,{2,1},{1,1},1)}});
    const auto before=live.getActionSnapshot();
    require(before["robots"][1]["actions"][0]["dependencies"].size()==1,"Fixture lacks dependency");
    auto preview=live.cloneForPreview();
    require(preview.getPlan(0).size()==1 && preview.updateFinishedNode(0,0),"Clone did not execute");
    require(preview.getPlan(1).size()==1,"Clone edge alias was not retained");
    require(live.getActionSnapshot()==before && live.getPlan(1).empty(),"Preview resolved a live dependency");
    auto appended=live.cloneForPreview();appended.addMAPFPlan({{move(0,{1,2},{1,3},2)},{}});
    require(live.getActionSnapshot()==before,"Appending preview changed original graph");
}
void acceptance(const fs::path& d,bool primary) {
    fs::create_directories(d/"routing");fs::create_directories(d/"search");
    write(d/"map.json",{{"n_row",5},{"n_col",5},{"weight",false},{"n_endpoint",0},{"n_agent_loc",1},
        {"maxtime",100},{"name","route_fixture"},{"layout",{".....",".....",".....",".....","....."}}});
    {std::ofstream f(d/"tasks.txt");f<<"1\n8,-1,0,0\n";}
    setenv("LSMART_ROUTING_DIR",(d/"routing").c_str(),1);setenv("LSMART_PLAN_SEARCH_DIR",(d/"search").c_str(),1);
    ExecutionManager manager(options(d));manager.init("0",{1,1});
    write(d/"routing/response_0.json",{{"clock",0}});
    require(manager.invokePlanner() && manager.isSimulationFrozen(),"Search did not freeze before cut");
    manager.getRobotsLocation();rejects([&]{manager.getRobotsLocation();});
    const auto before=manager.predictionActionSnapshot();
    const json direct={{{1,1,0,-1},{1,2,1,-1},{1,3,2,0}}};
    const json detour={{{1,1,0,-1},{2,1,1,-1},{2,2,2,-1},{2,3,3,-1},{1,3,4,0}}};
    const auto a=manager.previewPlan({{"clock",0},{"plan",direct}});
    const auto b=manager.previewPlan({{"clock",0},{"plan",detour}});
    require(a!=b && a["plan_version"]==1 && b["plan_version"]==1 && a["clock"]==1 && a["server_clock"]==0,
        "Prospective plans/metadata incorrect");
    require(manager.predictionActionSnapshot()==before,"Preview mutated live graph or queue");
    rejects([&]{manager.previewPlan({{"clock",1},{"plan",direct}});});
    auto wrong_start=direct;wrong_start[0][0][0]=0;
    rejects([&]{manager.previewPlan({{"clock",0},{"plan",wrong_start}});});
    rejects([&]{manager.previewPlan({{"clock",0},{"plan",json::array()}});});
    rejects([&]{manager.previewPlan({{"clock",0},{"plan",json::array({json::array()})}});});
    json plan={{"success",primary},{"congested",false},{"plan",direct},{"route_search",{
        {"clock",0},{"selected_index",0},{"selected_name","zero"},{"planner_seed",0},
        {"expected_preview_snapshot",primary ? a : json(nullptr)}}}};
    auto bad=plan;bad["route_search"]["clock"]=20;string text=bad.dump();rejects([&]{manager.addNewPlan(text);});
    require(manager.predictionActionSnapshot()==before,"Invalid selection mutated graph");
    if(primary) {
        bad=plan;bad["route_search"]["expected_preview_snapshot"]=b;text=bad.dump();rejects([&]{manager.addNewPlan(text);});
        require(manager.predictionActionSnapshot()==before,"Mismatching preview mutated graph");
    }
    text=plan.dump();manager.addNewPlan(text);
    auto actual=manager.predictionActionSnapshot();
    require(!manager.isSimulationFrozen() && actual["plan_version"]==1 && actual["robots"][0]["enqueued_node_id"]==-1,
        "Acceptance advanced physical state or dispatched actions");
    if(primary)require(actual["robots"]==a["robots"],"Exact saved winner was not installed");
    rejects([&]{manager.previewPlan({{"clock",0},{"plan",direct}});});
    unsetenv("LSMART_ROUTING_DIR");unsetenv("LSMART_PLAN_SEARCH_DIR");
}
}
int main() {
    const auto d=fs::temp_directory_path()/("route-preview-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    detached_edges();acceptance(d/"primary",true);acceptance(d/"fallback",false);fs::remove_all(d);
    std::cout<<"route preview purity and primary/fallback regressions passed\n";
}
