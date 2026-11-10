#include "AdmissionBridge.h"
#include "ExecutionManager.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <thread>

namespace fs = std::filesystem;
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F fn) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Invalid admission bridge input accepted");
}
void write(const fs::path& path, const json& value) {
    std::ofstream output(path.string()+".tmp"); output << value.dump() << std::endl; output.close();
    fs::rename(path.string()+".tmp",path);
}
json original(int clock=20) {
    return {{"type","routing_request"},{"clock",clock},{"observation_clock",clock+1},
        {"boundary","pre_controller_before_commit_cut"},{"schema_version",1},
        {"pose_frame",{{"clock",clock+1},{"agents",json::array({{{"id",0},{"x",-1.0},{"y",-1.0}}})}}},
        {"controllers",json::array({{{"id",0},{"clock",clock+1},{"queue",json::array()}}})},
        {"snapshot",{{"server_clock",clock},{"plan_version",4ULL},{"robots",json::array()}}},
        {"physics_state_complete",false}};
}
json accepted(int clock=20) {
    return {{"server_clock",clock},{"simulation_frozen",true},{"plan_version",5ULL},
        {"robots",json::array({{{"id",0},{"actions",json::array({{{"type","M"}}})}}})}};
}
void metadata_and_validation() {
    auto before=original(), after=accepted();
    for(const string source : {"primary","fallback"}) {
        const auto request=admission_bridge::makeRequest(before,after,20,source);
        require(request["boundary"]=="post_planner_before_dispatch" && request["type"]=="admission_request" &&
            request["clock"]==20 && request["observation_clock"]==21 && request["planner_source"]==source,
            "Post-plan request timing/source metadata incorrect");
        require(request["snapshot"]["plan_version"]==5 && request["snapshot"]["clock"]==21 &&
            request["snapshot"]["server_clock"]==20 && request["snapshot"]["simulation_frozen"]==true &&
            request["snapshot"]["robots"]==after["robots"], "Request kept a stale plan");
        require(request["controllers"]==before["controllers"] && request["pose_frame"]==before["pose_frame"],
            "Post-plan bridge changed frozen controller/pose state");
    }
    auto bad=before; bad["clock"]=19; rejects([&]{admission_bridge::makeRequest(bad,after,20,"primary");});
    bad=before; bad["observation_clock"]=22; rejects([&]{admission_bridge::makeRequest(bad,after,20,"primary");});
    bad=before; bad["controllers"][0]["clock"]=20; rejects([&]{admission_bridge::makeRequest(bad,after,20,"primary");});
    bad=after; bad["simulation_frozen"]=false; rejects([&]{admission_bridge::makeRequest(before,bad,20,"primary");});
    bad=after; bad["plan_version"]=4ULL; rejects([&]{admission_bridge::makeRequest(before,bad,20,"primary");});
    auto parse=[](json p){return admission_bridge::parseResponse(p,20,5,1,61,61,40);};
    require(parse({{"clock",20},{"plan_version",5}}).held_robot_ids.empty(), "No-op response failed to clear gate");
    require(parse({{"clock",20},{"entry_gate",nullptr}}).held_robot_ids.empty(), "Null gate failed to clear policy");
    for(auto value : vector<json>{json::object(),{{"clock",19}},{{"clock",20.0}},{{"clock",true}},
             {{"clock",20},{"plan_version",4}},{{"clock",20},{"plan_version",5.0}},
             {{"clock",20},{"unexpected",true}},{{"clock",20},{"entry_gate",false}}})
        rejects([&]{parse(value);});
    json gate={{"center",{30,30}},{"half_width",3.0},{"held_robot_ids",{0}},
               {"until_clock",40},{"protect_clearance",false}};
    require(parse({{"clock",20},{"plan_version",5},{"entry_gate",gate}}).held_robot_ids.count(0),
            "Post-plan gate was not parsed");
    gate["until_clock"]=41; rejects([&]{parse({{"clock",20},{"entry_gate",gate}});});
    require(admission_bridge::timeoutSeconds(nullptr)==120 && admission_bridge::timeoutSeconds("1")==1,
            "Admission timeout defaults incorrect");
    for(const char* setting : {"", "0", "3601", "-1", "1.5", "nan", "12x"})
        rejects([&]{admission_bridge::timeoutSeconds(setting);});
}
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
void acceptance_path(const fs::path& directory, bool primary) {
    fs::create_directories(directory/"routing"); fs::create_directories(directory/"admission");
    write(directory/"map.json",{{"n_row",5},{"n_col",5},{"weight",false},{"n_endpoint",0},
        {"n_agent_loc",1},{"maxtime",100},{"name","admission_fixture"},
        {"layout",{".....",".....",".....",".....","....."}}});
    {std::ofstream tasks(directory/"tasks.txt"); tasks<<"1\n8,-1,0,0\n";}
    setenv("LSMART_ROUTING_DIR",(directory/"routing").c_str(),1);
    setenv("LSMART_ADMISSION_DIR",(directory/"admission").c_str(),1);
    setenv("LSMART_ADMISSION_TIMEOUT_SECONDS","2",1);
    ExecutionManager manager(options(directory)); manager.init("0",{1,1});
    if(primary) manager.freezeSimulationIfNecessary();
    auto before=original(0); before["snapshot"]=manager.predictionActionSnapshot();
    write(directory/"routing/request_0.json",before); write(directory/"routing/response_0.json",{{"clock",0}});
    require(manager.invokePlanner(),"Fixture did not invoke planner");
    require(manager.isSimulationFrozen(),"Admission invoke did not establish the frozen barrier");
    manager.getRobotsLocation();
    auto host=std::async(std::launch::async,[&] {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(!fs::exists(directory/"admission/request_0.json")) {
            if(std::chrono::steady_clock::now()>deadline) throw std::runtime_error("No common-path admission request");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::ifstream input(directory/"admission/request_0.json"); json request; input>>request;
        write(directory/"admission/response_0.json",{{"clock",0},{"plan_version",request["plan_version"]}});
        return request;
    });
    string plan=json({{"success",primary},{"plan",{{{1,1,0,-1},{1,2,1,-1},{1,3,2,0}}}},
                      {"congested",false}}).dump();
    manager.addNewPlan(plan);
    const auto request=host.get();
    require(request["planner_source"]==(primary ? "primary" : "fallback") && request["snapshot"]["plan_version"]==1,
            "Primary/fallback acceptance missed common barrier");
    require(request["snapshot"]["robots"][0]["actions"].size()>0 &&
            request["controllers"]==before["controllers"] && request["pose_frame"]==before["pose_frame"],
            "Common barrier did not combine accepted actions with original frozen state");
    require(!manager.isSimulationFrozen() && manager.getCurrSimStep()==0,
            "Decision exchange advanced simulation or failed to unfreeze");
    require(manager.predictionActionSnapshot()["robots"][0]["enqueued_node_id"]==-1,
            "Admission exchange dispatched actions while frozen");
    unsetenv("LSMART_ADMISSION_DIR"); unsetenv("LSMART_ROUTING_DIR"); unsetenv("LSMART_ADMISSION_TIMEOUT_SECONDS");
}
void stale_files_and_timeout(const fs::path& directory) {
    fs::create_directories(directory/"routing"); fs::create_directories(directory/"admission");
    write(directory/"routing/request_20.json",original());
    write(directory/"admission/response_20.json",{{"clock",20},{"plan_version",5}});
    auto exchange=[&]{admission_bridge::exchange((directory/"routing").string(),(directory/"admission").string(),
        1,accepted(),20,"primary",1,61,61,40);};
    rejects(exchange);
    require(!fs::exists(directory/"admission/request_20.json"),"Stale response was accepted or overwritten");
    fs::remove(directory/"admission/response_20.json");
    const auto start=std::chrono::steady_clock::now(); rejects(exchange);
    require(std::chrono::steady_clock::now()-start>=std::chrono::seconds(1),"Missing response did not honor bounded wait");
}
}
int main() {
    const auto directory=fs::temp_directory_path()/("admission-bridge-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    metadata_and_validation(); acceptance_path(directory/"primary",true); acceptance_path(directory/"fallback",false);
    stale_files_and_timeout(directory/"invalid"); fs::remove_all(directory);
    std::cout<<"admission bridge regressions passed\n";
}
