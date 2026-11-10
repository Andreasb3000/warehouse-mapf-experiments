#include "RouteSearch.h"
#include <future>
namespace {
void require(bool x,const char* why) {if(!x)throw std::runtime_error(why);}
template<class F> void rejects(F f) {try{f();}catch(const std::exception&){return;}throw std::runtime_error("Invalid route input accepted");}
}
int main() {
    json response={{"clock",20},{"cell_penalties",{0,0,0}},{"route_candidates",json::array({
        {{"name","zero"},{"cell_penalties",{0,0,0}}},{{"name","mild"},{"cell_penalties",{0,2,0}}}})}};
    require(route_search::parseCandidates(response,20,3).size()==2,"Candidates not parsed");
    for(auto bad : vector<json>{json::object(),{{"clock",19}},{{"clock",20.0}}}) rejects([&]{route_search::parseCandidates(bad,20,3);});
    for(const string key : {"cell_penalties","route_candidates"}) {auto bad=response;bad[key]=nullptr;rejects([&]{route_search::parseCandidates(bad,20,3);});}
    for(auto costs : vector<json>{{1,0,0},{0,-1,0},{0,true,0},{0,1001,0},{0,0}}) {
        auto bad=response;bad["route_candidates"][0]["cell_penalties"]=costs;rejects([&]{route_search::parseCandidates(bad,20,3);});
    }
    auto duplicate=response;duplicate["route_candidates"][1]["name"]="zero";rejects([&]{route_search::parseCandidates(duplicate,20,3);});
    auto gate=response;gate["entry_gate"]={{"held_robot_ids",{0}}};rejects([&]{route_search::parseCandidates(gate,20,3);});
    const auto seed=route_search::plannerSeed(83,20);srand(seed);const auto first=rand();rand();rand();srand(seed);
    require(first==rand() && seed!=route_search::plannerSeed(83,40),"Per-boundary seed isolation failed");
    json original={{"clock",20},{"observation_clock",21},{"pose_frame",{{"clock",21}}},
        {"controllers",json::array({{{"clock",21}}})},{"snapshot",{{"plan_version",4}}}};
    json preview={{"clock",21},{"server_clock",20},{"simulation_frozen",true},{"plan_version",5}};
    json candidates=json::array({{{"evaluated",true},{"success",true},{"preview_snapshot",preview}},
        {{"evaluated",true},{"success",false},{"preview_snapshot",nullptr}},
        {{"evaluated",false},{"success",false},{"preview_snapshot",nullptr}}});
    auto request=route_search::makeRequest(original,20,seed,json::object(),candidates);
    require(request["pose_frame"]==original["pose_frame"] && request["controllers"]==original["controllers"] &&
        request["boundary"]=="post_commit_before_plan_selection","Frozen metadata lost");
    auto stale=candidates;stale[0]["preview_snapshot"]["plan_version"]=4;
    rejects([&]{route_search::makeRequest(original,20,seed,json::object(),stale);});
    require(route_search::parseSelection({{"clock",20},{"selected_index",0}},request)==0,"No-op rejected");
    for(auto bad : vector<json>{{{"clock",19},{"selected_index",0}},{{"clock",20},{"selected_index",1}},
        {{"clock",20},{"selected_index",2}},{{"clock",20},{"selected_index",-1}},
        {{"clock",20},{"selected_index",true}},{{"clock",20},{"selected_index",0.0}},
        {{"clock",20},{"selected_index",4294967296ULL}},
        {{"clock",20},{"selected_index",0},{"extra",1}}}) rejects([&]{route_search::parseSelection(bad,request);});
    request["candidates"][0]["success"]=false;
    require(route_search::parseSelection({{"clock",20},{"selected_index",0}},request)==0,"Base failure must retain normal fallback");
    const string directory="/tmp/route-search-test-"+std::to_string(route_search::plannerSeed(7,20));
    boost::filesystem::remove_all(directory);boost::filesystem::create_directories(directory);
    {std::ofstream out(directory+"/response_20.json");out<<"{}";}
    rejects([&]{route_search::exchange(directory,request,1);});boost::filesystem::remove(directory+"/response_20.json");
    rejects([&]{route_search::exchange(directory,request,1);});
    require(boost::filesystem::exists(directory+"/request_20.json"),"Request was not atomically published");
    boost::filesystem::remove_all(directory);
    cout<<"route candidate/selection regressions passed"<<endl;
}
