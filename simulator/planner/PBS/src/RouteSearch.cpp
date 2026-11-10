#include "RouteSearch.h"
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cstdio>

namespace route_search {
namespace {
vector<double> penalties(const json& a, size_t cells) {
    if (!a.is_array() || a.size()!=cells) throw std::runtime_error("Route costs must cover every map cell");
    vector<double> result;
    for (const auto& v : a) {
        if (!v.is_number() || v.is_boolean()) throw std::runtime_error("Route cost is not numeric");
        const double x=v.get<double>();
        if (!std::isfinite(x) || x<0 || x>1000) throw std::runtime_error("Route cost outside [0,1000]");
        result.push_back(x);
    }
    return result;
}
bool zero(const vector<double>& a) { return std::all_of(a.begin(),a.end(),[](double x){return x==0;}); }
}
vector<Candidate> parseCandidates(const json& r, int clock, size_t cells) {
    if (!r.is_object() || !r.contains("clock") || !r["clock"].is_number_integer() || r["clock"]!=clock ||
        !r.contains("cell_penalties") || !zero(penalties(r["cell_penalties"],cells)) ||
        (r.contains("entry_gate") && !r["entry_gate"].is_null()) || !r.contains("route_candidates") ||
        !r["route_candidates"].is_array() || r["route_candidates"].empty() || r["route_candidates"].size()>3)
        throw std::runtime_error("Route search requires zero initial costs and one to three candidates");
    vector<Candidate> result; set<string> names;
    for (const auto& c : r["route_candidates"]) {
        if (!c.is_object() || c.size()!=2 || !c.contains("name") || !c["name"].is_string() ||
            c["name"].get<string>().empty() || !names.insert(c["name"].get<string>()).second ||
            !c.contains("cell_penalties")) throw std::runtime_error("Invalid route candidate");
        result.push_back({c["name"].get<string>(),penalties(c["cell_penalties"],cells)});
    }
    if (!zero(result[0].penalties)) throw std::runtime_error("Route candidate zero must be the unmodified planner");
    return result;
}
uint32_t plannerSeed(int seed, int clock) {
    return static_cast<uint32_t>(seed) ^ (static_cast<uint32_t>(clock)*UINT32_C(0x9e3779b9));
}
int timeoutSeconds() {
    const char* raw=std::getenv("LSMART_PLAN_SEARCH_TIMEOUT_SECONDS");
    if (!raw || !*raw) return 120;
    const string s(raw);
    if (s.find_first_not_of("0123456789")!=string::npos) throw std::runtime_error("Invalid route search timeout");
    const int value=std::stoi(s);
    if(value<1 || value>3600) throw std::runtime_error("Route search timeout outside [1,3600]");
    return value;
}
int parseSelection(const json& r, const json& request) {
    if (!r.is_object() || r.size()!=2 || !r.contains("clock") || !r["clock"].is_number_integer() ||
        r["clock"]!=request.at("clock") || !r.contains("selected_index") || !r["selected_index"].is_number_integer())
        throw std::runtime_error("Invalid or stale route selection");
    if(r["selected_index"]<0 || r["selected_index"]>=request.at("candidates").size()) throw std::runtime_error("Invalid route candidate index");
    const int i=r["selected_index"].get<int>();
    const auto& c=request["candidates"][i];
    if(!c.at("evaluated").get<bool>() || (!c.at("success").get<bool>() && i!=0))
        throw std::runtime_error("Cannot select an unsuccessful alternate route");
    return i;
}
json makeRequest(const json& original, int clock, uint32_t seed, const json& instance, const json& candidates) {
    if(original.at("clock")!=clock || original.at("observation_clock")!=clock+1 ||
       original.at("pose_frame").at("clock")!=clock+1 || candidates.empty())
        throw std::runtime_error("Frozen route-search request has inconsistent clocks");
    for(const auto& c : original.at("controllers")) if(c.at("clock")!=clock+1)
        throw std::runtime_error("Frozen controller clock differs from route-search observation");
    for(const auto& c : candidates) if(c.at("success").get<bool>()) {
        const auto& s=c.at("preview_snapshot");
        if(s.at("server_clock")!=clock || s.at("clock")!=clock+1 || !s.at("simulation_frozen").get<bool>() ||
           s.at("plan_version").get<unsigned long long>() != original.at("snapshot").at("plan_version").get<unsigned long long>()+1)
            throw std::runtime_error("Candidate preview does not describe the current prospective accepted plan");
    }
    json request=original;
    request["type"]="plan_search_request"; request["boundary"]="post_commit_before_plan_selection";
    request["planner_seed"]=seed; request["seed_protocol"]="uint32(seed) XOR uint32(clock)*0x9e3779b9";
    request["mapf_instance"]=instance; request["candidates"]=candidates;
    return request;
}
int exchange(const string& directory, const json& request, int timeout_seconds) {
    boost::filesystem::create_directories(directory);
    const string suffix=std::to_string(request.at("clock").get<int>())+".json";
    const string input=directory+"/request_"+suffix, output=directory+"/response_"+suffix;
    if(boost::filesystem::exists(input) || boost::filesystem::exists(output))
        throw std::runtime_error("Stale route-search exchange files");
    {std::ofstream file(input+".tmp"); file<<request.dump()<<std::endl; if(!file) throw std::runtime_error("Cannot write route request");}
    if(std::rename((input+".tmp").c_str(),input.c_str())!=0) throw std::runtime_error("Cannot publish route request");
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(timeout_seconds);
    while(!boost::filesystem::exists(output)) {
        if(std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("Route selection timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::ifstream file(output); json response; file>>response;
    return parseSelection(response,request);
}
}
