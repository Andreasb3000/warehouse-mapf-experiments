// Exercise terminal metadata in both PBS low-level solvers and the full PBS
// priority search. No behavior is inferred solely from the exported marker.
#include "PBS.h"
#include "SpaceTimeAStar.h"
#include "SIPP.h"
#include <cstdio>

namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}

json instance_json(bool marked,bool movable=false,int task=-1,bool omit=false) {
    json stationary={{"location",17},{"id",task}};
    if(!omit){stationary["is_parking"]=marked;stationary["movable"]=movable;}
    return {{"starts",{{{"location",15}},{{"location",17}}}},
        {"goals",{{{{"location",19},{"id",10}}},{stationary}}}};
}

template<class Solver> void low_level(const shared_ptr<Graph>& graph) {
    Instance instance(graph,0);
    Path crossing;
    for(int location:{15,16,17,18,19})crossing.emplace_back(location);
    vector<Path*> paths={&crossing,nullptr};

    // This used to return a leave-and-return route for an idle completed
    // robot. A conflicting higher-priority branch must now be infeasible.
    for(int task:{-1,42}) {
        instance.loadAgents(instance_json(true,false,task));
        Solver parked(instance,1);
        const auto conflicting=parked.findOptimalPath({0},paths,1);
        require(conflicting.empty(),"Primary planner evicted a terminal robot to satisfy a crossing path");
        const auto clear=parked.findOptimalPath({},paths,1);
        require(clear.size()==1 && clear[0].location==17 && clear[0].task_id==task,
                "Terminal WAIT lost its location or pending service marker");
    }

    // Terminal robots are fixed for every active route even before PBS has
    // selected a priority relation. The direct path crosses the parked cell.
    Solver active(instance,0);
    vector<Path*> empty(2,nullptr);
    const auto detour=active.findOptimalPath({},empty,0);
    require(!detour.empty() && detour.front().location==15 && detour.back().location==19,
            "Active robot lost its feasible bypass around parking");
    require(std::none_of(detour.begin(),detour.end(),[](const PathEntry& p){return p.location==17;}),
            "Root low-level path crossed a terminal parked cell");

    // Intermediate/lifelong/movable goals retain ordinary yielding semantics.
    // Reusing the instance must clear a previous terminal flag, including when
    // old-format JSON does not contain the newly optional metadata.
    for(const auto& input:vector<json>{instance_json(false),instance_json(false,false,42),
            instance_json(true,true),instance_json(false,false,-1,true)}) {
        instance.loadAgents(input);
        Solver yielding(instance,1);
        const auto path=yielding.findOptimalPath({0},paths,1);
        require(!path.empty() && path.front().location==17 && path.back().location==17,
                "A nonterminal robot could no longer yield and return");
        require(std::any_of(path.begin(),path.end(),[](const PathEntry& p){return p.location!=17;}),
                "Yielding path ignored the conflicting higher-priority crossing");
        for(size_t t=0;t<path.size();++t) {
            const int other=crossing[std::min(t,crossing.size()-1)].location;
            require(path[t].location!=other,"Yielding path contains a vertex conflict");
            if(t) require(!(path[t-1].location==other && path[t].location==crossing[std::min(t-1,crossing.size()-1)].location),
                          "Yielding path contains an edge swap");
        }
    }

    // A final task that has not yet arrived is allowed to reach its bay.
    instance.loadAgents({{"starts",{{{"location",15}}}},
        {"goals",{{{{"location",19},{"id",55},{"is_parking",true},{"movable",false}}}}}});
    Solver moving(instance,0);
    const auto final=moving.findOptimalPath({},vector<Path*>(1,nullptr),0);
    require(final.size()==5 && final.back().location==19 && final.back().task_id==55,
            "Terminal metadata froze a robot before arrival");
}

void high_level(const shared_ptr<Graph>& graph,bool sipp,int seed) {
    srand(seed);
    Instance instance(graph,0);instance.loadAgents(instance_json(true));
    PBS planner(instance,sipp,0);
    require(planner.solve(5),"PBS failed a simple instance with parking and a bypass");
    const auto paths=planner.getPaths();
    require(paths.size()==2 && !paths[0].empty() && paths[1].size()==1,
            "PBS did not retain the terminal robot as a stationary route");
    require(std::get<0>(paths[1][0])==2 && std::get<1>(paths[1][0])==3,
            "PBS moved the terminal goal");
    for(const auto& state:paths[0])require(!(std::get<0>(state)==2 && std::get<1>(state)==3),
            "PBS accepted another robot through permanent parking");
    require(std::get<0>(paths[0].back())==2 && std::get<1>(paths[0].back())==5,
            "PBS lost the active robot's destination");
    planner.clearSearchEngines();
}
} // namespace

int main(int argc,char** argv) {
    const string filename="/tmp/lsmart-terminal-parking-test.json";
    try {
        string solver=argc>1?argv[1]:"both";
        require(solver=="both" || solver=="ASTAR" || solver=="SIPP","Unknown solver selection");
        const json map={{"n_row",5},{"n_col",7},{"weight",false},
            {"layout",{"@@@@@@@","@.....@","@.....@","@.....@","@@@@@@@"}}};
        {std::ofstream file(filename);file<<map.dump();}
        auto graph=std::make_shared<Graph>(filename,0);
        if(solver!="SIPP")low_level<SpaceTimeAStar>(graph);
        if(solver!="ASTAR")low_level<SIPP>(graph);
        for(int seed=0;seed<12;++seed) {
            if(solver!="SIPP")high_level(graph,false,seed);
            if(solver!="ASTAR")high_level(graph,true,seed);
        }
        std::remove(filename.c_str());
        std::cout<<"PBS terminal parking regressions passed ("<<solver<<", 12 priority seeds).\n";
        return 0;
    } catch(const std::exception& error){
        std::remove(filename.c_str());std::cerr<<error.what()<<'\n';return 1;
    }
}
