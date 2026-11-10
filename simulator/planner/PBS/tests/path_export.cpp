#include "PBS.h"
#include "Instance.h"
#include <cstdio>

struct PBSPathExportTestAccess {
    static void setPath(PBS& planner, Path& path) { planner.paths = {&path}; }
};

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(PBS& planner, const vector<int>& locations,
           const map<int,int>& input_markers, const map<int,int>& expected_markers) {
    Path path;
    for (size_t i=0;i<locations.size();++i) {
        path.emplace_back(locations[i]);
        const auto marker=input_markers.find(static_cast<int>(i));
        if(marker!=input_markers.end()) path.back().task_id=marker->second;
    }
    PBSPathExportTestAccess::setPath(planner,path);
    const auto exported=planner.getPaths();
    require(exported.size()==1 && exported[0].size()==locations.size(),"Export changed route length");
    for(size_t i=0;i<locations.size();++i) {
        const auto& step=exported[0][i];
        require(get<0>(step)==locations[i]/7 && get<1>(step)==locations[i]%7 && get<2>(step)==double(i),
                "Service-marker correction changed geometry or timestamps");
        const auto marker=expected_markers.find(static_cast<int>(i));
        const int expected=marker==expected_markers.end() ? -1 : marker->second;
        require(get<3>(step)==expected,"Goal service marker missing, duplicated, or moved");
    }
}
}

int main() {
    const char* filename="/tmp/lsmart-path-export-test.json";
    const json map={{"n_row",5},{"n_col",7},{"weight",false},
        {"layout",{"@@@@@@@","@.....@","@.....@","@.....@","@@@@@@@"}}};
    {std::ofstream output(filename);output<<map.dump();}
    auto graph=make_shared<Graph>(filename,0);
    Instance instance(graph,0);
    instance.loadAgents({{"starts",{{{"location",15}}}},
        {"goals",{{{{"location",19},{"id",130}}}}}});
    PBS planner(instance,false,0);
    // Actual failure pattern: first goal visit at4s, wait at5s, leave,
    // and return at15s with the sole task marker supplied by A*.
    const vector<int> revisit={15,16,17,18,19,19,12,11,10,9,8,15,16,17,18,19};
    check(planner,revisit,{{15,130}},{{15,130}});
    // Preserve an earlier legitimate marker and suppress duplicate service.
    check(planner,revisit,{{4,130},{15,130}},{{4,130}});
    check(planner,revisit,{{5,130},{15,130}},{{5,130}});
    // Ordinary, already-correct paths remain exactly unchanged, including ID0.
    check(planner,{15,16,17,18,19},{{4,130}},{{4,130}});
    check(planner,{15,16,17,18,19},{{4,0}},{{4,0}});
    check(planner,{19},{{0,130}},{{0,130}});
    check(planner,{19},{},{});
    check(planner,revisit,{},{});
    // A marker away from the goal must not create a service action.
    check(planner,{15,16,17,18,19},{{1,999},{4,130}},{{4,130}});
    planner.clearSearchEngines();
    std::remove(filename);
    cout<<"Path export regressions passed: revisit4/5/15, earlier valid service, single marker, unchanged geometry/timestamps, ID0, parking."<<endl;
}
