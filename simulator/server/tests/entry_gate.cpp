#include "ADG.h"
#include <limits>
#include <sstream>

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F fn) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Invalid gate was accepted");
}
json policy() {
    return {{"center", {10,20}}, {"half_width", 3.0},
            {"held_robot_ids", {0}}, {"until_clock", 20}};
}
EntryGate gate() { return EntryGate::parse(policy(), 0, 2, 61, 61, 20); }
EntryGate boundedGate() {
    auto value=policy(); value["protect_clearance"]=false;
    return EntryGate::parse(value,0,2,61,61,20);
}
Action move(int id, Location start_rc, Location goal_rc, double time=0, char kind='M') {
    Action a;
    a.robot_id=id; a.type=kind; a.time=time;
    a.start={start_rc.second,start_rc.first};
    a.goal={goal_rc.second,goal_rc.first};
    return a;
}
void init(ADG& adg, int count) {
    for(int i=0;i<count;++i) { string id=std::to_string(i); adg.createRobotIDToStartIndexMaps(id,{20,6}); }
}
void validation_geometry() {
    auto p=policy(); auto g=gate();
    require(g.enters({6,20},{7,20}), "Boundary contact must count as entry");
    require(g.enters({6,20},{14,20}), "Outside-to-outside crossing missed");
    require(g.enters({6,17},{14,17}), "Tangent to closed square missed");
    require(!g.enters({6,16},{14,16}), "Parallel outside path blocked");
    require(!g.enters({10,20},{16,20}), "Interior/exit action blocked");
    require(!g.enters({7,20},{6,20}), "Boundary exit blocked");
    require(!g.enters({20,6},{20,14}), "Row-column coordinate order lost");
    require(g.blocks(0,19,'M',{6,20},{7,20},false,false), "Eligible action not blocked");
    require(!g.blocks(0,20,'M',{6,20},{7,20},false,false), "Gate did not expire exactly");
    require(!g.blocks(1,0,'M',{6,20},{7,20},false,false), "Unselected robot blocked");
    require(!g.blocks(0,0,'T',{6,20},{7,20},false,false), "Turn blocked");
    require(!g.blocks(0,0,'S',{6,20},{7,20},false,false), "Service blocked");
    require(!g.blocks(0,0,'M',{6,20},{7,20},true,false), "Enqueued action blocked");
    require(!g.blocks(0,0,'M',{6,20},{7,20},false,true), "Committed action blocked");
    require(g.blocks(0,0,'M',{3,20},{4,20},false,false), "Upstream approach not staged");
    require(g.blocks(0,0,'M',{4,14},{5,14},false,false), "Diagonal-side approach not staged");
    require(!g.blocks(0,0,'M',{4,20},{3,20},false,false), "Away motion blocked");
    require(!g.blocks(0,0,'M',{4,19},{4,20},false,false), "Parallel motion blocked");
    require(!g.blocks(0,0,'M',{10,20},{16,20},false,false), "Interior exit blocked");
    require(!g.blocks(0,0,'M',{3,20},{4,20},true,false), "Queued approach blocked");
    require(!g.blocks(0,0,'M',{3,20},{4,20},false,true), "Committed approach blocked");
    for(auto ids : vector<json>{json::array({-1}),json::array({2}),json::array({0,0}),json::array({true}),json::array({0.5})}) {
        auto bad=p; bad["held_robot_ids"]=ids;
        rejects([&]{ EntryGate::parse(bad,0,2,61,61,20); });
    }
    for(auto expiry : vector<json>{0,21,-1,20.5,true,2147483648ULL}) {
        auto bad=p; bad["until_clock"]=expiry;
        rejects([&]{ EntryGate::parse(bad,0,2,61,61,20); });
    }
    for(auto width : vector<json>{0,-1,"3",std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        auto bad=p; bad["half_width"]=width;
        rejects([&]{ EntryGate::parse(bad,0,2,61,61,20); });
    }
    for(auto center : vector<json>{json::array({1,20}),json::array({10,60}),json::array({10}),json::array({10,"20"}),json::array({10,std::numeric_limits<double>::infinity()})}) {
        auto bad=p; bad["center"]=center;
        rejects([&]{ EntryGate::parse(bad,0,2,61,61,20); });
    }
    auto extra=p; extra["ignored"]=1; rejects([&]{EntryGate::parse(extra,0,2,61,61,20);});
    require(g.protect_clearance, "Legacy payload lost strict clearance protection");
    require(!boundedGate().protect_clearance, "Explicit bounded mode ignored");
    for(auto setting : vector<json>{0,1,"false",nullptr,json::array()}) {
        auto bad=p; bad["protect_clearance"]=setting;
        rejects([&]{EntryGate::parse(bad,0,2,61,61,20);});
    }
    auto empty=p; empty["held_robot_ids"]=json::array();
    require(!EntryGate::parse(empty,0,2,61,61,20).blocks(0,0,'M',{6,20},{7,20},false,false), "Empty selection changed behavior");
}
void dispatch_expiry_and_logs() {
    ADG adg(1,0,1); init(adg,1);
    auto service=move(0,{7,20},{7,20},1,'S'); service.task_id=77;
    adg.addMAPFPlan({{move(0,{6,20},{7,20}),service}});
    const auto before=adg.getActionSnapshot();
    adg.setEntryGate(gate()); adg.setEntryGateClock(0);
    std::ostringstream captured; auto* old=std::cout.rdbuf(captured.rdbuf());
    require(adg.getPlan(0).empty() && adg.getPlan(0).empty(), "Gate dispatched entering action");
    adg.setEntryGateClock(1); require(adg.getPlan(0).empty(), "Gate expired prematurely");
    std::cout.rdbuf(old);
    std::istringstream lines(captured.str()); string line; int count=0;
    while(std::getline(lines,line)) {
        auto event=json::parse(line);
        require(event["type"]=="entry_gate_blocked" && event["queue_empty"]==true &&
                event["protect_clearance"]==true &&
                event["action_start"]==json({6,20}) && event["clock"]==count,
                "Gate event lost timing, queue status, or row-column geometry");
        ++count;
    }
    require(count==2, "Gate events were not deduplicated per tick/robot");
    require(adg.getActionSnapshot()==before, "Gate modified accepted actions or dependencies");
    adg.setEntryGateClock(20);
    require(adg.getPlan(0).size()==2, "Expired gate did not release actions");
    require(adg.isTaskCompletionNode(0,1), "Gate lost finite task completion");
}
void committed_and_clear() {
    ADG queued(1,0,1); init(queued,1); queued.addMAPFPlan({{move(0,{6,20},{7,20})}});
    require(queued.getPlan(0).size()==1, "Initial enqueue failed");
    const auto before=queued.getActionSnapshot();
    queued.setEntryGate(gate());
    require(queued.getPlan(0).empty() && queued.getActionSnapshot()==before, "Gate revoked or duplicated queued action");
    require(queued.updateFinishedNode(0,0), "Gate prevented queued action completion");
    ADG committed(1,0,1); init(committed,1); committed.addMAPFPlan({{move(0,{6,20},{7,20})}});
    committed.computeCommitCut(); committed.addMAPFPlan({{}}); committed.setEntryGate(gate());
    require(committed.getActionSnapshot()["robots"][0]["committed_node_id"]==0,
            "Test did not construct committed but unqueued prefix");
    require(committed.getPlan(0).size()==1, "Gate withheld committed prefix");
    ADG clear(1,0,1); init(clear,1); clear.addMAPFPlan({{move(0,{6,20},{7,20})}});
    clear.setEntryGate(gate()); require(clear.getPlan(0).empty(), "Test gate inactive");
    clear.setEntryGate(EntryGate{}); require(clear.getPlan(0).size()==1, "Clearing gate did not restore dispatch");
    for(auto action : vector<Action>{move(0,{10,20},{11,20}),move(0,{10,20},{16,20}),move(0,{6,20},{6,20},0,'T')}) {
        ADG allowed(1,0,1); init(allowed,1); allowed.addMAPFPlan({{action}}); allowed.setEntryGate(gate());
        require(allowed.getPlan(0).size()==1, "Interior, exit, or turn withheld");
    }
}
void dependencies() {
    ADG adg(2,0,1); init(adg,2);
    adg.addMAPFPlan({{move(0,{6,20},{7,20})},{move(1,{6,19},{6,20},1)}});
    adg.setEntryGate(gate());
    require(adg.getPlan(0).empty() && adg.getPlan(1).empty(), "Gate bypassed ordinary dependency");
    require(adg.getActionSnapshot()["robots"][1]["actions"][0]["dependencies"].size()==1,
            "Gate artificially cleared dependency");
    adg.setEntryGateClock(20); require(adg.getPlan(0).size()==1, "Release failed");
    adg.updateFinishedNode(0,0); require(adg.getPlan(1).size()==1, "Finished action did not release dependency");
    ADG exit(2,0,1); init(exit,2);
    exit.addMAPFPlan({{move(0,{6,20},{7,20})},{move(1,{8,20},{6,20},1)}});
    exit.setEntryGate(gate());
    require(exit.getPlan(0).size()==1, "Gate held prerequisite of an interior exit");
    ADG chain(3,0,1); init(chain,3);
    chain.addMAPFPlan({{move(0,{6,20},{7,20})},
                      {move(1,{6,19},{6,20},1)},
                      {move(2,{8,20},{6,19},2)}});
    chain.setEntryGate(gate());
    require(chain.getPlan(0).size()==1,
            "Gate held transitive prerequisite A->B->C of an interior exit");
    ADG sequencing(3,0,1); init(sequencing,3);
    sequencing.addMAPFPlan({{move(0,{6,20},{7,20})},
                           {move(1,{6,19},{6,20},1),move(1,{6,20},{5,20},2),move(1,{5,20},{4,20},3)},
                           {move(2,{8,20},{5,20},4)}});
    sequencing.setEntryGate(gate());
    require(sequencing.getPlan(0).size()==1,
            "Gate ignored implicit same-robot sequencing in clearance dependencies");
}
void approach_dispatch() {
    ADG staged(1,0,1); init(staged,1);
    staged.addMAPFPlan({{move(0,{3,20},{4,20}), move(0,{4,20},{5,20},1)}});
    staged.setEntryGate(gate());
    require(staged.getPlan(0).empty(), "Upstream approach reached execution queue");
    require(staged.getActionSnapshot()["robots"][0]["enqueued_node_id"]==-1,
            "Staging mutated the queued prefix");
    staged.setEntryGateClock(20);
    require(staged.getPlan(0).size()==2, "Expiry failed to release staged approach");
    ADG clearance(2,0,1); init(clearance,2);
    clearance.addMAPFPlan({{move(0,{3,20},{4,20})}, {move(1,{10,20},{3,20},1)}});
    clearance.setEntryGate(gate());
    require(clearance.getPlan(0).size()==1, "Upstream staging held exit prerequisite");
}
void bounded_clearance() {
    for(bool transitive : {false,true}) {
        const int count=transitive ? 3 : 2;
        ADG adg(count,0,1); init(adg,count);
        vector<vector<Action>> plans{{move(0,{6,20},{7,20})}};
        if(transitive) plans.push_back({move(1,{6,19},{6,20},1)});
        plans.push_back({move(count-1,{8,20},transitive ? Location{6,19} : Location{6,20},2)});
        adg.addMAPFPlan(plans); const auto before=adg.getActionSnapshot();
        adg.setEntryGate(boundedGate());
        require(adg.getPlan(0).empty(), "Bounded mode unexpectedly exempted uncommitted exit ancestor");
        require(adg.getPlan(count-1).empty(), "Bounded mode bypassed dependent exit ordering");
        require(adg.getActionSnapshot()==before, "Bounded mode modified dependencies or routes");
        adg.setEntryGateClock(20); require(adg.getPlan(0).size()==1, "Bounded hold did not expire");
        adg.updateFinishedNode(0,0);
        if(transitive) { require(adg.getPlan(1).size()==1, "Intermediate dependency did not release"); adg.updateFinishedNode(1,0); }
        require(adg.getPlan(count-1).size()==1, "Exit did not release after actual prerequisite completion");
    }
    ADG committed(2,0,1); init(committed,2);
    committed.addMAPFPlan({{move(0,{5,20},{6,20}),move(0,{6,20},{7,20},1)},
                           {move(1,{8,20},{6,20},2)}});
    committed.computeCommitCut(); committed.addMAPFPlan({{}, {}});
    require(committed.getActionSnapshot()["robots"][0]["committed_node_id"]==1,
            "Commitment closure did not retain the other robot's prerequisite");
    committed.setEntryGate(boundedGate());
    require(committed.getPlan(0).size()==2, "Bounded mode withheld another committed robot's prerequisite");
    require(committed.getPlan(1).empty(), "Bounded mode released dependency before execution");
    committed.updateFinishedNode(0,1);
    require(committed.getPlan(1).size()==1, "Committed dependent exit failed to release");
}
}
int main() {
    validation_geometry(); dispatch_expiry_and_logs(); committed_and_clear(); dependencies(); approach_dispatch(); bounded_clearance();
    std::cout << "entry gate regressions passed\n";
}
