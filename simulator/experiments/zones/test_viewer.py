"""Check replay event fidelity and safe standalone serialization."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from experiments.zones.generate import write_case
from experiments.zones.viewer import HTML, build_replay_data, write_viewer


class ZoneReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = write_case(Path(self.temporary.name))
        self.run = self.directory / "run-01"
        self.run.mkdir()
        self.scenario = json.loads((self.directory / "scenario.json").read_text())
        self.agent = self.scenario["agents"][0]
        frames = [{"clock": clock, "agents": [{"id": agent["id"],
                    "x": -agent["start"][0], "y": -agent["start"][1], "rz": 0} for agent in self.scenario["agents"]]}
                  for clock in (0, 1, 2, 3)]
        (self.run / "trajectories.jsonl").write_text("\n".join(json.dumps(frame) for frame in frames))
        pickup, dropoff = self.agent["goals"]
        events = [
            {"type": "safety_config", "robots": [{"agent_id": 0, "enabled": True, "slow_front_m": 1.8}]},
            {"type": "safety_state", "clock": 1, "agent_id": 0, "mode": "slow", "blocker_kind": "robot", "blocker_id": "1"},
            {"type": "goal_reached", "clock": 2, "agent": 0, "row": pickup[0], "col": pickup[1]},
            {"type": "goal_reached", "clock": 2, "agent": 0, "row": pickup[0], "col": pickup[1]},
            {"type": "goal_reached", "clock": 3, "agent": 0, "row": dropoff[0], "col": dropoff[1]},
        ]
        (self.run / "simulator.log").write_text("\n".join("[worker] " + json.dumps(event) for event in events))

    def test_actual_pose_and_safety_events_preserved_and_duplicate_goals_ignored(self):
        data = build_replay_data(self.run)
        self.assertEqual(data["frames"][0][1][0], [0, -self.agent["start"][0], -self.agent["start"][1], 0])
        self.assertEqual(data["safety_history"][0], [[1, "slow", "robot", "1", "not recorded"]])
        self.assertEqual(data["safety_configs"][0]["slow_front_m"], 1.8)
        self.assertEqual(data["payload_history"][0], [[2, "pickup", "payload-000"], [3, "dropoff", "payload-000"]])

    def test_prefers_analyzed_payload_events(self):
        event = {"clock": 1, "agent_id": 0, "operation": "pickup", "payload_id": "validated", "loaded": True}
        (self.run / "payload_events.jsonl").write_text(json.dumps(event) + "\n")
        data = build_replay_data(self.run)
        self.assertEqual(data["payload_history"][0], [[1, "pickup", "validated"]])
        self.assertEqual(data["payload_source"], "payload_events.jsonl")

    def test_native_trigger_reason_and_inner_policy_are_preserved(self):
        with (self.run / "simulator.log").open("a") as stream:
            stream.write("\n" + json.dumps({"type": "safety_state", "clock": 2, "agent_id": 0,
                                          "mode": "slow", "blocker_kind": "robot", "blocker_id": "2",
                                          "trigger_reason": "outer_polygon_overlap"}))
        data = build_replay_data(self.run)
        self.assertEqual(data["safety_history"][0][-1], [2, "slow", "robot", "2", "outer_polygon_overlap"])

    def test_embedded_data_cannot_terminate_script_and_has_no_external_requests(self):
        self.scenario["description"] = "</script><script>alert('injected')</script>"
        (self.directory / "scenario.json").write_text(json.dumps(self.scenario))
        path = write_viewer(self.run)
        text = path.read_text()
        self.assertEqual(path, self.run / "replay.html")
        self.assertNotIn(self.scenario["description"], text)
        self.assertNotIn("__DATA__", text)
        self.assertNotIn("fetch(", text)
        self.assertNotIn("src=\"http", text)
        raw = text.split('<script id="replay-data" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertEqual(json.loads(raw)["description"], self.scenario["description"])

    def test_rejects_empty_or_nonchronological_trajectory(self):
        for frames in ([], [{"clock": 1, "agents": []}, {"clock": 0, "agents": []}]):
            (self.run / "trajectories.jsonl").write_text("\n".join(json.dumps(frame) for frame in frames))
            with self.assertRaises(ValueError):
                build_replay_data(self.run)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to exercise the standalone JavaScript")
    def test_javascript_renders_events_and_maps_physical_heading(self):
        # Exercise the actual embedded JavaScript with a minimal DOM/canvas,
        # including safety/payload state changes and the coordinate transform.
        script = HTML.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        harness = r'''
const vm=require('vm'),assert=require('assert');
const replay=DATA,context=new Proxy({}, {get:()=>()=>{}}),elements=new Map();
function element(id=''){return {value:id==='polygons'?'all':id==='speed'?'1':'0',checked:true,
width:1100,height:1100,textContent:'',children:[],append(...items){this.children.push(...items);},
replaceChildren(){this.children=[];},addEventListener(){},getContext(){return context;}};}
const document={getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);},createElement(){return element();}};
document.getElementById('replay-data').textContent=JSON.stringify(replay);
const scope=vm.createContext({document,requestAnimationFrame(){}});vm.runInContext(SCRIPT,scope);
assert.equal(document.getElementById('loaded-count').textContent,0);
vm.runInContext('index=2;draw()',scope);
assert.equal(document.getElementById('loaded-count').textContent,1);
assert.equal(document.getElementById('slow-count').textContent,1);
vm.runInContext('index=3;draw()',scope);
assert.equal(document.getElementById('loaded-count').textContent,0);
assert.equal(document.getElementById('delivered-count').textContent,1);
const north=vm.runInContext('point([0,-10,-20,0],1,0)',scope);
const left=vm.runInContext('point([0,-10,-20,0],0,1)',scope);
const center=vm.runInContext('point([0,-10,-20,0],0,0)',scope);
assert(north[1]<center[1]);assert.equal(north[0],center[0]);
assert(left[0]<center[0]);assert.equal(left[1],center[1]);
vm.runInContext("for(const agent of data.agents){data.safety_configs[agent.id]={...settings(agent.id),robot_trigger:'outer_overlap',inner_stop_enabled:false};}let drawn=[];polygon=(pose,config,prefix)=>drawn.push(prefix);draw();",scope);
assert(vm.runInContext("drawn.includes('slow') && !drawn.includes('stop')",scope));
assert(document.getElementById('details').children.some(node=>node.textContent==='Outer polygons overlap'));
assert(document.getElementById('details').children.some(node=>node.textContent==='disabled — outer only'));
vm.runInContext("delete data.safety.inner_stop_enabled;for(const agent of data.agents){delete data.safety_configs[agent.id].inner_stop_enabled;delete (agent.safety||{}).inner_stop_enabled;}drawn=[];draw();",scope);
assert(vm.runInContext("drawn.includes('stop')",scope));
'''
        harness = harness.replace("DATA", json.dumps(build_replay_data(self.run))).replace("SCRIPT", json.dumps(script))
        result = subprocess.run([shutil.which("node")], input=harness, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
