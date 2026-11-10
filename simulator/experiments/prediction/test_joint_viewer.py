"""Regression checks for frozen forecast rendering and safe serialization."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from experiments.prediction.joint_viewer import HTML, write_viewer


def demo():
    def pose(row, col, agent_id=0, mode="normal"):
        return {"id": agent_id, "row": row, "col": col, "yaw": 0, "mode": mode}
    return {
        "title": "Joint rollout check",
        "scenario": {"layout": ["@@@@@", "@...@", "@...@", "@...@", "@@@@@"], "cell_size_m": 1},
        "metrics": {"summary": "Recorded comparison"},
        "links": [{"label": "Stress experiment", "href": "../stress/index.html"},
                  {"label": "Unsafe URL", "href": "javascript:alert(1)"},
                  {"label": "Documentation", "href": "https://example.com/docs/2026"}],
        "samples": [{
            "origin_s": 10, "plan_version": 2, "trigger": "outer_overlap", "diagnostics": {"plan_exhausted_at_s": {"0": .5, "1": None}},
            "frames": [
                {"offset_s": 0, **{key: [pose(3, 2), pose(2, 3, 1)] for key in ("actual", "joint", "plan_only", "hold")}},
                {"offset_s": .5, "actual": [pose(2.8, 2), pose(2, 3, 1)],
                 "joint": [pose(2.9, 2, mode="slow"), pose(2, 3, 1)],
                 "plan_only": [pose(2.5, 2), pose(2, 3, 1)], "hold": [pose(3, 2), pose(2, 3, 1)]},
                {"offset_s": 1, "actual": [pose(2.6, 2), pose(2, 3, 1)],
                 "joint": [pose(2.8, 2, mode="slow"), pose(2, 3, 1)],
                 "plan_only": [pose(2, 2), pose(2, 3, 1)], "hold": [pose(3, 2), pose(2, 3, 1)]},
            ],
        }],
    }


class JointViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, data):
        (self.directory / "demo_data.json").write_text(json.dumps(data))
        return write_viewer(self.directory)

    def test_self_contained_safe_data_and_retained_forecast_frames(self):
        data = demo()
        data["title"] = '</script><script>alert("untrusted")</script>'
        path = self.write(data)
        html = path.read_text()
        self.assertEqual(path.name, "index.html")
        self.assertNotIn(data["title"], html)
        self.assertNotIn("__DATA__", html)
        self.assertNotIn("fetch(", html)
        self.assertNotIn('src="http', html)
        embedded = html.split('<script id="forecast-data" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertEqual(json.loads(embedded), data)
        self.assertIn("without interpolation", html)
        self.assertIn("Future observations and future plans are excluded", html)

    def test_rejects_invalid_geometry_and_nonchronological_frames(self):
        variants = []
        data = demo(); data["scenario"]["layout"][0] = "@"; variants.append(data)
        data = demo(); data["samples"] = []; variants.append(data)
        data = demo(); data["samples"][0]["frames"][1]["offset_s"] = 0; variants.append(data)
        data = demo(); data["samples"][0]["frames"][1]["joint"][0]["row"] = float("nan"); variants.append(data)
        data = demo(); del data["samples"][0]["frames"][1]["hold"]; variants.append(data)
        data = demo(); data["samples"][0]["frames"][1]["actual"].append(copy.deepcopy(data["samples"][0]["frames"][1]["actual"][0])); variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    self.write(data)

    @unittest.skipUnless(shutil.which("node"), "Node required for browser script smoke test")
    def test_javascript_scrub_method_selection_and_frozen_playback(self):
        script = HTML.rsplit("<script>", 1)[1].split("</script>", 1)[0]
        harness = r'''
const vm=require('vm'),assert=require('assert'),elements=new Map(),context=new Proxy({}, {get:()=>()=>{}});
function element(id=''){return {value:id==='speed'?'1':'0',width:820,height:820,textContent:'',children:[],events:{},classList:{remove(){}},append(...nodes){this.children.push(...nodes);},replaceChildren(){this.children=[];},addEventListener(type,fn){this.events[type]=fn;},getContext(){return context;},getBoundingClientRect(){return {left:0,top:0,width:820,height:820};}};}
const document={title:'',getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);},createElement(){return element();},querySelectorAll(){return [];}};
document.getElementById('forecast-data').textContent=JSON.stringify(DATA);
const scope=vm.createContext({document,requestAnimationFrame(){}});vm.runInContext(SCRIPT,scope);
assert.equal(document.getElementById('mean-error').textContent,'0.00');
assert.equal(document.getElementById('context-trigger').textContent,'Outer polygons overlap');
assert.equal(document.getElementById('summary').textContent,'Recorded comparison');
assert.equal(document.getElementById('resource-links').children.length,2);
assert.equal(document.getElementById('resource-links').children[1].href,'https://example.com/docs/2026');
assert.equal(document.getElementById('resource-links').children[0].href,'../stress/index.html');
assert.equal(document.getElementById('scrub').max,'2');
vm.runInContext('index=1;draw()',scope);
assert.equal(document.getElementById('mean-error').textContent,'0.05');
assert.equal(document.getElementById('p95-error').textContent,'0.10');
assert.equal(document.getElementById('slow-count').textContent,'1');
assert.equal(document.getElementById('absolute').textContent,'simulator time 10.5 s');
assert.equal(document.getElementById('context-coverage').textContent,'0 / 2 before this horizon');
vm.runInContext("selected='0';draw()",scope);
assert.equal(document.getElementById('mean-error').textContent,'0.10');
assert.equal(document.getElementById('error-label').textContent,'POSITION ERROR · ROBOT 0');
vm.runInContext("method='plan_only';draw()",scope);
assert.equal(document.getElementById('mean-error').textContent,'0.30');
assert.equal(document.getElementById('forecast-title').textContent,'Plan only');
vm.runInContext("method='hold';draw()",scope);
assert.equal(document.getElementById('mean-error').textContent,'0.20');
// Playback picks the last retained frame; it does not invent a pose at +0.75s.
vm.runInContext("setOrigin(0);playing=true;lastTime=0;tick(750)",scope);
assert.equal(vm.runInContext('index',scope),1);
assert.equal(document.getElementById('offset').textContent,'+0.5 s');
vm.runInContext('tick(1000)',scope);
assert.equal(vm.runInContext('index',scope),2);
assert.equal(vm.runInContext('playing',scope),false);
assert(document.getElementById('context-coverage').textContent.startsWith('1 / 2 before this horizon'));
const origin=vm.runInContext('sample().origin_s',scope);assert.equal(origin,10);
// A click chooses a plotted robot by ID rather than changing the forecast.
vm.runInContext("const g=geometry($('actual-canvas')),p=point(frame().actual[0],g);pick({clientX:p[0],clientY:p[1]},$('actual-canvas'),'actual')",scope);
assert.equal(vm.runInContext('selected',scope),'0');
assert.equal(vm.runInContext('sample().frames[2].joint[0].row',scope),2.8);
'''
        harness = harness.replace("DATA", json.dumps(demo())).replace("SCRIPT", json.dumps(script))
        completed = subprocess.run([shutil.which("node")], input=harness, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
