#!/usr/bin/env python3
"""Write a standalone, interactive replay of a recorded zone simulation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.baseline.analyze import _objects


def _json(path):
    return json.loads(path.read_text())


def _input_directory(run_dir):
    for directory in (run_dir, *run_dir.parents):
        if (directory / "config.json").is_file():
            config = _json(directory / "config.json")
            scenario_path = directory / config.get("scenario", "scenario.json")
            if scenario_path.is_file():
                return directory, config, scenario_path
        if directory == ROOT:
            break
    raise FileNotFoundError("Cannot find this run's config.json and scenario.json")


def build_replay_data(run_dir):
    """Load actual poses, native safety transitions, and acknowledged payload events."""
    run_dir = Path(run_dir).resolve()
    directory, config, scenario_path = _input_directory(run_dir)
    scenario = _json(scenario_path)
    map_document = _json(scenario_path.parent / scenario["map_file"])
    tasks_path = scenario_path.parent / scenario.get("tasks_file", "tasks.json")
    tasks = _json(tasks_path) if tasks_path.is_file() else {}
    frames = []
    with (run_dir / "trajectories.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            frames.append([frame["clock"], [[agent["id"], agent["x"], agent["y"], agent["rz"]]
                                             for agent in frame["agents"]]])
    if not frames:
        raise ValueError("Replay requires at least one recorded trajectory frame")
    if any(first[0] >= second[0] for first, second in zip(frames, frames[1:])):
        raise ValueError("Replay trajectory clocks must be strictly increasing")

    safety_history, safety_configs = defaultdict(list), {}
    log_path = run_dir / "simulator.log"
    goal_events = []
    saw_safety_config = False
    if log_path.is_file():
        for event in _objects(log_path.read_text(errors="replace")):
            if event.get("type") == "safety_config":
                saw_safety_config = True
                for settings in event.get("robots", []):
                    safety_configs[settings["agent_id"]] = settings
            elif event.get("type") == "safety_state":
                safety_history[event["agent_id"]].append([
                    event["clock"], event["mode"], event.get("blocker_kind", ""),
                    event.get("blocker_id", ""), event.get("trigger_reason", "not recorded"),
                ])
            elif event.get("type") == "goal_reached":
                goal_events.append(event)

    payload_path = run_dir / "payload_events.jsonl"
    payload_events = []
    if payload_path.is_file():
        payload_source = "payload_events.jsonl"
        with payload_path.open() as handle:
            payload_events = [json.loads(line) for line in handle if line.strip()]
    else:
        payload_source = "native goal_reached acknowledgements"
        task_by_robot = {task["robot_id"]: task for task in tasks.get("tasks", [])}
        next_stage = defaultdict(int)
        for event in sorted(goal_events, key=lambda value: value["clock"]):
            agent_id = event["agent"]
            task = task_by_robot.get(agent_id)
            if task is None or next_stage[agent_id] >= len(task["stages"]):
                continue
            stage = task["stages"][next_stage[agent_id]]
            if [event["row"], event["col"]] != stage["location"]:
                continue
            payload_events.append({
                "clock": event["clock"], "agent_id": agent_id,
                "operation": stage["operation"], "payload_id": task["payload_id"],
                "loaded": stage["operation"] == "pickup",
            })
            next_stage[agent_id] += 1
    payload_history = defaultdict(list)
    for event in payload_events:
        payload_history[event["agent_id"]].append([event["clock"], event["operation"],
                                                  event.get("payload_id", "")])
    for histories in (safety_history, payload_history):
        for history in histories.values():
            history.sort(key=lambda event: event[0])
    return {
        "name": scenario["name"], "description": scenario.get("description", ""),
        "run_dir": str(run_dir), "layout": map_document["layout"],
        "zones": scenario.get("zones", []), "agents": scenario["agents"],
        "network": scenario.get("network", {}), "tasks": tasks,
        "ticks_per_second": config["parameters"]["ticks_per_second"],
        "safety": scenario.get("safety", {}), "safety_configs": safety_configs,
        "safety_history": dict(safety_history), "safety_recorded": saw_safety_config,
        "payload_history": dict(payload_history), "payload_source": payload_source,
        "frames": frames,
    }


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Warehouse simulation replay</title>
<style>
:root{font:14px system-ui,sans-serif;color:#162337;background:#f0f4f8;color-scheme:light}*{box-sizing:border-box}
body{margin:0;padding:22px}header,main,footer{max-width:1500px;margin:auto}h1{font-size:25px;margin:0 0 7px}p{line-height:1.55;margin:5px 0 14px;color:#536175}
.badge{display:inline-block;background:#dbeafe;color:#1e40af;padding:4px 8px;border-radius:6px;font-size:11px;font-weight:700;margin-bottom:10px}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:white;padding:13px;border-radius:10px;margin:18px 0 12px;border:1px solid #d9e1eb}
button,select{font:inherit;border:1px solid #c8d2df;border-radius:6px;padding:7px 10px;background:white;color:#162337}button{cursor:pointer}button.primary{background:#2563eb;color:white;border-color:#2563eb;min-width:78px}
label{display:flex;align-items:center;gap:7px}input[type=range]{accent-color:#2563eb}#scrub{flex:1;min-width:180px}#time{font-variant-numeric:tabular-nums;min-width:120px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px}.card{background:white;border:1px solid #d9e1eb;border-radius:10px;padding:16px;margin-bottom:14px}.stage{padding:0;overflow:hidden;position:relative;align-self:start}
canvas{display:block;width:100%;height:auto;cursor:crosshair;background:#e2e8f0}.stage-note{padding:10px 14px;color:#536175;font-size:12px}
h2{font-size:16px;margin:0 0 12px}h3{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#64748b;margin:16px 0 7px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stat{padding:9px 5px;border-radius:7px;text-align:center;background:#f1f5f9}.stat strong{display:block;font-size:23px;font-variant-numeric:tabular-nums}.stat span{font-size:11px;color:#536175}.slow{color:#b45309}.stop{color:#b91c1c}.loaded{color:#7c3aed}
.details{display:grid;grid-template-columns:90px 1fr;gap:7px;font-size:12px}.details dt{color:#64748b}.details dd{margin:0;overflow-wrap:anywhere}.help{font-size:12px}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#536175}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.matrix{width:100%;border-collapse:collapse;font-size:10px;text-align:center}.matrix th{font-weight:500;padding:4px 0;color:#536175}.matrix td{padding:5px 1px;border:1px solid white;background:#f1f5f9}.matrix td.demand{background:#dbeafe;color:#1e40af;font-weight:700}
.tiny{font-size:11px;color:#64748b;overflow-wrap:anywhere}footer{font-size:12px;color:#64748b;margin-top:16px;line-height:1.6}#robot-select{width:100%;margin-bottom:12px}.zone-summary{font-size:12px;line-height:1.7}#selected-trail{width:14px;height:14px}
@media(max-width:960px){body{padding:12px}.layout{grid-template-columns:1fr}aside{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{margin-bottom:0}aside .card:last-child{grid-column:1/-1}}@media(max-width:600px){aside{display:block}.card{margin-bottom:12px}h1{font-size:21px}.controls{gap:8px}}
</style></head><body>
<header><span class="badge">RECORDED EXECUTION</span> <span class="badge" id="policy-label"></span><h1>Nine-zone warehouse replay</h1><p id="description"></p>
<div class="controls"><button id="play" class="primary" type="button">Play</button><button id="reset" type="button">Reset</button><span id="time"></span><input id="scrub" type="range" min="0" step="1" value="0" aria-label="Simulation time">
<label>Speed <select id="speed"><option value=".25">¼×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="5">5×</option><option value="10">10×</option></select></label>
<label>Polygons <select id="polygons"><option value="all" selected>All robots</option><option value="selected">Selected robot</option><option value="off">Hidden</option></select></label>
<label><input type="checkbox" id="selected-trail" checked>Selected trail</label></div></header>
<main class="layout"><section class="card stage"><canvas id="canvas" width="1100" height="1100" aria-label="Recorded robot motion across nine warehouse zones"></canvas><div class="stage-note">Click a robot to inspect it. Robot IDs are enlarged for readability; body discs and safety rectangles follow the map scale. Yellow regions mark junctions.</div></section>
<aside><section class="card"><h2>At this recorded time</h2><div class="stats"><div class="stat"><strong id="clear-count">0</strong><span>Clear</span></div><div class="stat slow"><strong id="slow-count">0</strong><span>Slow region</span></div><div class="stat stop"><strong id="stop-count">0</strong><span>Stop region</span></div><div class="stat loaded"><strong id="loaded-count">0</strong><span>Loaded</span></div><div class="stat"><strong id="delivered-count">0</strong><span>Delivered</span></div><div class="stat"><strong id="robot-count">0</strong><span>Robots</span></div></div>
<p class="help" id="safety-note"></p><div class="legend"><span><i class="dot" style="background:#2563eb"></i>Clear</span><span><i class="dot" style="background:#d97706"></i>Slow</span><span><i class="dot" style="background:#dc2626"></i>Stop</span><span><i class="dot" style="background:#7c3aed"></i>Payload square</span></div></section>
<section class="card"><h2>Inspect robot</h2><select id="robot-select" aria-label="Selected robot"></select><dl class="details" id="details"></dl><p class="tiny">The blocker and trigger reason are the most recently recorded controller report. Older recordings update only on mode changes. A safety region can remain active while a parked robot is stationary.</p></section>
<section class="card"><h2>Zone task allocation</h2><p class="help">Rows: pickup zone. Columns: delivery zone. Numbers count transport tasks.</p><table class="matrix" id="matrix" aria-label="Source to destination task allocation"></table><h3>Layout</h3><div class="zone-summary" id="zone-summary"></div></section></aside></main>
<footer>This replays actual recorded poses and safety transitions; it is not a forecast or a live connection. Loading is represented by acknowledged pickup/dropoff events. Payload identity and mass are labels, with no load-dependent vehicle physics.<br><span id="source"></span></footer>
<script id="replay-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('replay-data').textContent), $=id=>document.getElementById(id);
const canvas=$('canvas'),ctx=canvas.getContext('2d'),frames=data.frames,tps=data.ticks_per_second;
const rows=data.layout.length,cols=data.layout[0].length,pad=22,scale=Math.min((canvas.width-2*pad)/cols,(canvas.height-2*pad)/rows);
const agents=new Map(data.agents.map(agent=>[agent.id,agent])), safetyColors={clear:'#2563eb',slow:'#d97706',stop:'#dc2626'};
const xPixel=col=>pad+(col+.5)*scale,yPixel=row=>pad+(row+.5)*scale;
let index=0,selected=data.agents[0].id,playing=false,lastTimestamp=null,playClock=frames[0][0];
function stateAt(history,clock,fallback){if(!history||!history.length)return fallback;let left=0,right=history.length;while(left<right){let mid=(left+right)>>1;if(history[mid][0]<=clock)left=mid+1;else right=mid;}return left?history[left-1]:fallback;}
function safetyState(id,clock){return stateAt(data.safety_history[id],clock,[0,'clear','','','none']);}
function payloadState(id,clock){return stateAt(data.payload_history[id],clock,[0,'empty','']);}
function settings(id){return {robot_trigger:'body_in_outer',inner_stop_enabled:true,...data.safety,...(agents.get(id).safety||{}),...(data.safety_configs[id]||{})};}
function triggerLabel(mode){return mode==='outer_overlap'?'Outer polygons overlap':'Robot body enters outer polygon';}
function point(pose,forward,left){const yaw=pose[3],x=pose[1]+forward*Math.cos(yaw)-left*Math.sin(yaw),y=pose[2]+forward*Math.sin(yaw)+left*Math.cos(yaw);return [xPixel(-y),yPixel(-x)];}
const background=document.createElement('canvas');background.width=canvas.width;background.height=canvas.height;const bg=background.getContext('2d');
bg.fillStyle='#edf2f7';bg.fillRect(0,0,background.width,background.height);
for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){bg.fillStyle=data.layout[row][col]==='@'?'#d6dee8':'#ffffff';bg.fillRect(pad+col*scale,pad+row*scale,scale+.2,scale+.2);}
for(const zone of data.zones){const b=zone.bounds,z=zone.conflict_zone;bg.strokeStyle=zone.color||'#64748b';bg.lineWidth=1;bg.setLineDash([4,6]);bg.strokeRect(pad+b.cols[0]*scale,pad+b.rows[0]*scale,(b.cols[1]-b.cols[0]+1)*scale,(b.rows[1]-b.rows[0]+1)*scale);bg.setLineDash([]);bg.fillStyle='#fbbf2470';bg.fillRect(pad+z.cols[0]*scale,pad+z.rows[0]*scale,(z.cols[1]-z.cols[0]+1)*scale,(z.rows[1]-z.rows[0]+1)*scale);bg.fillStyle=zone.color||'#475569';bg.font='bold 14px system-ui';bg.fillText(`${zone.id} · ${zone.degree}-way${zone.degree===2?' corner':''}`,pad+(b.cols[0]+1)*scale,pad+(b.rows[0]+2)*scale);}
for(const agent of data.agents)for(let stage=0;stage<agent.goals.length;stage++){const [row,col]=agent.goals[stage],x=xPixel(col),y=yPixel(row);bg.strokeStyle=agent.color||'#64748b';bg.lineWidth=1;bg.strokeRect(x-.35*scale,y-.35*scale,.7*scale,.7*scale);bg.fillStyle='#64748b';bg.font='8px system-ui';bg.textAlign='center';bg.fillText(`${stage===0?'P':'D'}${agent.id}`,x,y+3);}
bg.textAlign='left';
function polygon(pose,config,prefix){const front=config[`${prefix}_front_m`],back=config[`${prefix}_back_m`],half=config[`${prefix}_half_width_m`];if(![front,back,half].every(Number.isFinite))return;ctx.beginPath();[[front,half],[front,-half],[-back,-half],[-back,half]].forEach(([f,l],i)=>{const [x,y]=point(pose,f,l);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.closePath();ctx.fillStyle=prefix==='slow'?'#f59e0b22':'#dc26262c';ctx.strokeStyle=prefix==='slow'?'#d97706':'#dc2626';ctx.lineWidth=prefix==='slow'?1:1.2;ctx.fill();ctx.stroke();}
function detail(label,value){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$('details').append(dt,dd);}
function draw(){const [clock,poses]=frames[index];ctx.drawImage(background,0,0);const show=$('polygons').value;
if($('selected-trail').checked){ctx.beginPath();ctx.lineWidth=2;ctx.strokeStyle='#7c3aed88';let started=false;const from=Math.max(0,index-100);for(let i=from;i<=index;i++){const pose=frames[i][1].find(p=>p[0]===selected);if(!pose)continue;const x=xPixel(-pose[2]),y=yPixel(-pose[1]);if(started)ctx.lineTo(x,y);else{ctx.moveTo(x,y);started=true;}}ctx.stroke();}
const counts={clear:0,slow:0,stop:0,loaded:0,delivered:0};for(const pose of poses){const id=pose[0],config=settings(id);if(config.enabled&&show!=='off'&&(show==='all'||id===selected)){polygon(pose,config,'slow');if(config.inner_stop_enabled)polygon(pose,config,'stop');}}
for(const pose of poses){const id=pose[0],config=settings(id),state=safetyState(id,clock),payload=payloadState(id,clock),x=xPixel(-pose[2]),y=yPixel(-pose[1]);counts[state[1]]=(counts[state[1]]||0)+1;if(payload[1]==='pickup')counts.loaded++;if(payload[1]==='dropoff')counts.delivered++;if(id===selected){ctx.beginPath();ctx.arc(x,y,10,0,2*Math.PI);ctx.fillStyle='#2563eb15';ctx.strokeStyle='#0f172a';ctx.lineWidth=1.4;ctx.fill();ctx.stroke();}ctx.beginPath();ctx.arc(x,y,(config.robot_radius_m||.085)*scale,0,2*Math.PI);ctx.fillStyle=safetyColors[state[1]]||'#2563eb';ctx.fill();const [hx,hy]=point(pose,.35,0);ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(hx,hy);ctx.strokeStyle=safetyColors[state[1]]||'#2563eb';ctx.lineWidth=1.7;ctx.stroke();if(payload[1]==='pickup'){ctx.fillStyle='#7c3aed';ctx.fillRect(x-2.5,y-2.5,5,5);}ctx.font='bold 10px system-ui';ctx.textAlign='center';ctx.lineWidth=3;ctx.strokeStyle='#ffffff';ctx.strokeText(String(id),x+9,y-7);ctx.fillStyle=safetyColors[state[1]]||'#2563eb';ctx.fillText(String(id),x+9,y-7);}
for(const key of ['clear','slow','stop','loaded','delivered'])$(key+'-count').textContent=(!data.safety_recorded&&['clear','slow','stop'].includes(key))?'—':counts[key];$('robot-count').textContent=poses.length;$('scrub').value=index;$('time').textContent=`${(clock/tps).toFixed(1)} / ${(frames.at(-1)[0]/tps).toFixed(1)} s`;
const agent=agents.get(selected),pose=poses.find(p=>p[0]===selected),state=safetyState(selected,clock),payload=payloadState(selected,clock),config=settings(selected);$('details').replaceChildren();detail('Robot',selected);detail('Transport',`${agent.zone_id||'?'} → ${agent.destination_zone_id||'?'}`);detail('Payload',`${agent.payload_id||'—'} · ${payload[1]==='pickup'?'loaded':payload[1]==='dropoff'?'delivered':'awaiting pickup'}`);detail('Safety',config.enabled?(data.safety_recorded?state[1]:'no recorded state'):'disabled');detail('Trigger rule',triggerLabel(config.robot_trigger));detail('Inner stop',config.inner_stop_enabled?'enabled':'disabled — outer only');detail('Trigger reason',state[4]||'not recorded');detail('Blocker',state[2]?`${state[2]} ${state[3]}`:'—');if(pose){detail('Grid pose',`row ${(-pose[1]).toFixed(2)}, col ${(-pose[2]).toFixed(2)}`);detail('Heading',`${(pose[3]*180/Math.PI).toFixed(1)}° (0° = north)`);}detail('Speed cap',config.enabled?`${Math.round(100*(config.speed_scale||.2))}% in slow region`:'unrestricted by rectangles');}
function setPlaying(value){playing=value;lastTimestamp=null;playClock=frames[index][0];$('play').textContent=playing?'Pause':'Play';}
$('play').addEventListener('click',()=>{if(!playing&&index===frames.length-1){index=0;draw();}setPlaying(!playing);});$('reset').addEventListener('click',()=>{setPlaying(false);index=0;draw();});$('scrub').max=frames.length-1;$('scrub').addEventListener('input',()=>{index=Number($('scrub').value);playClock=frames[index][0];lastTimestamp=null;draw();});$('polygons').addEventListener('change',draw);$('selected-trail').addEventListener('change',draw);
$('robot-select').addEventListener('change',()=>{selected=Number($('robot-select').value);draw();});canvas.addEventListener('click',event=>{const box=canvas.getBoundingClientRect(),x=(event.clientX-box.left)*canvas.width/box.width,y=(event.clientY-box.top)*canvas.height/box.height;let best=null,bestDistance=24;for(const pose of frames[index][1]){const distance=Math.hypot(xPixel(-pose[2])-x,yPixel(-pose[1])-y);if(distance<bestDistance){best=pose[0];bestDistance=distance;}}if(best!==null){selected=best;$('robot-select').value=best;draw();}});
function animate(timestamp){if(playing){if(lastTimestamp!==null)playClock+=(timestamp-lastTimestamp)/1000*tps*Number($('speed').value);lastTimestamp=timestamp;let previous=index;while(index+1<frames.length&&frames[index+1][0]<=playClock)index++;if(index!==previous)draw();if(index===frames.length-1)setPlaying(false);}requestAnimationFrame(animate);}
$('description').textContent=data.description;$('source').textContent=`Recording: ${data.run_dir} · Payload events: ${data.payload_source}`;const policies=[...new Set(data.agents.map(agent=>triggerLabel(settings(agent.id).robot_trigger)))],innerPolicies=[...new Set(data.agents.map(agent=>settings(agent.id).inner_stop_enabled))],anySafety=data.agents.some(agent=>settings(agent.id).enabled);$('policy-label').textContent=!anySafety?'Safety disabled':`${policies.join(' / ')} · ${innerPolicies.length===1?(innerPolicies[0]?'inner stop enabled':'outer only'):'mixed inner-stop settings'}`;$('safety-note').textContent=!anySafety?'Safety response is disabled for this recording.':data.safety_recorded?`Modes come from the native safety controller. The outer polygon caps forward motion at ${Math.round(100*settings(data.agents[0].id).speed_scale)}% of nominal speed. ${innerPolicies.includes(true)?'Enabled inner polygons stop movement and turning.':'Inner stopping is disabled; no inner polygons are drawn.'} Obstacle triggers use physical obstacle overlap with the outer polygon in both variants.`:'No native safety configuration was recorded. Polygon outlines show configured geometry only; safety counts are unavailable.';
for(const agent of data.agents){const option=document.createElement('option');option.value=agent.id;option.textContent=`Robot ${agent.id} · ${agent.zone_id||'?'} → ${agent.destination_zone_id||'?'}`;$('robot-select').append(option);}
const zoneOrder=data.network.zone_order||[],matrix=data.network.allocation_matrix||[];const header=document.createElement('tr');for(const label of ['↘',...zoneOrder]){const th=document.createElement('th');th.textContent=label;header.append(th);}$('matrix').append(header);matrix.forEach((row,i)=>{const tr=document.createElement('tr'),th=document.createElement('th');th.textContent=zoneOrder[i];tr.append(th);row.forEach(value=>{const td=document.createElement('td');td.textContent=value;if(value)td.className='demand';tr.append(td);});$('matrix').append(tr);});$('zone-summary').textContent=`${data.zones.filter(z=>z.degree===2).length} two-way corners · ${data.zones.filter(z=>z.degree===3).length} T junctions · ${data.zones.filter(z=>z.degree===4).length} four-way crossing. Initially ${data.network.robots_per_zone||'?'} robots per zone.`;
draw();requestAnimationFrame(animate);
</script></body></html>
'''


def write_viewer(run_dir):
    """Write run_dir/replay.html with no external assets or server dependency."""
    run_dir = Path(run_dir)
    data = build_replay_data(run_dir)
    # JSON is embedded as text, not executable JavaScript. Escape HTML-sensitive
    # characters so a payload/name cannot terminate the script element.
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    path = run_dir / "replay.html"
    path.write_text(HTML.replace("__DATA__", encoded))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(write_viewer(args.run_dir))


if __name__ == "__main__":
    main()
