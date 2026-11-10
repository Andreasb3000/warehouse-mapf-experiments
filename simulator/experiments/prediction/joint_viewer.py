#!/usr/bin/env python3
"""Build a self-contained viewer for frozen forecasts and recorded outcomes."""

import argparse
import json
import math
from pathlib import Path


def _validate(data):
    layout = data.get("scenario", {}).get("layout", [])
    if not layout or any(not isinstance(row, str) or len(row) != len(layout[0]) for row in layout):
        raise ValueError("scenario.layout must be a nonempty rectangular list of strings")
    samples = data.get("samples", [])
    if not samples:
        raise ValueError("At least one forecast sample is required")
    for sample in samples:
        if not isinstance(sample.get("origin_s"), (int, float)) or not math.isfinite(sample["origin_s"]):
            raise ValueError("Each sample needs a finite origin_s")
        frames = sample.get("frames", [])
        if not frames:
            raise ValueError("Each sample needs recorded forecast frames")
        previous = -1
        for frame in frames:
            offset = frame.get("offset_s")
            if not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0 or offset <= previous:
                raise ValueError("Frame offsets must be finite, nonnegative, and strictly increasing")
            previous = offset
            for method in ("actual", "joint", "plan_only", "hold"):
                poses = frame.get(method)
                if not isinstance(poses, list):
                    raise ValueError(f"Frame needs a {method} pose list")
                seen = set()
                for pose in poses:
                    if "id" not in pose or str(pose["id"]) in seen:
                        raise ValueError("Pose IDs must be present and unique within each frame")
                    seen.add(str(pose["id"]))
                    if any(not isinstance(pose.get(key), (int, float)) or not math.isfinite(pose[key])
                           for key in ("row", "col", "yaw")):
                        raise ValueError("Poses require finite row, col, and yaw")
    return data


def write_viewer(output_dir):
    """Read demo_data.json and return the generated standalone index.html path.

    Coordinates are zero-indexed row/column and yaw is the native ARGoS heading:
    yaw=0 faces decreasing row, yaw=pi/2 faces decreasing column. Optional
    scenario.cell_size_m converts grid distances to metres. No interpolation,
    observations, or route updates are introduced by this viewer.
    """
    output_dir = Path(output_dir)
    data = _validate(json.loads((output_dir / "demo_data.json").read_text()))
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    path = output_dir / "index.html"
    path.write_text(HTML.replace("__DATA__", encoded))
    return path


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Warehouse forecast comparison</title>
<style>
:root{color-scheme:light;--ink:#192f43;--muted:#627487;--blue:#2464c4;--orange:#d96b26;--line:#dce4ea;--paper:#fff;--bg:#f2f5f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1440px;margin:auto;padding:28px 28px 36px}h1{margin:3px 0 8px;font-size:28px;letter-spacing:-.7px;line-height:1.2}p{margin:0}.eyebrow{font-size:11px;letter-spacing:1.5px;font-weight:750;color:var(--blue);text-transform:uppercase}.subtitle{max-width:970px;color:var(--muted)}.badge{display:inline-block;padding:4px 9px;border:1px solid #cbdbe9;border-radius:20px;color:#34536e;background:#f5faff;font-size:11px;font-weight:650}.headerline{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}.controls{margin:22px 0 14px;display:grid;grid-template-columns:minmax(200px,1.4fr) minmax(160px,1fr) minmax(150px,.7fr) minmax(100px,.6fr);gap:12px;background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px}label{display:block;font-size:11px;font-weight:700;color:var(--muted);letter-spacing:.3px;margin-bottom:5px}select,button,input{font:inherit}select{width:100%;padding:9px 10px;border:1px solid #cbd7e1;border-radius:7px;background:white;color:var(--ink)}button{background:var(--blue);border:0;border-radius:7px;color:white;padding:9px 16px;cursor:pointer;font-weight:650}button.secondary{background:#eaf0f7;color:var(--ink)}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #89b5f2;outline-offset:2px}.transport{display:flex;align-items:center;gap:12px;background:var(--paper);padding:12px 16px;border:1px solid var(--line);border-radius:10px}.transport input{flex:1;min-width:60px;accent-color:var(--blue)}.clock{font-variant-numeric:tabular-nums;white-space:nowrap;font-size:13px;min-width:130px}.clock strong{font-size:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}.stat{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:12px 15px}.stat small{display:block;color:var(--muted);font-size:11px}.stat b{font-size:20px;letter-spacing:-.4px}.stat span{font-size:12px;color:var(--muted);margin-left:5px}.maps{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;overflow:hidden}.panelhead{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px 16px;border-bottom:1px solid var(--line)}.panelhead h2{margin:0;font-size:14px}.panelhead small{font-size:11px;color:var(--muted)}.mapcanvas{display:block;width:100%;height:auto;aspect-ratio:1;cursor:crosshair}.legend{display:flex;gap:16px;flex-wrap:wrap;padding:10px 16px;border-top:1px solid var(--line);font-size:11px;color:var(--muted)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--blue);margin-right:5px}.dot.orange{background:var(--orange)}.dot.ghost{background:#acc8ed}.bottom{display:grid;grid-template-columns:1.4fr 1fr;gap:14px;margin-top:14px}.chartcanvas{display:block;width:100%;height:170px}.details{padding:14px 16px}.details dl{display:grid;grid-template-columns:135px 1fr;gap:6px 12px;margin:0}.details dt{color:var(--muted)}.details dd{margin:0;overflow-wrap:anywhere}.note{color:var(--muted);font-size:12px;margin-top:14px}.note strong{color:var(--ink)}.methodnote{min-height:42px;margin-top:10px;color:var(--muted);font-size:12px}.footnote{color:var(--muted);font-size:11px;margin-top:10px}.resource-links{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.resource-links a{font-size:12px;color:var(--blue);text-decoration:none;border:1px solid #cbdbe9;padding:5px 10px;border-radius:6px;background:white}.resource-links a:hover{text-decoration:underline}.summary{padding:10px 14px;background:#eaf1f8;border-radius:8px;margin-top:14px;font-size:12px;color:#35516b}.hidden{display:none}@media(max-width:850px){main{padding:18px 14px}.controls{grid-template-columns:1fr 1fr}.stats{grid-template-columns:1fr 1fr}.maps,.bottom{grid-template-columns:1fr}.headerline{display:block}.headerline .badge{margin-top:10px}h1{font-size:24px}.transport{gap:8px;padding:10px}.clock{min-width:108px;font-size:11px}.clock strong{font-size:16px}}
</style></head><body><main>
<div class="headerline"><div><div class="eyebrow">Warehouse robot research · prediction</div><h1 id="title">Frozen forecasts, measured outcomes</h1><p class="subtitle">Advance every robot from one observation, then compare the predicted motion with the recorded simulator. Choose a forecast origin and inspect where the predictions diverge.</p></div><span class="badge">OFFLINE EVALUATION</span></div>
<div id="summary" class="summary hidden"></div><nav id="resource-links" class="resource-links hidden" aria-label="Related experiment reports"></nav>
<div class="controls"><div><label for="origin">FORECAST ORIGIN</label><select id="origin"></select></div><div><label for="method">PREDICTION METHOD</label><select id="method"><option value="joint">Joint rollout + safety</option><option value="plan_only">Plan only</option><option value="hold">Hold current position</option></select></div><div><label for="robot">ROBOT / TRACE</label><select id="robot"><option value="all">All robots</option></select></div><div><label for="speed">PLAYBACK SPEED</label><select id="speed"><option value="1">1× real time</option><option value="2">2×</option><option value="5">5×</option></select></div></div>
<div class="transport"><button id="play">Play</button><button id="reset" class="secondary" title="Return to forecast origin">↺</button><input id="scrub" type="range" min="0" value="0" step="1" aria-label="Forecast horizon frame"><div class="clock"><strong id="offset">+0.0 s</strong><br><span id="absolute">simulator time 0.0 s</span></div></div>
<div class="methodnote" id="method-note"></div>
<div class="stats"><div class="stat"><small id="error-label">MEAN POSITION ERROR · ALL ROBOTS</small><b id="mean-error">—</b><span class="unit"></span></div><div class="stat"><small>95TH PERCENTILE · ALL ROBOTS</small><b id="p95-error">—</b><span class="unit"></span></div><div class="stat"><small>ROBOTS COMPARED</small><b id="robot-count">—</b><span>matched IDs</span></div><div class="stat"><small id="mode-label">JOINT ROLLOUT · SLOWING</small><b id="slow-count">—</b><span id="mode-unit">robots at this frame</span></div></div>
<div class="maps"><section class="panel"><div class="panelhead"><h2>Recorded execution</h2><small>Held-out future · display and scoring only</small></div><canvas id="actual-canvas" class="mapcanvas" width="820" height="820" aria-label="Recorded robot positions"></canvas><div class="legend"><span><i class="dot"></i>Recorded robot</span><span>Click a robot to inspect its trace</span></div></section><section class="panel"><div class="panelhead"><h2 id="forecast-title">Joint rollout + safety</h2><small id="plan-label">Current accepted plan</small></div><canvas id="forecast-canvas" class="mapcanvas" width="820" height="820" aria-label="Predicted robot positions"></canvas><div class="legend"><span><i class="dot orange"></i>Predicted robot</span><span><i class="dot ghost"></i>Recorded reference</span><span>Line = selected robot’s position error</span></div></section></div>
<div class="bottom"><section class="panel"><div class="panelhead"><h2 id="chart-title">Mean position error over the horizon</h2><small id="chart-unit">Distance</small></div><canvas id="error-chart" class="chartcanvas" width="900" height="190" aria-label="Position error across forecast horizon"></canvas><div class="legend"><span><i class="dot orange"></i>Joint + safety</span><span><i class="dot"></i>Plan only</span><span><i class="dot" style="background:#8c98a3"></i>Hold</span></div></section><section class="panel"><div class="panelhead"><h2>Forecast context</h2><small>Fixed at its origin</small></div><div class="details"><dl><dt>Origin</dt><dd id="context-origin"></dd><dt>Plan version</dt><dd id="context-plan"></dd><dt>Safety trigger</dt><dd id="context-trigger"></dd><dt>Selected robot</dt><dd id="context-robot">All robots</dd><dt>Recorded samples</dt><dd id="context-frames"></dd><dt>Joint plan exhausted</dt><dd id="context-coverage">Not recorded</dd></dl></div></section></div>
<p class="note"><strong>A conditional forecast, not a live replan.</strong> Predictions use the state and accepted actions available at the chosen origin. Future observations and future plans are excluded. The forecast stays frozen while you play or scrub; the blue recording is used only to show and score what happened. Unscheduled actions, later replanning, and controller differences can cause divergence. A known action list may end because a robot finished or because its next task route has not yet been supplied; the forecast cannot invent that future route.</p>
<p class="footnote">The viewer displays retained forecast frames directly, without interpolation. Position error is one diagnostic; it does not by itself measure congestion, collisions, or routing quality. Trails show the selected robot from the forecast origin up to the displayed frame.</p>
</main><script id="forecast-data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('forecast-data').textContent);
const $=id=>document.getElementById(id),layout=data.scenario.layout,rows=layout.length,cols=layout[0].length;
const scaleM=Number(data.scenario.cell_size_m)>0?Number(data.scenario.cell_size_m):1,unit=Number(data.scenario.cell_size_m)>0?'m':'grid units';
const names={joint:'Joint rollout + safety',plan_only:'Plan only',hold:'Hold current position'};
const colors={joint:'#d96b26',plan_only:'#2464c4',hold:'#8c98a3'};
let originIndex=0,index=0,method='joint',selected='all',playing=false,lastTime=null,playElapsed=0;
const sample=()=>data.samples[originIndex],frame=()=>sample().frames[index];
function addOption(element,value,label){const option=document.createElement('option');option.value=String(value);option.textContent=label;element.append(option);}
function displayValue(value){return value===undefined||value===null?'not recorded':typeof value==='object'?JSON.stringify(value):String(value);}
function percentile(values,p){if(!values.length)return null;const ordered=[...values].sort((a,b)=>a-b);return ordered[Math.max(0,Math.ceil(p*ordered.length)-1)];}
function pairs(fr,kind){const recorded=new Map(fr.actual.map(p=>[String(p.id),p]));return fr[kind].flatMap(p=>{const a=recorded.get(String(p.id));return a?[{id:String(p.id),actual:a,predicted:p,error:Math.hypot(p.row-a.row,p.col-a.col)*scaleM}]:[];});}
function mean(values){return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;}
function errorsAt(fr,kind,selection='all'){return pairs(fr,kind).filter(p=>selection==='all'||p.id===selection).map(p=>p.error);}
function fixed(value){return value===null||!Number.isFinite(value)?'—':value.toFixed(2);}
function geometry(canvas){const margin=28,cell=Math.min((canvas.width-margin*2)/cols,(canvas.height-margin*2)/rows);return {cell,x:(canvas.width-cell*cols)/2,y:(canvas.height-cell*rows)/2};}
function point(p,g){return [g.x+(p.col+.5)*g.cell,g.y+(p.row+.5)*g.cell];}
function background(ctx,canvas,g){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f7f9fb';ctx.fillRect(g.x,g.y,cols*g.cell,rows*g.cell);ctx.fillStyle='#ccd7df';for(let row=0;row<rows;row++)for(let col=0;col<cols;col++)if('@#TOW'.includes(layout[row][col]))ctx.fillRect(g.x+col*g.cell,g.y+row*g.cell,g.cell+.3,g.cell+.3);ctx.strokeStyle='#d3dde5';ctx.strokeRect(g.x,g.y,cols*g.cell,rows*g.cell);}
function trace(ctx,g,kind,color){if(selected==='all')return;ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.globalAlpha=.7;ctx.beginPath();let started=false;for(let i=0;i<=index;i++){const pose=sample().frames[i][kind].find(p=>String(p.id)===selected);if(!pose){started=false;continue;}const [x,y]=point(pose,g);if(started)ctx.lineTo(x,y);else ctx.moveTo(x,y);started=true;}ctx.stroke();ctx.globalAlpha=1;}
function robot(ctx,p,g,color,ghost=false){const [x,y]=point(p,g),isSelected=String(p.id)===selected,radius=Math.max(3.2,Math.min(7,g.cell*.3));ctx.globalAlpha=selected!=='all'&&!isSelected?.36:ghost?.25:1;ctx.fillStyle=color;ctx.strokeStyle=ghost?color:'#fff';ctx.lineWidth=1.2;ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fill();ctx.stroke();if(!ghost){ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-Math.sin(p.yaw)*radius*.9,y-Math.cos(p.yaw)*radius*.9);ctx.stroke();}if(isSelected){ctx.globalAlpha=1;ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.beginPath();ctx.arc(x,y,radius+4,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#192f43';ctx.font='bold 12px system-ui';ctx.fillText(String(p.id),x+radius+6,y-4);}ctx.globalAlpha=1;}
function drawMap(canvas,kind){const ctx=canvas.getContext('2d'),g=geometry(canvas);background(ctx,canvas,g);trace(ctx,g,kind,kind==='actual'?'#2464c4':'#d96b26');if(kind!=='actual'){trace(ctx,g,'actual','#7fa7df');for(const p of frame().actual)robot(ctx,p,g,'#2464c4',true);if(selected!=='all'){const pair=pairs(frame(),kind).find(p=>p.id===selected);if(pair){const a=point(pair.actual,g),b=point(pair.predicted,g);ctx.strokeStyle='#ab762c';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);}}}for(const p of frame()[kind])robot(ctx,p,g,kind==='actual'?'#2464c4':'#d96b26');}
function drawChart(){const canvas=$('error-chart'),ctx=canvas.getContext('2d'),pad={left:52,right:18,top:20,bottom:32},w=canvas.width-pad.left-pad.right,h=canvas.height-pad.top-pad.bottom,frames=sample().frames,tMax=Math.max(.1,frames[frames.length-1].offset_s);const values={};let yMax=.1;for(const kind of ['joint','plan_only','hold']){values[kind]=frames.map(f=>mean(errorsAt(f,kind,selected)));for(const value of values[kind])if(value!==null)yMax=Math.max(yMax,value);}yMax*=1.1;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.font='11px system-ui';ctx.lineWidth=1;for(let i=0;i<=3;i++){const y=pad.top+h*(1-i/3);ctx.strokeStyle='#e4ebf0';ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+w,y);ctx.stroke();ctx.fillStyle='#65788a';ctx.textAlign='right';ctx.fillText((yMax*i/3).toFixed(1),pad.left-8,y+4);}for(const kind of ['hold','plan_only','joint']){ctx.strokeStyle=colors[kind];ctx.lineWidth=kind===method?2.5:1.5;ctx.globalAlpha=kind===method?1:.65;ctx.beginPath();let started=false;frames.forEach((f,i)=>{const value=values[kind][i];if(value===null){started=false;return;}const x=pad.left+w*f.offset_s/tMax,y=pad.top+h*(1-value/yMax);if(started)ctx.lineTo(x,y);else ctx.moveTo(x,y);started=true;});ctx.stroke();ctx.globalAlpha=1;}const x=pad.left+w*frame().offset_s/tMax;ctx.strokeStyle='#516477';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,pad.top+h);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#65788a';ctx.textAlign='left';ctx.fillText('0 s',pad.left,pad.top+h+21);ctx.textAlign='right';ctx.fillText('+'+tMax.toFixed(1)+' s',pad.left+w,pad.top+h+21);ctx.textAlign='left';}
function draw(){const fr=frame(),all=pairs(fr,method),visible=all.filter(p=>selected==='all'||p.id===selected);$('scrub').value=String(index);$('offset').textContent='+'+fr.offset_s.toFixed(1)+' s';$('absolute').textContent='simulator time '+(sample().origin_s+fr.offset_s).toFixed(1)+' s';$('mean-error').textContent=fixed(mean(visible.map(p=>p.error)));$('p95-error').textContent=fixed(percentile(all.map(p=>p.error),.95));$('robot-count').textContent=String(all.length);$('slow-count').textContent=method==='joint'?String(fr.joint.filter(p=>p.mode==='slow').length):'—';$('error-label').textContent=selected==='all'?'MEAN POSITION ERROR · ALL ROBOTS':'POSITION ERROR · ROBOT '+selected;$('forecast-title').textContent=names[method];$('plan-label').textContent='Plan '+displayValue(sample().plan_version)+' · frozen at '+sample().origin_s.toFixed(1)+' s';$('chart-title').textContent=selected==='all'?'Mean position error over the horizon':'Robot '+selected+' · position error over the horizon';$('context-robot').textContent=selected==='all'?'All robots':'Robot '+selected;const exhausted=sample().diagnostics?.plan_exhausted_at_s;if(exhausted){const count=Object.values(exhausted).filter(t=>t!==null&&t<fr.offset_s-1e-8).length;$('context-coverage').textContent=count+' / '+Object.keys(exhausted).length+' before this horizon'+(selected!=='all'&&exhausted[selected]!==undefined?' · robot '+selected+': '+(exhausted[selected]===null?'not exhausted':exhausted[selected].toFixed(1)+' s'):'');}else $('context-coverage').textContent='Not recorded';$('method-note').textContent={joint:'Joint rollout: advances the fleet together using accepted action dependencies and predicted safety interactions. It approximates the native controller.',plan_only:'Plan-only baseline: advances accepted actions without the joint safety response. This exposes how much the interaction model changes the forecast.',hold:'Hold baseline: keeps each robot at its observed origin position for the entire horizon. This provides a simple reference for stationary or slow traffic.'}[method];drawMap($('actual-canvas'),'actual');drawMap($('forecast-canvas'),method);drawChart();}
function pause(){playing=false;lastTime=null;$('play').textContent='Play';}
function setOrigin(value){pause();originIndex=Number(value);index=0;playElapsed=0;const s=sample();$('scrub').max=String(s.frames.length-1);$('robot').replaceChildren();addOption($('robot'),'all','All robots');const ids=[...new Set(s.frames[0].actual.map(p=>String(p.id)))].sort((a,b)=>Number(a)-Number(b));for(const id of ids)addOption($('robot'),id,'Robot '+id);if(!ids.includes(selected))selected='all';$('robot').value=selected;$('context-origin').textContent=s.origin_s.toFixed(1)+' s simulator time';$('context-plan').textContent=displayValue(s.plan_version);$('context-trigger').textContent={body_in_outer:'Body enters outer polygon',outer_overlap:'Outer polygons overlap'}[s.trigger]||displayValue(s.trigger);$('context-frames').textContent=s.frames.length+' retained frames · '+s.frames[s.frames.length-1].offset_s.toFixed(1)+' s horizon';draw();}
function tick(timestamp){if(playing){if(lastTime!==null){playElapsed+=(timestamp-lastTime)/1000*Number($('speed').value);while(index+1<sample().frames.length&&sample().frames[index+1].offset_s<=playElapsed+1e-9)index++;if(index===sample().frames.length-1)pause();draw();}if(playing)lastTime=timestamp;}requestAnimationFrame(tick);}
function pick(event,canvas,kind){const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height,g=geometry(canvas);let distance=18,chosen='all';for(const pose of frame()[kind]){const p=point(pose,g),d=Math.hypot(x-p[0],y-p[1]);if(d<distance){distance=d;chosen=String(pose.id);}}selected=chosen;$('robot').value=chosen;draw();}
$('title').textContent=data.title||'Frozen forecasts, measured outcomes';document.title=$('title').textContent;
if(typeof data.metrics?.summary==='string'){$('summary').textContent=data.metrics.summary;$('summary').classList.remove('hidden');}
else if(Array.isArray(data.metrics?.headline)&&data.metrics.headline.length){$('summary').textContent=data.metrics.headline.map(item=>displayValue(item.label)+': '+displayValue(item.value)).join(' · ');$('summary').classList.remove('hidden');}
if(Array.isArray(data.links)){for(const link of data.links){if(typeof link.label!=='string'||typeof link.href!=='string')continue;const href=link.href.trim();if(!href||/[\u0000-\u0020]/.test(href)||(/^[a-z][a-z0-9+.-]*:/i.test(href)&&!/^https?:\/\//i.test(href)))continue;const anchor=document.createElement('a');anchor.href=href;anchor.textContent=link.label;anchor.rel='noopener';$('resource-links').append(anchor);$('resource-links').classList.remove('hidden');}}
for(let i=0;i<data.samples.length;i++){const s=data.samples[i];addOption($('origin'),i,s.origin_s.toFixed(1)+' s · '+({body_in_outer:'body entry',outer_overlap:'polygon overlap'}[s.trigger]||displayValue(s.trigger))+(s.label?' · '+s.label:''));}
for(const el of document.querySelectorAll('.unit'))el.textContent=unit;$('chart-unit').textContent=unit;
$('origin').addEventListener('change',()=>setOrigin($('origin').value));$('method').addEventListener('change',()=>{method=$('method').value;draw();});$('robot').addEventListener('change',()=>{selected=$('robot').value;draw();});$('scrub').addEventListener('input',()=>{pause();index=Number($('scrub').value);playElapsed=frame().offset_s;draw();});$('reset').addEventListener('click',()=>{pause();index=0;playElapsed=0;draw();});$('play').addEventListener('click',()=>{if(playing){pause();return;}if(index===sample().frames.length-1)index=0;playElapsed=frame().offset_s;playing=true;lastTime=null;$('play').textContent='Pause';draw();});$('actual-canvas').addEventListener('click',event=>pick(event,$('actual-canvas'),'actual'));$('forecast-canvas').addEventListener('click',event=>pick(event,$('forecast-canvas'),method));
setOrigin(0);requestAnimationFrame(tick);
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(write_viewer(args.output_dir))


if __name__ == "__main__":
    main()
