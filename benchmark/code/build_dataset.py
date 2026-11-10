#!/usr/bin/env python3
"""Extract fixed causal features and targets; independent recordings can parallelize."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from features import REPORT,REPO,DT,HISTORY_OFFSETS,WINDOWS,pose_array,masks,rasterize,window_mean,predicted_rasters,observation_clock
from experiments.prediction.joint_evaluate import read_run
from experiments.prediction.joint import rollout


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(entry):
    out=REPORT/'data/cache'/entry['run_id']; out.mkdir(exist_ok=True,parents=True)
    if (out/'done.json').exists():
        saved=json.loads((out/'done.json').read_text())
        if saved.get('adapter_version')==2:return saved
    started=time.perf_counter()
    config,scenario,map_doc,snapshots,frames,_=read_run(Path(entry['run_directory']))
    assert config['parameters']['ticks_per_second']==10
    shape=(len(map_doc['layout']),len(map_doc['layout'][0]))
    road,roi,service=masks(scenario,map_doc)
    # Pose archive is for labels/history only. Predictor API receives a bounded slice.
    max_clock=max(frames)
    first=min(frames)
    states=np.zeros((max_clock+1,27,3),dtype=float)
    for clock,frame in frames.items(): states[clock]=pose_array(frame)
    if first>0: states[:first]=states[first]
    data={k:[] for k in ('history','plan','static','target','joint','kinematic','persistence','zero')}
    sample_meta=[]; examples=[]
    for event in sorted(snapshots,key=observation_clock):
        # Archived exact snapshots and visualizer poses already share the
        # physics clock. The server's preceding control-step label is not used.
        clock=observation_clock(event)
        if clock<120 or clock%20 or clock+200>max_clock: continue
        if any(k not in frames for k in range(clock-120,clock+201)): continue
        t=time.perf_counter()
        # Copy only available history; future truth is not part of feature encoding.
        past_states=states[clock-120:clock+1].copy()
        past_states[-1]=pose_array(event['pose_frame'])
        past_raster,_=rasterize(past_states,scenario,shape)
        history=np.stack([past_raster[120+int(offset/DT)] for offset in HISTORY_OFFSETS])
        static=np.stack([road.astype(float),service,np.full(shape,entry['trigger']=='outer_overlap')]).astype(np.float32)
        history_time=time.perf_counter()-t
        kwargs=dict(horizon_s=20,speed_m_s=1,station_dwell_s=2)
        t=time.perf_counter()
        kin=rollout(event['snapshot'],event['pose_frame'],scenario,map_doc,dt=.5,controllers=None,safety_enabled=False,**kwargs)
        kin_core=time.perf_counter()-t
        t=time.perf_counter()
        kin_rasters,kin_states=predicted_rasters(kin,past_states,scenario,shape)
        kin_target=window_mean(kin_rasters)
        plan=kin_target.copy(); plan[:,1]=0
        kin_encode=time.perf_counter()-t
        t=time.perf_counter()
        joint=rollout(event['snapshot'],event['pose_frame'],scenario,map_doc,dt=.1,controllers=event['controllers'],**kwargs)
        joint_core=time.perf_counter()-t
        t=time.perf_counter()
        joint_rasters,joint_states=predicted_rasters(joint,past_states,scenario,shape)
        joint_target=window_mean(joint_rasters)
        joint_encode=time.perf_counter()-t
        persistence=np.broadcast_to(history[-1],joint_target.shape).copy()
        # All forecast construction above finishes before future labels are read.
        future_states=np.concatenate([past_states[-21:-1],states[clock:clock+201].copy()])
        future_states[20]=pose_array(event['pose_frame'])
        truth_rasters,_=rasterize(future_states,scenario,shape)
        target=window_mean(truth_rasters[20:])
        values=dict(history=history,plan=plan,static=static,target=target,joint=joint_target,
                    kinematic=kin_target,persistence=persistence,zero=np.zeros_like(target))
        for key,value in values.items(): data[key].append(value.astype(np.float16))
        active=np.linalg.norm(states[clock+100,:,:2]-states[clock,:,:2],axis=1)>.25
        positions={}
        for name,forecast in [('joint',joint_states),('kinematic',kin_states)]:
            positions[name]={str(h):{'all_mae_m':float(np.linalg.norm(forecast[int(h/DT),:,:2]-states[clock+int(h/DT),:,:2],axis=1).mean()),
                                        'errors_m':np.linalg.norm(forecast[int(h/DT),:,:2]-states[clock+int(h/DT),:,:2],axis=1).tolist()}
                             for h in (5,10,20)}
        sample_meta.append({**entry,'origin_s':clock*DT,'timing_s':{'history':history_time,'kinematic_core':kin_core,
            'kinematic_encode':kin_encode,'joint_core':joint_core,'joint_encode':joint_encode},
            'plan_exhausted_at_s':joint['diagnostics']['plan_exhausted_at_s'],'position':positions})
    if not sample_meta: raise ValueError('No eligible forecast origins')
    np.savez_compressed(out/'arrays.npz',**{k:np.stack(v) for k,v in data.items()},road=road,roi=roi)
    (out/'index.json').write_text(json.dumps({'samples':sample_meta},indent=2))
    metadata={**entry,'adapter_version':2,'samples':len(sample_meta),'extraction_wall_s':time.perf_counter()-started,
              'arrays_sha256':digest(out/'arrays.npz'),'inputs':{name:digest(Path(entry['run_directory'])/name)
                for name in ('scenario.json','config.json','map.json','simulator.log')}}
    (out/'done.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'complete':entry['run_id'],'samples':len(sample_meta),'seconds':metadata['extraction_wall_s']}),flush=True)
    return metadata


def combine(entries):
    arrays={};index=[];manifest=[];road=roi=None
    for entry in entries:
        out=REPORT/'data/cache'/entry['run_id']
        with np.load(out/'arrays.npz') as values:
            for k in values.files:
                if k in ('road','roi'): continue
                arrays.setdefault(k,[]).append(values[k])
            if road is None: road,roi=values['road'],values['roi']
            else: assert np.array_equal(road,values['road']) and np.array_equal(roi,values['roi'])
        index.extend(json.loads((out/'index.json').read_text())['samples'])
        manifest.append(json.loads((out/'done.json').read_text()))
    np.savez_compressed(REPORT/'data/dataset.npz',**{k:np.concatenate(v) for k,v in arrays.items() if k not in ('joint','kinematic','persistence','zero')},road=road,roi=roi)
    np.savez_compressed(REPORT/'data/baseline_predictions.npz',**{k:np.concatenate(v) for k,v in arrays.items() if k in ('joint','kinematic','persistence','zero')})
    (REPORT/'data/dataset_index.json').write_text(json.dumps({'samples':index,'runs':manifest,
            'dt_s':DT,'history_offsets_s':HISTORY_OFFSETS,'windows_s':WINDOWS,
            'definitions':'Mean occupancy and causal two-second low-speed occupancy; this is not physical free-flow delay.'},indent=2))
    print('Combined',len(index),'samples',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=REPORT/'data/manifest.json')
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--combine',action='store_true');args=p.parse_args()
    entries=json.loads(args.manifest.read_text())['runs']
    with ProcessPoolExecutor(max_workers=args.jobs) as pool: list(pool.map(extract,entries))
    if args.combine: combine(entries)
