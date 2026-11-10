#!/usr/bin/env python3
"""Serial in-memory forecast queries, with feature construction and output decode.

This is NOT sensor-to-actuator latency: snapshot capture, RPC/network, disk I/O,
route optimization and simulator execution are excluded. No latency is inferred
by summing component percentiles. The measured query computes its own features.
"""
import argparse
from collections import defaultdict
import json
import platform
import subprocess
import time
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from features import REPORT,DT,HISTORY_OFFSETS,pose_array,masks,rasterize,window_mean,predicted_rasters,observation_clock
from local_model import predict_one
from learned import SpatialPredictor,synchronize
from experiments.prediction.joint_evaluate import read_run
from experiments.prediction.joint import rollout
from runtime import host_info, evaluation_origins


def summarize(values):
    return {'n':len(values),'median_s':float(np.median(values)),'p95_s':float(np.percentile(values,95)),
            'mean_s':float(np.mean(values)),'min_s':float(np.min(values)),'max_s':float(np.max(values))}


def prepare_queries():
    queries=[]
    for entry in json.loads((REPORT/'data/manifest.json').read_text())['runs']:
        if entry['split']!='test':continue
        _,scenario,map_doc,snapshots,frames,_=read_run(__import__('pathlib').Path(entry['run_directory']))
        shape=(len(map_doc['layout']),len(map_doc['layout'][0]));road,roi,service=masks(scenario,map_doc)
        static=np.stack([road,service,np.full(shape,entry['trigger']=='outer_overlap')]).astype(np.float32)
        lookup={observation_clock(e):e for e in snapshots}
        for clock in evaluation_origins(lookup, frames):
            past=np.stack([pose_array(frames[c]) for c in range(clock-120,clock+1)])
            past[-1]=pose_array(lookup[clock]['pose_frame'])
            queries.append({'run_id':entry['run_id'],'origin_s':clock/10,'past':past,
                            'event':lookup[clock],'scenario':scenario,'map_doc':map_doc,'shape':shape,'static':static})
    return queries


def query(q,method,model=None,device='cpu'):
    """Inputs are native structured snapshot + available pose history, not rasters."""
    event=q['event'];components={}
    if model is not None and isinstance(model,SpatialPredictor):synchronize(device)
    start=time.perf_counter()
    if method=='zero':
        result=np.zeros((3,2,*q['shape']),dtype=np.float32)
    elif method=='joint':
        t=time.perf_counter()
        out=rollout(event['snapshot'],event['pose_frame'],q['scenario'],q['map_doc'],
                    controllers=event['controllers'],dt=.1,horizon_s=20)
        components['core']=time.perf_counter()-t
        rr,_=predicted_rasters(out,q['past'],q['scenario'],q['shape']);result=window_mean(rr)
    elif method=='persistence':
        past,_=rasterize(q['past'][-21:],q['scenario'],q['shape'])
        result=np.broadcast_to(past[-1],(3,*past[-1].shape)).copy()
    elif method=='kinematic':
        out=rollout(event['snapshot'],event['pose_frame'],q['scenario'],q['map_doc'],
                    controllers=None,dt=.5,horizon_s=20,safety_enabled=False)
        rr,_=predicted_rasters(out,q['past'],q['scenario'],q['shape']);result=window_mean(rr)
    else:
        t=time.perf_counter()
        past,_=rasterize(q['past'],q['scenario'],q['shape'])
        history=np.stack([past[120+round(offset/DT)] for offset in HISTORY_OFFSETS])
        components['history_features']=time.perf_counter()-t
        if method=='persistence':result=np.broadcast_to(history[-1],(3,*history[-1].shape)).copy()
        else:
            t=time.perf_counter()
            out=rollout(event['snapshot'],event['pose_frame'],q['scenario'],q['map_doc'],
                        controllers=None,dt=.5,horizon_s=20,safety_enabled=False)
            rr,_=predicted_rasters(out,q['past'],q['scenario'],q['shape']);plan=window_mean(rr)
            components['plan_and_raster']=time.perf_counter()-t
            if method=='kinematic':result=plan
            else:
                plan[:,1]=0
                t=time.perf_counter()
                if method=='local_ridge':result=predict_one(model,history,plan,q['static'])
                else:
                    inputs=[torch.from_numpy(np.asarray(a,dtype=np.float32)[None]).to(device) for a in (history,plan,q['static'])]
                    with torch.inference_mode():result=model(*inputs).cpu().numpy()[0]
                    synchronize(device)
                components['model_with_transfer']=time.perf_counter()-t
    # Ensure an actual usable CPU result has been produced before stopping clock.
    checksum=float(np.asarray(result).sum())
    elapsed=time.perf_counter()-start
    return elapsed,components,checksum


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
    torch.set_num_threads(2);queries=prepare_queries()
    t=time.perf_counter();ridge=joblib.load(REPORT/'results/local/model.joblib');ridge_load=time.perf_counter()-t
    models={};loads={}
    for name in ('convlstm','residual_convlstm'):
        for device in ('cpu','mps') if torch.backends.mps.is_available() else ('cpu',):
            t=time.perf_counter();cp=torch.load(REPORT/f'results/learned/{name}/model.pt',map_location='cpu',weights_only=True)
            model=SpatialPredictor(hidden=cp['hidden'],residual=cp['residual']).to(device)
            model.load_state_dict(cp['state_dict']);model.eval();synchronize(device)
            models[(name,device)]=model;loads[f'{name}_{device}']=time.perf_counter()-t
    cases=[('zero','cpu',None),('persistence','cpu',None),('kinematic','cpu',None),('joint','cpu',None),('local_ridge','cpu',ridge)]
    cases += [(name,device,model) for (name,device),model in models.items()]
    outputs={}
    with threadpool_limits(limits=2):
        for method,device,model in cases:
            cold,_,_=query(queries[0],method,model,device)
            for i in range(5):query(queries[i%len(queries)],method,model,device)
            rows=[];parts=defaultdict(list)
            for repeat in range(args.repeats):
                for q in queries:
                    elapsed,components,checksum=query(q,method,model,device)
                    for k,v in components.items():parts[k].append(v)
                    rows.append({'run_id':q['run_id'],'origin_s':q['origin_s'],'repeat':repeat,'query_s':elapsed,'checksum':checksum})
            name=f'{method}_{device}' if model is not None and isinstance(model,SpatialPredictor) else method
            outputs[name]={'summary':summarize([r['query_s'] for r in rows]),'first_query_s':cold,'components':{k:summarize(v) for k,v in parts.items()},'samples':rows}
            print(json.dumps({name:outputs[name]['summary']}),flush=True)
    result={'hardware':host_info(),
            'python':platform.python_version(),'torch':torch.__version__,'threads':2,'query_count':len(queries),
            'load_s':{'local_ridge':ridge_load,**loads},'methods':outputs,
            'includes':'History rasterization when needed, current-plan projection when needed, inference, host-device transfers, and common output conversion. All methods return three2channel61x61 forecasts on host.',
            'excludes':'Simulator state capture, RPC/network, disk loading, routing/decision computation, training. This is in-memory forecast query time, not full deployed closed-loop latency.',
            'cadences_s':[.1,.5,1.,2.],
            'utilization_rule':'For one full-fleet query every Delta seconds: p95_query_s / Delta is a rough sequential budget fraction. It is not a throughput scaling law.'}
    (REPORT/'results/runtime.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
