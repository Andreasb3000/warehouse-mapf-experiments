#!/usr/bin/env python3
"""Paired native forecast-guided routing with a causal file handshake.

The predictor never receives future frames. Native PBS retains committed ADG
prefixes and applies nonnegative forecast penalties to its replacement suffixes.
All methods share one planner/search mode. Full measured barrier wall time is
charged to unfinished tasks by decision_metrics, as a synchronous deployment
model rather than pretending the native clock advanced while host code ran.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import uuid

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from features import REPORT,pose_array,masks,observation_clock
from record_dataset import make_workload,write_json,sha256
from decision_policy import METHODS,POLICY,forecast,cell_penalties
from decision_metrics import summarize_run
from experiments.baseline.analyze import _objects
from experiments.zones.generate import render_svg
from experiments.zones.run import docker_prefix,mounts,environment,flags
from experiments.junction.scenario import load_scenario
from runtime import validate_native_image, python_source_hashes


def inputs(seed,regime,trigger,horizon,image):
    map_doc,scenario,config,tasks,assignment=make_workload(seed,horizon)
    if regime=='within_zone':
        rng=random.Random(seed);destinations={}
        for zone in scenario['zones']:
            bays=deepcopy(zone['bays']['dropoffs']);rng.shuffle(bays);destinations[zone['id']]=iter(bays)
        allocation=[[0]*9 for _ in range(9)]
        zids={z['id']:i for i,z in enumerate(scenario['zones'])}
        for agent,task in zip(scenario['agents'],tasks['tasks']):
            zone=agent['zone_id'];cell=next(destinations[zone]);agent['goals'][1]=cell
            agent['destination_zone_id']=zone;task['destination_zone_id']=zone
            task['stages'][1].update(zone_id=zone,location=cell)
            allocation[zids[zone]][zids[zone]]+=1
        scenario['network']['allocation_matrix']=allocation;tasks['allocation_matrix']=allocation
        assignment={'workload_seed':seed,'distribution':'within-zone unique destination permutation','allocation_matrix':allocation}
    scenario['safety'].update(robot_trigger=trigger,inner_stop_enabled=False)
    config.update(image=image,wall_timeout_s=900)
    config['parameters'].update(sim_duration=round(horizon*10),sim_window_tick=20,single_agent_solver='ASTAR')
    return map_doc,scenario,config,tasks,assignment


class LiveHistory:
    """Read only complete JSON log lines; bound retained history to causal poses."""
    def __init__(self,path):self.path=path;self.offset=0;self.fragment='';self.frames={};self.goals=0
    def update(self):
        if not self.path.exists():return
        with self.path.open() as stream:
            stream.seek(self.offset);chunk=stream.read();self.offset=stream.tell()
        text=self.fragment+chunk;lines=text.split('\n');self.fragment=lines.pop()
        for event in _objects('\n'.join(lines)):
            if event.get('type')=='tick':self.frames[event['clock']]=pose_array(event)
            elif event.get('type')=='goal_reached':self.goals+=1
        if self.frames:
            cutoff=max(self.frames)-140
            self.frames={k:v for k,v in self.frames.items() if k>=cutoff}
    def past(self,event):
        clock=observation_clock(event);self.update()
        if any(k not in self.frames for k in range(clock-120,clock)):
            raise ValueError(f'Missing causal history before {clock}: {sorted(self.frames)[:3]}..{sorted(self.frames)[-3:]}')
        return np.stack([self.frames[k] for k in range(clock-120,clock)]+[pose_array(event['pose_frame'])])


def run_case(spec):
    directory=Path(spec['run_directory']);protocol=spec['protocol'];method=spec['method']
    completed=directory/'decision_metrics.json'
    if completed.exists():
        old=json.loads((directory/'run_spec.json').read_text())
        if old!=spec:raise ValueError(f'Existing run differs: {directory}')
        from decision_resume import validate_resume
        validate_resume(directory, spec, POLICY)
        return json.loads(completed.read_text())
    if directory.exists():raise ValueError(f'Incomplete run present; inspect before rerunning: {directory}')
    directory.mkdir(parents=True);bridge=directory/'bridge';bridge.mkdir()
    map_doc,scenario,config,tasks,assignment=inputs(spec['workload_seed'],spec['demand_regime'],spec['trigger'],protocol['horizon_s'],spec['image_id'])
    for name,value in [('config.json',config),('scenario.json',scenario),('map.json',map_doc),('tasks.json',tasks),('assignment.json',assignment),('run_spec.json',spec)]:write_json(directory/name,value)
    (directory/'layout.svg').write_text(render_svg(map_doc,scenario))
    load_scenario(directory/'scenario.json',map_doc['layout'],27)
    shape=(len(map_doc['layout']),len(map_doc['layout'][0]));road,_,service=masks(scenario,map_doc)
    static=np.stack([road,service,np.full(shape,spec['trigger']=='outer_overlap')]).astype(np.float32)
    model=joblib.load(REPORT/'results/local/model.joblib') if method=='local_ridge' else None
    params=dict(config['parameters'],container=True,visualizer='web',save_stats=True,scenario_file='/inputs/scenario.json',stats_name='/results/stats.json',argos_config_filepath='/results/simulation.argos',output_log='/results/simulator.log')
    docker=docker_prefix(spec['docker_host']);container='predictor-decision-'+uuid.uuid4().hex[:10]
    command=docker+['run','--rm','--init','--name',container,'--platform','linux/amd64','--ulimit','nofile=65536:65536',*mounts(directory),*environment(config),'-e','LSMART_ROUTING_DIR=/bridge','-e','LSMART_ROUTING_INTERVAL_TICKS=20','-v',f'{bridge}:/bridge','-v',f'{directory}:/results',spec['image_id'],'python3','run_lifelong.py','/inputs/map.json',*flags(params)]
    write_json(directory/'command.json',command)
    write_json(directory/'provenance.json',{'image_id':spec['image_id'],'policy':POLICY,
        'model_sha256':sha256(REPORT/'results/local/model.joblib') if model is not None else None,
        'dependency_scope':'fresh-source-v1', 'execution_source_sha256':python_source_hashes(),
        'host_code_sha256':{str(p.relative_to(REPORT)):sha256(p) for p in (Path(__file__),REPORT/'code/decision_policy.py',REPORT/'code/features.py',REPORT/'code/local_model.py')},
        'input_sha256':{f:sha256(directory/f) for f in ('map.json','scenario.json','tasks.json','config.json')}})
    print('START',spec['run_id'],flush=True);start=time.perf_counter();history=LiveHistory(directory/'simulator.log');queries=[];clock=0
    with (directory/'launcher.log').open('w') as launcher,threadpool_limits(limits=1):
        process=subprocess.Popen(command,stdout=launcher,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                request=bridge/f'request_{clock}.json'
                if not request.exists():
                    if time.perf_counter()-start>900:raise TimeoutError(spec['run_id'])
                    time.sleep(.002);continue
                event=json.loads(request.read_text());assert event['clock']==clock
                query_start=time.perf_counter();history.update()
                reason='forecast'
                if clock<120 or history.goals>=54:
                    penalty=np.zeros(shape);prediction=None
                    reason='history_warmup' if clock<120 else 'all_tasks_completed'
                else:
                    past=history.past(event)
                    prediction=forecast(event,past,scenario,map_doc,static,method,model)
                    penalty=cell_penalties(prediction,road)
                payload={'clock':clock,'cell_penalties':penalty.ravel().tolist()}
                write_json(bridge/f'response_{clock}.json',payload)
                row={'clock':clock,'observation_clock':event['observation_clock'],'origin_s':clock/10,'method':method,'host_query_s':time.perf_counter()-query_start,'reason':reason,'nonzero_penalty_cells':int(np.count_nonzero(penalty)),'penalty_max':float(penalty.max()),'slow_forecast_total':None if prediction is None else prediction[:,1].sum(axis=(1,2)).tolist()}
                queries.append(row)
                with (directory/'host_queries.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                clock+=20
            if process.returncode:raise RuntimeError(f'Native container failed {spec["run_id"]}: {process.returncode}')
        except BaseException:
            subprocess.run(docker+['stop','--time','2',container],capture_output=True)
            process.wait(timeout=30)
            raise
    events=list(_objects((directory/'simulator.log').read_text()))
    barriers={e['clock']:e for e in events if e.get('type')=='routing_barrier_complete'}
    plans={e['clock']:e for e in events if e.get('type')=='routing_plan'}
    expected=list(range(0,round(protocol['horizon_s']*10),20))
    if sorted(barriers)!=expected or [q['clock'] for q in queries]!=expected:raise ValueError(f'Incomplete barrier record: {spec["run_id"]}, {len(barriers)}, {len(queries)}')
    with (directory/'routing_decisions.jsonl').open('w') as stream:
        for row in queries:
            row['query_s']=barriers[row['clock']]['wall_s'];row['planner']=plans.get(row['clock'])
            stream.write(json.dumps(row)+'\n')
    metadata={k:v for k,v in spec.items() if k not in ('protocol','docker_host')}
    result=summarize_run(directory,metadata,protocol)
    result['recording_wall_s']=time.perf_counter()-start
    result['forecast_queries']=sum(q['reason']=='forecast' for q in queries)
    result['native_plan_failures']=sum(not p.get('success',False) for p in plans.values())
    write_json(completed,result)
    print('DONE',spec['run_id'],f'{result["recording_wall_s"]:.1f}s',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='warehouse-congestion:local');p.add_argument('--docker-host',default=None)
    p.add_argument('--jobs',type=int,default=1);p.add_argument('--pilot',action='store_true');p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));p.add_argument('--triggers',nargs='+',default=['body_in_outer','outer_overlap']);p.add_argument('--horizon-s',type=int)
    args=p.parse_args();protocol=json.loads((REPORT/'configs/decision_protocol.json').read_text())
    if not args.pilot and args.jobs!=1:raise ValueError('Formal timing comparisons require serial runs (--jobs 1)')
    if args.horizon_s:
        if not args.pilot:raise ValueError('Horizon overrides are pilot-only')
        protocol['horizon_s']=args.horizon_s
    docker=docker_prefix(args.docker_host)
    image=subprocess.check_output(docker+['image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    native_provenance=REPORT/'native/build_proof.json'
    validate_native_image(docker, image, native_provenance)
    root=REPORT/'data'/('decision_pilots' if args.pilot else 'decisions')
    workloads=[(s,'cross_zone') for s in (args.seeds or ([7901,7902] if args.pilot else protocol['primary_workload_seeds']))]
    if not args.pilot and not args.seeds:workloads += [(s,'within_zone') for s in protocol['control_workload_seeds']]
    specs=[]
    for seed,regime in workloads:
        for trigger in args.triggers:
            for method in args.methods:
                run_id=f'w{seed}_{regime}_{trigger}_{method}'
                specs.append(dict(run_id=run_id,workload_seed=seed,demand_regime=regime,trigger=trigger,method=method,run_directory=str(root/run_id),protocol=protocol,image_id=image,docker_host=args.docker_host))
    random.Random(20261031).shuffle(specs)
    manifest={'schema_version':1,'status':'running','pilot':args.pilot,'protocol':protocol,'policy':POLICY,'image_id':image,'native_provenance_sha256':sha256(native_provenance),'expected_runs':len(specs),'runs':[{k:v for k,v in s.items() if k not in ('protocol','docker_host')} for s in specs]}
    path=REPORT/'data'/('decision_pilot_manifest.json' if args.pilot else 'decision_manifest.json');write_json(path,manifest)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_case,s):s for s in specs}
        try:
            for future in as_completed(futures):future.result()
        except BaseException as e:
            manifest.update(status='failed',error=str(e));write_json(path,manifest)
            for f in futures:f.cancel()
            raise
    manifest['status']='complete';write_json(path,manifest)
    print('COMPLETE',len(specs),path,flush=True)


if __name__=='__main__':main()
