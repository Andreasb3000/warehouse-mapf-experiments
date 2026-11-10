"""Shared causal observation/target adapter for all report predictors.

Units: each raster entry is a robot count, or its time average. Low-speed
occupancy is an operational proxy, not excess traversal time or queue length.
"""
from pathlib import Path
import sys
import numpy as np

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parent / 'simulator'
sys.path.insert(0, str(REPO))
from experiments.prediction.flow_metrics import approaches_for, zones_for

DT = .1
WINDOWS = ((0., 5.), (5., 10.), (10., 20.))
HISTORY_OFFSETS = (-10, -8, -6, -4, -2, 0)


def observation_clock(event):
    """Return the physics/pose clock, preserving native controller poll phase.

    Archived passive snapshots use this clock at the top level. Live bridge
    requests instead use the server clock for file handshakes and expose a
    separate observation_clock. Physics advances before the controller hook,
    so the physics observation is one tick ahead of the server's control step.
    Never use the handshake clock to index visualizer pose history.
    """
    clock = event.get('observation_clock', event['pose_frame']['clock'])
    if type(clock) is not int or clock < 0:
        raise ValueError('Observation clock must be a nonnegative integer tick')
    if event['pose_frame']['clock'] != clock or event['snapshot']['clock'] != clock:
        raise ValueError('Pose and accepted-plan observation clocks differ')
    if any(controller['clock'] != clock for controller in event.get('controllers', [])):
        raise ValueError('Controller observation clock differs from pose clock')
    if 'observation_clock' in event:
        if event['snapshot'].get('server_clock', event['clock']) != event['clock']:
            raise ValueError('Live handshake clock differs from server control clock')
    elif event['clock'] != clock:
        raise ValueError('Passive snapshot top-level clock differs from pose clock')
    return clock


def pose_array(frame):
    return np.array([[-a['x'], -a['y'], a['rz']] for a in sorted(frame['agents'], key=lambda a:a['id'])], dtype=np.float64)


def rollout_array(result):
    times = np.array([f['offset_s'] for f in result['frames']])
    states = np.array([[[a['row'], a['col'], a['yaw']] for a in sorted(f['agents'], key=lambda a:a['id'])]
                       for f in result['frames']], dtype=np.float64)
    return times, states


def interpolate(times, states, desired):
    """Resample modeled trajectories only; native truth always remains at 0.1 s."""
    result = np.empty((len(desired), states.shape[1], 3))
    for i in range(states.shape[1]):
        for k in range(3):
            values = np.unwrap(states[:,i,k]) if k==2 else states[:,i,k]
            result[:,i,k] = np.interp(desired, times, values)
    return result


def masks(scenario, map_doc):
    layout = map_doc['layout']; h,w=len(layout),len(layout[0])
    road = np.array([[v not in '@T' for v in row] for row in layout])
    service = np.zeros((h,w), dtype=np.float32)
    for agent in scenario['agents']:
        for row,col in agent['goals']: service[round(row), round(col)] = 1
    roi = np.zeros((h,w), dtype=bool)
    rr,cc = np.indices((h,w))
    for zone in zones_for(scenario):
        r,c=zone['conflict_zone']['rows'],zone['conflict_zone']['cols']
        roi |= (rr>=r[0])&(rr<=r[1])&(cc>=c[0])&(cc<=c[1])
    for app in approaches_for(scenario,layout):
        start,end=app['upstream_exit_row_col'],app['downstream_entry_row_col']
        along = rr if app['axis']==0 else cc; across = cc if app['axis']==0 else rr
        low,high=sorted((start[app['axis']],end[app['axis']]))
        roi |= (along>=low)&(along<high)&(across>=app['transverse_min_m'])&(across<app['transverse_max_m'])
    return road, roi&road, service


def rasterize(states, scenario, shape, dt=DT):
    """Causal two-second history detector; first two seconds are unavailable.

    Each robot contributes at its nearest cell center. Boundary ties use floor
    (position+0.5). No obstacle mask is applied to predictions: off-road mass is
    retained so the evaluator can report it, rather than silently repair it.
    """
    states=np.asarray(states); n,agents,_=states.shape; h,w=shape
    history=round(2/dt)
    distance = np.concatenate([np.zeros((1,agents)),np.cumsum(np.linalg.norm(np.diff(states[:,:,:2],axis=0),axis=2),axis=0)])
    yaw=np.unwrap(states[:,:,2],axis=0)
    goals=np.unique(np.array([g for a in scenario['agents'] for g in a['goals']],dtype=float),axis=0)
    result=np.zeros((n,2,h,w),dtype=np.float32)
    eligible=np.zeros((n,agents),dtype=bool)
    for t in range(n):
        rc=np.floor(states[t,:,:2]+.5).astype(int)
        valid=(rc[:,0]>=0)&(rc[:,0]<h)&(rc[:,1]>=0)&(rc[:,1]<w)
        if not valid.all(): raise ValueError('Robot outside raster domain')
        np.add.at(result[t,0], (rc[:,0],rc[:,1]), 1)
        if t<history: continue
        speed=(distance[t]-distance[t-history])/2
        angular=np.max(np.abs(yaw[t-history:t+1]-yaw[t-history]),axis=0)
        at_service=np.any(np.linalg.norm(states[t,:,None,:2]-goals[None,:,:],axis=-1)<=.25,axis=1)
        slow=(speed<=.3+1e-8)&(angular<=.1)&~at_service
        eligible[t]=slow
        np.add.at(result[t,1],(rc[slow,0],rc[slow,1]),1)
    return result,eligible


def window_mean(future_rasters, sample_dt=DT):
    return np.stack([future_rasters[round(a/sample_dt):round(b/sample_dt)].mean(axis=0) for a,b in WINDOWS])


def predicted_rasters(result, past_states, scenario, shape):
    times,states=rollout_array(result)
    states=interpolate(times,states,np.arange(201)*DT)
    combined=np.concatenate([past_states[-21:-1],states])
    rasters,_=rasterize(combined,scenario,shape)
    return rasters[20:],states


def truth_targets(states, clock, scenario, shape):
    """Offline label API; callers invoke only after origin features are frozen."""
    segment=states[clock-20:clock+201]
    rasters,_=rasterize(segment,scenario,shape)
    return window_mean(rasters[20:]), segment[20:]
