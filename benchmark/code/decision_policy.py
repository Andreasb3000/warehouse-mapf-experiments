"""Frozen forecast-to-routing adapter, independent of native execution labels."""
import numpy as np
from scipy.ndimage import maximum_filter
from features import DT,HISTORY_OFFSETS,rasterize,window_mean,predicted_rasters
from local_model import predict_one
from experiments.prediction.joint import rollout

METHODS=('zero','persistence','kinematic','local_ridge','joint')
POLICY={'refresh_s':2.,'history_warmup_s':12.,'penalty_weight':8.,
        'window_weights':[.5,.3,.2],'spatial_max_filter_cells':3,
        'max_cell_penalty':1000.,
        'description':'Nonnegative static destination-cell penalty added to PBS space-time A* MOVE edges; WAIT retains its base cost. Window-weighted slow occupancy is expanded over a 3x3 neighborhood. Accepted committed prefixes remain native; all suffixes are replanned jointly every 2 s.'}


def forecast(event,past,scenario,map_doc,static,method,model=None):
    """Uses the same frozen forecast definitions as the offline comparison."""
    shape=static.shape[-2:]
    if method not in METHODS:raise ValueError(method)
    if method=='zero':return np.zeros((3,2,*shape),dtype=np.float32)
    if method=='persistence':
        hist,_=rasterize(past[-21:],scenario,shape)
        return np.broadcast_to(hist[-1],(3,*hist[-1].shape)).copy()
    if method=='joint':
        result=rollout(event['snapshot'],event['pose_frame'],scenario,map_doc,controllers=event['controllers'],dt=.1,horizon_s=20)
        rasters,_=predicted_rasters(result,past,scenario,shape)
        return window_mean(rasters)
    result=rollout(event['snapshot'],event['pose_frame'],scenario,map_doc,controllers=None,dt=.5,horizon_s=20,safety_enabled=False)
    rasters,_=predicted_rasters(result,past,scenario,shape);plan=window_mean(rasters)
    if method=='kinematic':return plan
    hist,_=rasterize(past,scenario,shape)
    history=np.stack([hist[120+round(offset/DT)] for offset in HISTORY_OFFSETS])
    plan[:,1]=0
    return predict_one(model,history,plan,static)


def cell_penalties(prediction,road,policy=POLICY):
    prediction=np.asarray(prediction)
    if prediction.shape!=(3,2,*road.shape) or not np.isfinite(prediction).all() or np.any(prediction<0):
        raise ValueError('Invalid forecast field')
    slow=np.tensordot(np.asarray(policy['window_weights']),prediction[:,1],axes=(0,0))
    expanded=maximum_filter(slow,size=policy['spatial_max_filter_cells'],mode='constant',cval=0)
    return np.minimum(policy['max_cell_penalty'],policy['penalty_weight']*expanded)*road
