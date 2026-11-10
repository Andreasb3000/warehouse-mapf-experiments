#!/usr/bin/env python3
"""Compare temporal observation/projection sampling without changing native physics."""
import json
import time
from pathlib import Path
import numpy as np
from features import REPORT,DT,WINDOWS,pose_array,rasterize,window_mean,predicted_rasters,masks,observation_clock
from experiments.prediction.joint_evaluate import read_run
from experiments.prediction.joint import rollout
from runtime import evaluation_origins


def main():
    manifest=json.loads((REPORT/'data/manifest.json').read_text())['runs'];rows=[]
    for entry in manifest:
        if entry['split']!='test':continue
        config,scenario,map_doc,snaps,frames,_=read_run(Path(entry['run_directory']))
        shape=(len(map_doc['layout']),len(map_doc['layout'][0]));road,roi,_=masks(scenario,map_doc)
        lookup={observation_clock(e):e for e in snaps}
        for clock in evaluation_origins(lookup, frames):
            event=lookup[clock];past=np.array([pose_array(frames[k]) for k in range(clock-20,clock+1)])
            past[-1]=pose_array(event['pose_frame'])
            truth=np.array([pose_array(frames[k]) for k in range(clock-20,clock+201)])
            truth[20]=pose_array(event['pose_frame'])
            rasters,_=rasterize(truth,scenario,shape);reference=window_mean(rasters[20:])
            for step in (.1,.5,1.):
                stride=round(step/DT)
                sampled,_=rasterize(truth[::stride],scenario,shape,dt=step)
                coarse=window_mean(sampled[round(2/step):],sample_dt=step)
                t=time.perf_counter()
                out=rollout(event['snapshot'],event['pose_frame'],scenario,map_doc,controllers=None,safety_enabled=False,dt=step,horizon_s=20)
                core=time.perf_counter()-t
                rr,_=predicted_rasters(out,past,scenario,shape);pred=window_mean(rr)
                rows.append({'run_id':entry['run_id'],'origin_s':clock/10,'sampling_s':step,
                             'truth_downsampling_roi_mae':np.abs(coarse-reference)[:,:,roi].mean(axis=(0,2)).tolist(),
                             'plan_projection_roi_rmse':np.sqrt(((pred-reference)**2)[:,:,roi].mean(axis=(0,2))).tolist(),
                             'plan_projection_core_s':core})
    result={'rows':rows,'meaning':'Sampling sensitivity on native held-out trajectories plus constant-speed plan projection dt. Native and controller-aware joint integrator always0.1s. These are not retrained learned-model comparisons.'}
    (REPORT/'results/discretization.json').write_text(json.dumps(result,indent=2))
    print('Discretization comparisons',len(rows))


if __name__=='__main__':main()
