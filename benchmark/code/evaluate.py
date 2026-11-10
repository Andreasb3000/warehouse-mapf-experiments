#!/usr/bin/env python3
"""One shared score definition for trajectories and direct raster forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np
from features import REPORT,WINDOWS


def measure(pred,truth,road,roi):
    rows=[]
    for k,(a,b) in enumerate(WINDOWS):
        channels={}
        for c,name in enumerate(('occupancy','slow_occupancy')):
            p=pred[:,k,c].astype(np.float64);t=truth[:,k,c].astype(np.float64);d=p-t
            regions={}
            for region,mask in [('roads',np.broadcast_to(road,t.shape)),('roi',np.broadcast_to(roi,t.shape)),
                                ('positive',np.broadcast_to(road,t.shape)&(t>0)),
                                ('positive_roi',np.broadcast_to(roi,t.shape)&(t>0))]:
                errors=d[mask]
                regions[region]={'cells':int(mask.sum()),'mae':float(np.abs(errors).mean()) if len(errors) else None,
                                 'rmse':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                                 'p95_absolute':float(np.quantile(np.abs(errors),.95)) if len(errors) else None,
                                 'bias':float(errors.mean()) if len(errors) else None}
            totals=(d[:,road].sum(axis=1))*(b-a)
            channels[name]={'regions':regions,'total_robot_seconds_mae':float(np.abs(totals).mean()),
                            'actual_mean_total_robot_seconds':float(t[:,road].sum(axis=1).mean()*(b-a)),
                            'offroad_predicted_robot_seconds':float(p[:,~road].sum(axis=1).mean()*(b-a))}
        rows.append({'window_s':[a,b],'channels':channels})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=REPORT/'data/dataset.npz')
    p.add_argument('--index',type=Path,default=REPORT/'data/dataset_index.json');p.add_argument('--output',type=Path,default=REPORT/'results/evaluation.json')
    args=p.parse_args();ds=np.load(args.data);meta=json.loads(args.index.read_text())['samples'];target=ds['target']
    test=np.array([i for i,m in enumerate(meta) if m['split']=='test'])
    road,roi=ds['road'],ds['roi'];methods={}
    with np.load(REPORT/'data/baseline_predictions.npz') as b:
        for method in b.files: methods[method]=b[method][test]
    local=REPORT/'results/local/predictions.npy'
    if local.exists():methods['local_ridge']=np.load(local,mmap_mode='r')[test]
    for name in ('convlstm','residual_convlstm'):
        file=REPORT/f'results/learned/{name}/predictions.npy'
        if file.exists():methods[name]=np.load(file,mmap_mode='r')[test]
    truth=target[test];test_meta=[meta[i] for i in test]
    # Realized slow occupancy is used for evaluation cohorts only.
    result={'windows_s':WINDOWS,'test_origins':len(test),'test_workloads':sorted({m['workload_seed'] for m in test_meta}),
            'road_cells':int(road.sum()),'roi_cells':int(roi.sum()),'units':'mean robots per cell; multiplying by window duration gives robot-seconds',
            'methods':{},'per_run':{},'interpretation':'Same held-out origins/labels for all methods. Positive cells are selected using truth for scoring only. Windows overlap; independent split units are whole workloads. Workload counts are reported explicitly; this is not a broad generalization study.'}
    for method,pred in methods.items():
        assert pred.shape==truth.shape,(method,pred.shape,truth.shape)
        assert np.isfinite(pred).all() and (pred>=-1e-5).all()
        result['methods'][method]=measure(pred,truth,road,roi)
        result['per_run'][method]={}
        for run in sorted({m['run_id'] for m in test_meta}):
            ids=np.array([i for i,m in enumerate(test_meta) if m['run_id']==run])
            result['per_run'][method][run]=measure(pred[ids],truth[ids],road,roi)
    # Development-set rankings do not enter checkpoint selection here.
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(REPORT/'results/test_predictions.npz',truth=truth,**methods,indices=test,road=road,roi=roi)
    print(json.dumps({'test_origins':len(test),'methods':list(methods),'output':str(args.output)},indent=2))


if __name__=='__main__':main()
