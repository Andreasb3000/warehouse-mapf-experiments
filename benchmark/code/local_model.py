"""Cheap learned correction to planned occupancy using local traffic features."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from features import REPORT


def local_features(history,plan,static):
    """One feature row per cell; shared coefficients transfer across this floor.

    Only current/past rasters and origin-planned motion are accepted. Features
    contain no absolute time/robot/cell IDs, held-out labels, or future plans.
    """
    history=np.asarray(history,dtype=np.float32);plan=np.asarray(plan,dtype=np.float32)
    current=history[-1];mean=history.mean(axis=0);trend=current-history[-3]
    base=np.concatenate([current,mean,trend,plan[:,0],static],axis=0)
    nearby=np.concatenate([uniform_filter(base,size=(1,k,k),mode='constant') for k in (3,7)],axis=0)
    features=np.concatenate([base,nearby,current[0:1]*plan[:,0],current[1:2]*plan[:,0]],axis=0)
    return features.reshape(len(features),-1).T


def predict_one(model,history,plan,static):
    h,w=static.shape[-2:]
    correction=model.predict(local_features(history,plan,static)).T.reshape(3,2,h,w)
    result=correction+plan
    result[:,0]=np.maximum(0,result[:,0])
    result[:,1]=np.clip(result[:,1],0,result[:,0])
    result*=static[0]
    return result.astype(np.float32)


def validation_loss(pred,truth,road,roi):
    parts=[]
    for c,weight in enumerate((.3,.7)):
        error=(pred[:,c]-truth[:,c])**2
        positive=truth[:,c]>0
        value=.4*error[:,road].mean()+.3*error[:,roi].mean()
        value+=.3*error[positive].mean() if positive.any() else 0
        parts.append(weight*value)
    return float(sum(parts))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=REPORT/'data/dataset.npz')
    p.add_argument('--index',type=Path,default=REPORT/'data/dataset_index.json');p.add_argument('--output',type=Path,default=REPORT/'results/local')
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    ds=np.load(args.data);meta=json.loads(args.index.read_text())['samples']
    train=[i for i,m in enumerate(meta) if m['split']=='train'];val=[i for i,m in enumerate(meta) if m['split']=='validation']
    history,plan,static,target=[ds[k] for k in ('history','plan','static','target')]
    road,roi=ds['road'],ds['roi'];mask=road.ravel()
    X=[];Y=[];weights=[];started=time.perf_counter()
    for i in train:
        X.append(local_features(history[i],plan[i],static[i])[mask])
        Y.append((target[i].astype(np.float32)-plan[i]).reshape(6,-1).T[mask])
        positive=np.any(target[i,:,1]>0,axis=0).ravel()[mask]
        occupied=np.any(target[i,:,0]>0,axis=0).ravel()[mask]
        weights.append(1.+9*positive+2*occupied)
    X=np.concatenate(X).astype(np.float64);Y=np.concatenate(Y).astype(np.float64);weights=np.concatenate(weights)
    prep=time.perf_counter()-started
    candidates=[];best=None;best_loss=float('inf')
    with threadpool_limits(limits=2):
        for alpha in (.1,10.,1000.):
            t=time.perf_counter();model=Ridge(alpha=alpha,solver='cholesky').fit(X,Y,sample_weight=weights)
            loss=float(np.mean([validation_loss(predict_one(model,history[i],plan[i],static[i]),target[i].astype(np.float32),road,roi) for i in val]))
            candidates.append({'alpha':alpha,'validation_loss':loss,'fit_and_validation_s':time.perf_counter()-t})
            if loss<best_loss: best,best_loss=model,loss
    training_total=time.perf_counter()-started
    export_started=time.perf_counter()
    joblib.dump(best,args.output/'model.joblib')
    prediction=np.stack([predict_one(best,history[i],plan[i],static[i]) for i in range(len(meta))])
    np.save(args.output/'predictions.npy',prediction.astype(np.float16))
    report={'method':'local_ridge_residual','input_feature_count':int(X.shape[1]),'parameters':int(best.coef_.size+best.intercept_.size),
            'train_samples':len(train),'validation_samples':len(val),'selected_alpha':float(best.alpha),'candidates':candidates,
            'training_feature_preparation_s':prep,'training_total_s':training_total,
            'prediction_and_export_s':time.perf_counter()-export_started,'fit_dtype':'float64',
            'training_weights':'1 + 9 * any future slow occupancy + 2 * any future occupancy, applied only to training labels',
            'selection':'Validation only: .3 occupancy + .7 low-speed of (.4 road MSE + .3 ROI MSE + .3 positive-cell MSE)',
            'novelty':'Project-specific local spatial feature regression correction; not claimed new literature contribution'}
    (args.output/'training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
