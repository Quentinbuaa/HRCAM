import hashlib
import json
import os
from pathlib import Path

METHODS=('JSD','NCD','CD','KLD','SNO','BAT','SpatialW1','FAS','BAI','L1','L2','SSIM','H-RCAM')


def bind():
    cpus=set(os.sched_getaffinity(0))&set(range(24,32))
    if not cpus:raise RuntimeError('Approved CPUs unavailable')
    os.sched_setaffinity(0,cpus)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
    os.environ['CUDA_VISIBLE_DEVICES']=''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def weighted_metrics(y,scores,weights):
    """Exact metrics for replicated observations, grouping all score ties."""
    import numpy as np
    order=np.argsort(-scores,kind='stable');s=scores[order];yy=y[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    positives=np.add.reduceat(weights[:,order]*yy[None,:],starts,axis=1)
    negatives=np.add.reduceat(weights[:,order]*(1-yy)[None,:],starts,axis=1)
    tp=np.cumsum(positives,axis=1);fp=np.cumsum(negatives,axis=1)
    npos=tp[:,-1];nneg=fp[:,-1];valid=(npos>0)&(nneg>0)
    auroc=np.full(len(weights),np.nan);ap=np.full(len(weights),np.nan)
    wins=((tp-.5*positives)*negatives).sum(1)
    auroc[valid]=wins[valid]/(npos[valid]*nneg[valid])
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0)
    ap[valid]=(positives[valid]*precision[valid]).sum(1)/npos[valid]
    return valid,auroc,ap


def rank_metrics(rows,key):
    """Independent scalar rank/count implementation for audit."""
    a=sorted([(float(r[key]),bool(r['inconsistent'])) for r in rows],reverse=True)
    ptotal=sum(y for _,y in a);ntotal=len(a)-ptotal
    if not ptotal or not ntotal:return {'auroc':None,'ap':None}
    tp=fp=0;wins=ap=0.;i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[j][0]==a[i][0]:j+=1
        p=sum(y for _,y in a[i:j]);n=j-i-p
        wins+=tp*n+.5*p*n;tp+=p;fp+=n;ap+=p/ptotal*tp/(tp+fp);i=j
    return {'auroc':wins/(ptotal*ntotal),'ap':ap}


def weighted_median(values,weights):
    import numpy as np
    order=np.argsort(values,kind='stable');v=values[order]
    cumulative=np.cumsum(weights[:,order],axis=1);total=cumulative[:,-1]
    lower=(total+1)//2;upper=(total+2)//2
    first=np.argmax(cumulative>=lower[:,None],axis=1)
    second=np.argmax(cumulative>=upper[:,None],axis=1)
    out=(v[first]+v[second])/2
    return np.where(total>0,out,np.nan)

