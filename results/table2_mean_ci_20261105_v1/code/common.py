import hashlib
import json
import os
from pathlib import Path

METHODS=('CD','KLD','SNO','NCD','JSD','BAT','SpatialW1','FAS','BAI','L1','L2','SSIM','H-RCAM')


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


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def weighted_median(values,weights):
    import numpy as np
    order=np.argsort(values,kind='stable');v=values[order]
    cumulative=np.cumsum(weights[:,order],axis=1);total=cumulative[:,-1]
    lower=(total+1)//2;upper=(total+2)//2
    first=np.argmax(cumulative>=lower[:,None],axis=1)
    second=np.argmax(cumulative>=upper[:,None],axis=1)
    out=(v[first]+v[second])/2
    return np.where(total>0,out,np.nan)


def independent_rank(rows,method):
    # Integer half-win counts avoid deriving U from rounded AUROC values.
    a=sorted([(float(r[method]),bool(r['inconsistent'])) for r in rows],reverse=True)
    n1=sum(y for _,y in a);n0=len(a)-n1;higher_positive=0;twice_wins=0;i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[j][0]==a[i][0]:j+=1
        p=sum(y for _,y in a[i:j]);n=j-i-p
        twice_wins+=2*higher_positive*n+p*n;higher_positive+=p;i=j
    return {'A_U':twice_wins/(2*n0*n1),'U':(2*n0*n1-twice_wins)/2,'n0':n0,'n1':n1}
