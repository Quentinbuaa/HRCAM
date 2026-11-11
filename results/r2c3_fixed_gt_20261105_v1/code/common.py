"""Small shared helpers; bind before importing numerical libraries."""
import hashlib
import json
import os
from pathlib import Path


def bind():
    cpus=set(os.sched_getaffinity(0))&set(range(24,32))
    if not cpus:raise RuntimeError('Approved CPUs unavailable')
    os.sched_setaffinity(0,cpus)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='4'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def enrich(row):
    j=row['jsd'];c=row['d4']/2 if row['valid_cam'] else None
    return {**row,'ncd':c,'HRCAM':2*j*c/(j+c+1e-12) if row['valid_cam'] else None}


def comparisons(rows,modes=('own','fixed','ground_truth')):
    """Return common pair rows and a score matrix keyed by convention/component."""
    import numpy as np
    lookup={mode:{(r['source_id'],r['background']):r for r in rows
                  if r['cam_mode']==mode and r['mask_eligible'] and r['valid_cam']} for mode in modes}
    keys=sorted(set.intersection(*(set(lookup[mode]) for mode in modes)))
    reference=[lookup[modes[0]][key] for key in keys]
    scores={mode+'_'+method:np.array([lookup[mode][key][field] for key in keys],float)
            for mode in modes for method,field in [('HRCAM','HRCAM'),('JSD','jsd'),('NCD','ncd')]}
    for key in keys:
        assert len({lookup[mode][key]['inconsistent'] for mode in modes})==1
        assert len({lookup[mode][key]['label'] for mode in modes})==1
    return reference,scores,keys
