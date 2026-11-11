import argparse
import copy
import json
import os
from pathlib import Path
import time
from common import METHODS,bind,sha,write_json,read_rows

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();bind()
import numpy as np
started=time.time();previous=args.root/'runs/table2_revision_20261105_v1';scores=args.root/'runs/table3_revision_20261105_v1'
assert json.loads((previous/'RUN_STATUS.json').read_text())['state']=='complete'
assert json.loads((previous/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
old=json.loads((previous/'summary.json').read_text());old_provenance=json.loads((previous/'PROVENANCE.json').read_text())
result={};provenance={'previous_summary_sha256':sha(previous/'summary.json'),'datasets':{}}
for name in ('pets_v1','cub_v1'):
    path=scores/(name+'_scores.jsonl');before=sha(path);assert before==old_provenance[name]['score_sha256']
    rows=read_rows(path);y=np.array([r['inconsistent'] for r in rows],bool)
    ids=np.array([r['source_id'] for r in rows]);unique,index=np.unique(ids,return_inverse=True)
    assert np.all(np.bincount(index)==3)
    rng=np.random.default_rng(20261011)
    counts=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(1000)])
    weights=counts[:,index];w0=weights[:,~y];w1=weights[:,y];n0=w0.sum(1);n1=w1.sum(1)
    valid=(n0>0)&(n1>0);assert int(valid.sum())==old[name]['bootstrap_resamples']
    out=copy.deepcopy({k:v for k,v in old[name].items() if k!='methods'});out['methods']={}
    for method in METHODS:
        values=np.array([r[method] for r in rows],float)
        draws=(w1@values[y])/n1-(w0@values[~y])/n0
        entry={k:old[name]['methods'][method][k] for k in ('delta_median','delta_mean','delta_max','delta_min','delta_median_ci95')}
        entry['delta_mean_ci95']=np.quantile(draws[valid],[.025,.975]).tolist()
        out['methods'][method]=entry
    result[name]=out;assert sha(path)==before
    provenance['datasets'][name]={'score_file':str(path),'score_sha256':before}
    print(name,'mean-gap intervals calculated; checked median intervals retained',flush=True)
write_json(args.run/'summary.json',result);write_json(args.run/'PROVENANCE.json',provenance)
write_json(args.run/'CALCULATION_STATUS.json',{'state':'complete','seconds':time.time()-started,
    'new_mean_intervals':26,'reused_median_intervals':26,'cpu_affinity':sorted(os.sched_getaffinity(0)),
    'numerical_threads':4,'new_model_inference':False})
