import argparse
import json
from pathlib import Path
import time
import os
from common import METHODS,bind,sha,write_json,read_rows,weighted_median


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind()
    import numpy as np
    source=args.root/'runs/table3_revision_20261105_v1'
    assert json.loads((source/'RUN_STATUS.json').read_text())['state']=='complete'
    assert json.loads((source/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
    ranking=json.loads((source/'summary.json').read_text())
    fixture=json.loads(Path(__file__).with_name('BASELINE_TABLE2.json').read_text())
    started=time.time();summary={};provenance={};published_checks=0
    for name in ('pets_v1','cub_v1'):
        path=source/(name+'_scores.jsonl');before=sha(path);rows=read_rows(path)
        y=np.array([r['inconsistent'] for r in rows],bool);ids=np.array([r['source_id'] for r in rows])
        unique,index=np.unique(ids,return_inverse=True);rng=np.random.default_rng(20261011)
        counts=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(1000)])
        weights=counts[:,index];valid=(weights[:,y].sum(1)>0)&(weights[:,~y].sum(1)>0)
        assert valid.sum()==ranking[name]['bootstrap_resamples']
        output={}
        for method in METHODS:
            values=np.array([r[method] for r in rows],float);a,b=values[~y],values[y]
            delta=weighted_median(b,weights[:,y])-weighted_median(a,weights[:,~y])
            au=ranking[name]['ranking'][method]['auroc'];n0,n1=int((~y).sum()),int(y.sum())
            output[method]={'delta_median':float(np.median(b)-np.median(a)),
                'delta_mean':float(b.mean()-a.mean()),'delta_max':float(b.max()-a.max()),'delta_min':float(b.min()-a.min()),
                'U':round(2*n0*n1*(1-au))/2,'A_U':au,
                'delta_median_ci95':np.quantile(delta[valid],[.025,.975]).tolist(),
                'A_U_ci95':ranking[name]['ranking'][method]['auroc_ci95']}
            for repeat in (0,1,7,29,73):
                positive=np.repeat(b,weights[repeat,y]);negative=np.repeat(a,weights[repeat,~y])
                actual=float(np.median(positive)-np.median(negative))
                assert abs(actual-delta[repeat])<2e-12
            if method in fixture['datasets'][name]:
                old=fixture['datasets'][name][method]
                for statistic in ('delta_median','delta_mean','delta_max','delta_min','A_U'):
                    assert round(output[method][statistic],4)==old[statistic],(name,method,statistic)
                    published_checks+=1
                assert output[method]['U']==old['U'];published_checks+=1
        assert sha(path)==before
        summary[name]={'pairs':len(rows),'sources':len(unique),'consistent_pairs':int((~y).sum()),
            'inconsistent_pairs':int(y.sum()),'bootstrap_seed':20261011,'bootstrap_resamples':int(valid.sum()),'methods':output}
        provenance[name]={'score_file':str(path),'score_sha256':before,'Table3_summary_sha256':sha(source/'summary.json')}
        print(name,'all median differences and CIs complete',flush=True)
    assert published_checks==120
    write_json(args.run/'summary.json',summary);write_json(args.run/'PROVENANCE.json',provenance)
    write_json(args.run/'CALCULATION_STATUS.json',{'state':'complete','seconds':time.time()-started,
        'original_statistics_reproduced':published_checks,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'numerical_threads':4,'new_model_inference':False})


if __name__=='__main__':main()
