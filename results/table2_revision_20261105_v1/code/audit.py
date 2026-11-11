"""Check point statistics and bootstrap medians independently of weighted medians."""
import argparse
import json
from pathlib import Path
import statistics
import time
from common import METHODS,bind,sha,write_json,read_rows,independent_rank


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind()
    import numpy as np
    started=time.time();summary=json.loads((args.run/'summary.json').read_text())
    provenance=json.loads((args.run/'PROVENANCE.json').read_text());source=args.root/'runs/table3_revision_20261105_v1'
    ranking=json.loads((source/'summary.json').read_text());points=0;median_cis=0;rank_cis=0;max_error=0
    for name in ('pets_v1','cub_v1'):
        path=source/(name+'_scores.jsonl');assert sha(path)==provenance[name]['score_sha256']
        assert sha(source/'summary.json')==provenance[name]['Table3_summary_sha256']
        rows=read_rows(path);groups={}
        for row in rows:groups.setdefault(row['source_id'],[]).append(row)
        ids=sorted(groups);rng=np.random.default_rng(20261011)
        bootstraps={m:[] for m in METHODS}
        for repeat in range(1000):
            materialized=[row for i in rng.integers(0,len(ids),len(ids)) for row in groups[ids[int(i)]]]
            negative=[r for r in materialized if not r['inconsistent']];positive=[r for r in materialized if r['inconsistent']]
            if not positive or not negative:continue
            for method in METHODS:
                bootstraps[method].append(statistics.median(r[method] for r in positive)-statistics.median(r[method] for r in negative))
        for method in METHODS:
            a=[r[method] for r in rows if not r['inconsistent']];b=[r[method] for r in rows if r['inconsistent']]
            actual={'delta_median':statistics.median(b)-statistics.median(a),
                'delta_mean':statistics.fmean(b)-statistics.fmean(a),'delta_max':max(b)-max(a),'delta_min':min(b)-min(a)}
            actual.update({k:v for k,v in independent_rank(rows,method).items() if k in ('U','A_U')})
            saved=summary[name]['methods'][method]
            for key,value in actual.items():
                error=abs(value-saved[key]);max_error=max(max_error,error)
                assert error<2e-12,(name,method,key,error);points+=1
            assert len(bootstraps[method])==summary[name]['bootstrap_resamples']
            expected=np.quantile(bootstraps[method],[.025,.975])
            error=float(np.max(np.abs(expected-saved['delta_median_ci95'])))
            max_error=max(max_error,error);assert error<2e-12,(name,method,error);median_cis+=1
            assert saved['A_U_ci95']==ranking[name]['ranking'][method]['auroc_ci95'];rank_cis+=1
        print(name,'independent point/rank/median-CI checks passed',flush=True)
    assert (points,median_cis,rank_cis)==(156,26,26)
    write_json(args.run/'INDEPENDENT_AUDIT.json',{'state':'passed','point_entries':points,
        'new_median_intervals':median_cis,'exactly_reused_audited_AUROC_intervals':rank_cis,
        'max_absolute_error':max_error,'seconds':time.time()-started,
        'method':'Python statistics.median/fmean, materialized source-cluster draws and integer rank counts'})


if __name__=='__main__':main()
