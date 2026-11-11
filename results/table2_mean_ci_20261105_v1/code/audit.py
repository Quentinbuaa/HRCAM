import argparse
import json
from pathlib import Path
import statistics
import time
from common import METHODS,bind,sha,write_json,read_rows

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();bind()
import numpy as np
started=time.time();summary=json.loads((args.run/'summary.json').read_text());provenance=json.loads((args.run/'PROVENANCE.json').read_text())
previous=args.root/'runs/table2_revision_20261105_v1';old=json.loads((previous/'summary.json').read_text())
assert sha(previous/'summary.json')==provenance['previous_summary_sha256']
fixture=json.loads((previous/'code/BASELINE_TABLE2.json').read_text())
point_checks=0;mean_cis=0;median_cis=0;submitted_checks=0;max_error=0
for name in ('pets_v1','cub_v1'):
    info=provenance['datasets'][name];path=Path(info['score_file']);assert sha(path)==info['score_sha256']
    rows=read_rows(path);groups={}
    for row in rows:groups.setdefault(row['source_id'],[]).append(row)
    ids=sorted(groups);assert all(len(group)==3 for group in groups.values())
    rng=np.random.default_rng(20261011);draws={m:[] for m in METHODS}
    for repeat in range(1000):
        sample=[r for i in rng.integers(0,len(ids),len(ids)) for r in groups[ids[int(i)]]]
        a=[r for r in sample if not r['inconsistent']];b=[r for r in sample if r['inconsistent']]
        if not a or not b:continue
        for method in METHODS:draws[method].append(statistics.fmean(r[method] for r in b)-statistics.fmean(r[method] for r in a))
    for method in METHODS:
        a=[r[method] for r in rows if not r['inconsistent']];b=[r[method] for r in rows if r['inconsistent']]
        actual={'delta_median':statistics.median(b)-statistics.median(a),
            'delta_mean':statistics.fmean(b)-statistics.fmean(a),'delta_max':max(b)-max(a),'delta_min':min(b)-min(a)}
        entry=summary[name]['methods'][method]
        for key,value in actual.items():
            error=abs(value-entry[key]);max_error=max(max_error,error);assert error<2e-12;point_checks+=1
            if method in fixture['datasets'][name]:
                assert round(value,4)==fixture['datasets'][name][method][key];submitted_checks+=1
        assert len(draws[method])==summary[name]['bootstrap_resamples']
        error=float(np.max(np.abs(np.quantile(draws[method],[.025,.975])-entry['delta_mean_ci95'])))
        max_error=max(max_error,error);assert error<2e-12;mean_cis+=1
        assert entry['delta_median_ci95']==old[name]['methods'][method]['delta_median_ci95'];median_cis+=1
    assert sha(path)==info['score_sha256']
    print(name,'independent mean CIs and all descriptive entries verified',flush=True)
assert (point_checks,mean_cis,median_cis,submitted_checks)==(104,26,26,80)
write_json(args.run/'INDEPENDENT_AUDIT.json',{'state':'passed','point_entries':point_checks,
    'new_mean_intervals':mean_cis,'exactly_reused_audited_median_intervals':median_cis,
    'submitted_descriptive_values_reproduced':submitted_checks,'max_absolute_error':max_error,
    'seconds':time.time()-started,'method':'Materialized source draws, statistics.fmean/median and direct extrema'})
