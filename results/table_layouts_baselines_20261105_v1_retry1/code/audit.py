import argparse,json,math,time
from pathlib import Path
from common import METHODS,bind,sha,write_json,read_rows,rank_metrics
from cases import cases,load_maps

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();bind()
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import jensenshannon
summary=json.loads((args.run/'summary.json').read_text());provenance=json.loads((args.run/'PROVENANCE.json').read_text())
report={'state':'passed','point_entries':0,'new_intervals':0,'explicit_distance_checks':0,'max_error':0.,'cases':{}};started=time.time()
for case in cases(args.root):
    key=case['key'];rows=read_rows(args.run/(key+'_scores.jsonl'));saved=summary[key]
    for path,expected in provenance[key]['before'].items():assert sha(Path(path))==expected
    for method in METHODS:
        for metric,value in rank_metrics(rows,method).items():
            error=abs(value-saved['ranking'][method][metric]);assert error<2e-12;report['max_error']=max(report['max_error'],error);report['point_entries']+=1
    groups={}
    for row in rows:groups.setdefault(row['source_id'],[]).append(row)
    sources=sorted(groups);rng=np.random.default_rng(case['seed'])
    checked_methods=METHODS if case['architecture']=='ResNet-18' else ('L1','L2','SSIM','H-RCAM')
    draws={m:{metric:[] for metric in ('auroc','ap')} for m in checked_methods}
    for repeat in range(1000):
        selected=[r for i in rng.integers(0,len(sources),len(sources)) for r in groups[sources[int(i)]]]
        for method in checked_methods:
            for metric,value in rank_metrics(selected,method).items():draws[method][metric].append(value)
        if repeat%250==0:print(key,'independent bootstrap audit',repeat,flush=True)
    for method in checked_methods:
        if case['architecture']=='DenseNet-121' and method=='H-RCAM':continue
        for metric in ('auroc','ap'):
            checks=[(np.quantile(draws[method][metric],[.025,.975]),saved['ranking'][method][metric+'_ci95'])]
            if method!='H-RCAM':checks.append((np.quantile(np.array(draws['H-RCAM'][metric])-draws[method][metric],[.025,.975]),saved['paired_H_minus_baseline'][method][metric]['ci95']))
            for actual,expected in checks:
                error=float(np.max(np.abs(actual-expected)));assert error<2e-12;report['max_error']=max(report['max_error'],error);report['new_intervals']+=1
    positions,maps=load_maps(case);lookup={(r['source_id'],r['background']):r for r in rows}
    axis=np.arange(-5,6,dtype=float);g=np.exp(-axis*axis/(2*1.5**2));g/=g.sum();window=np.outer(g,g)
    maximum_distance_error=0.
    for k in np.linspace(0,len(sources)-1,8,dtype=int):
        source=sources[int(k)]
        for bg_index,bg in enumerate(('black','gray','white')):
            a,b=maps(positions[source],bg_index);a=np.asarray(a,float);b=np.asarray(b,float);row=lookup[(source,bg)]
            ma=convolve2d(a,window,mode='valid');mb=convolve2d(b,window,mode='valid')
            va=convolve2d(a*a,window,mode='valid')-ma*ma;vb=convolve2d(b*b,window,mode='valid')-mb*mb;cov=convolve2d(a*b,window,mode='valid')-ma*mb
            sim=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
            expected={'L1':math.fsum(abs(float(x)-float(y)) for x,y in zip(a.ravel(),b.ravel()))/a.size,
                'L2':math.dist(a.ravel(),b.ravel())/math.sqrt(a.size),'SSIM':float(1-sim.mean())}
            error=max(abs(expected[m]-row[m]) for m in expected);assert error<2e-12;maximum_distance_error=max(maximum_distance_error,error)
            ja=a.ravel()+1e-8;jb=b.ravel()+1e-8;js=float(jensenshannon(ja/ja.sum(),jb/jb.sum(),base=2)**2)
            nc=(1-float(np.corrcoef(a.ravel(),b.ravel())[0,1]))/2
            assert max(abs(js-row['JSD']),abs(nc-row['NCD']))<2e-10
            report['explicit_distance_checks']+=1
    report['cases'][key]={'pairs':len(rows),'sources':len(sources),'distance_max_error':maximum_distance_error}
    for path,expected in provenance[key]['after'].items():assert sha(Path(path))==expected
    del maps;print(key,'all independent checks passed',flush=True)
assert (report['point_entries'],report['new_intervals'],report['explicit_distance_checks'])==(260,272,240)
report['seconds']=time.time()-started;write_json(args.run/'INDEPENDENT_AUDIT.json',report)
