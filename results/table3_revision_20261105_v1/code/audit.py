"""Independent rank/count CI reconstruction and explicit SSIM-window audit."""
import argparse
import json
import math
from pathlib import Path
from common import METHODS,bind,write_json,read_rows,rank_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind()
    import numpy as np
    from scipy.signal import convolve2d
    summary=json.loads((args.run/'summary.json').read_text());result={'state':'passed','datasets':{},
        'point_entries_verified':0,'point_max_error':0.,'intervals_verified':0,'interval_endpoint_max_error':0.}
    for name in ('pets_v1','cub_v1'):
        rows=read_rows(args.run/(name+'_scores.jsonl'));reported=summary[name]
        for method in METHODS:
            actual=rank_metrics(rows,method)
            for metric,value in actual.items():
                error=abs(value-reported['ranking'][method][metric]);assert error<2e-12
                result['point_entries_verified']+=1;result['point_max_error']=max(result['point_max_error'],error)
        sources=sorted({r['source_id'] for r in rows});groups={s:[r for r in rows if r['source_id']==s] for s in sources}
        rng=np.random.default_rng(20261011);draws={m:{metric:[] for metric in ('auroc','ap')} for m in METHODS}
        for repeat in range(1000):
            selected=[r for i in rng.integers(0,len(sources),len(sources)) for r in groups[sources[int(i)]]]
            if len({r['inconsistent'] for r in selected})<2:continue
            for method in METHODS:
                metrics=rank_metrics(selected,method)
                for metric,value in metrics.items():draws[method][metric].append(value)
        for method in METHODS:
            for metric in ('auroc','ap'):
                pairs=[(np.quantile(draws[method][metric],[.025,.975]),reported['ranking'][method][metric+'_ci95'])]
                if method!='H-RCAM':
                    pairs.append((np.quantile(np.array(draws['H-RCAM'][metric])-draws[method][metric],[.025,.975]),
                        reported['paired_H_minus_baseline'][method][metric]['ci95']))
                for actual,expected in pairs:
                    error=float(np.max(np.abs(actual-expected)));assert error<2e-12
                    result['intervals_verified']+=1;result['interval_endpoint_max_error']=max(result['interval_endpoint_max_error'],error)
        print(name,'all point estimates and source-cluster intervals independently verified',flush=True)
        with np.load(args.root/'runs'/name/'holdout_heatmaps.npz') as z:
            ids,original,follow=z['source_ids'],z['original_maps'],z['maps']
        positions={int(s):i for i,s in enumerate(ids)};lookup={(r['source_id'],r['background']):r for r in rows}
        axis=np.arange(-5,6,dtype=float);g=np.exp(-axis*axis/(2*1.5**2));g/=g.sum();window=np.outer(g,g)
        checked=0;max_distance_error=0.
        for k in np.linspace(0,len(sources)-1,8,dtype=int):
            source=sources[int(k)];pos=positions[source];a=np.asarray(original[pos],float)
            for bg_index,bg in enumerate(('black','gray','white')):
                b=np.asarray(follow[pos,bg_index,0],float);row=lookup[(source,bg)]
                # Direct valid-window 2D convolution is independent of the
                # separable Gaussian-filter implementation used to calculate.
                ma=convolve2d(a,window,mode='valid');mb=convolve2d(b,window,mode='valid')
                va=convolve2d(a*a,window,mode='valid')-ma*ma
                vb=convolve2d(b*b,window,mode='valid')-mb*mb
                cov=convolve2d(a*b,window,mode='valid')-ma*mb
                similarity=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
                ssim_distance=float(1-similarity.mean())
                l1=math.fsum(abs(float(x)-float(y)) for x,y in zip(a.ravel(),b.ravel()))/a.size
                l2=math.dist(a.ravel(),b.ravel())/math.sqrt(a.size)
                error=max(abs(l1-row['L1']),abs(l2-row['L2']),abs(ssim_distance-row['SSIM']))
                assert error<2e-12;max_distance_error=max(max_distance_error,error);checked+=1
        result['datasets'][name]={'explicit_window_distance_checks':checked,'distance_max_error':max_distance_error,
            'source_clusters':len(sources),'pairs':len(rows),'successful_resamples':len(draws['H-RCAM']['auroc'])}
    assert result['point_entries_verified']==52 and result['intervals_verified']==100
    write_json(args.run/'INDEPENDENT_AUDIT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
