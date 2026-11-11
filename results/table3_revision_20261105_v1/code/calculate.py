"""Saved-map distances and source-cluster CIs for a standalone Table 3."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time
import os
from common import METHODS,bind,sha,write_json,read_rows,weighted_metrics

PUBLISHED={
 'JSD':[.8778,.6354,.7796,.8492], 'NCD':[.8816,.6777,.7939,.8586],
 'CD':[.7998,.5596,.7012,.8061], 'KLD':[.8149,.4453,.6568,.7411],
 'SNO':[.8529,.6281,.7603,.8311], 'BAT':[.7833,.5007,.6865,.7835],
 'SpatialW1':[.8482,.6119,.7469,.8312], 'FAS':[.6367,.3514,.5906,.7191],
 'BAI':[.5790,.2932,.5443,.6880], 'H-RCAM':[.8953,.6888,.8066,.8684]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind();sys.path.insert(0,str(args.root))
    import numpy as np
    import scipy
    import sklearn
    from scipy.ndimage import gaussian_filter
    from sklearn.metrics import roc_auc_score,average_precision_score
    from metrics import apply_minmax
    started=time.time();summary={};provenance={}
    for dataset_index,name in enumerate(('pets_v1','cub_v1')):
        folder=args.root/'runs'/name
        inputs={file:folder/file for file in ('holdout_records.jsonl','frozen_analysis.json','holdout_heatmaps.npz','classifier.pt')}
        inputs['transport']=args.root/'results/rq1_v3'/(name+'_holdout_transport_own.jsonl')
        before={key:sha(path) for key,path in inputs.items()}
        rows=sorted([r for r in read_rows(inputs['holdout_records.jsonl']) if r['cam_mode']=='own'
                    and r['mask_eligible'] and r['valid_cam']],key=lambda r:(r['source_id'],r['background']))
        assert len(rows)==(2979 if name=='pets_v1' else 2889)
        transport={(r['source_id'],r['background']):r for r in read_rows(inputs['transport'])}
        cfg=json.loads(inputs['frozen_analysis.json'].read_text())
        raw=np.array([[r[k] for k in ('d1','d2','d3','d4')] for r in rows]);norm,_=apply_minmax(raw,cfg['normalization']['own'])
        score_rows=[]
        for i,r in enumerate(rows):
            key=r['source_id'],r['background'];j,c=r['jsd'],r['d4']/2
            score_rows.append({'source_id':r['source_id'],'background':r['background'],'inconsistent':r['inconsistent'],
                'JSD':j,'NCD':c,'CD':float(norm[i,0]),'KLD':float(norm[i,1]),'SNO':float(norm[i,2]),
                'BAT':transport[key]['BAT'],'SpatialW1':transport[key]['spatial_w1'],
                'FAS':r['foreground_shift'],'BAI':r['background_increase'],'H-RCAM':2*j*c/(j+c+1e-12)})
        with np.load(inputs['holdout_heatmaps.npz']) as z:
            source_ids,original,follow=z['source_ids'],z['original_maps'],z['maps']
        positions={int(s):i for i,s in enumerate(source_ids)};keys={(r['source_id'],r['background']) for r in rows}
        def filters(a):
            return gaussian_filter(a,sigma=1.5,truncate=3.5,mode='reflect')
        def extra_distances(source):
            pos=positions[source];a=np.asarray(original[pos],np.float64);mu_a=filters(a);var_a=filters(a*a)-mu_a*mu_a
            out=[]
            for bg_index,bg in enumerate(('black','gray','white')):
                if (source,bg) not in keys:continue
                b=np.asarray(follow[pos,bg_index,0],np.float64)
                assert np.isfinite(a).all() and np.isfinite(b).all() and min(a.min(),b.min())>=0 and max(a.max(),b.max())<=1.000001
                difference=a-b;mu_b=filters(b);var_b=filters(b*b)-mu_b*mu_b;cov=filters(a*b)-mu_a*mu_b
                ssim=((2*mu_a*mu_b+.01**2)*(2*cov+.03**2))/((mu_a**2+mu_b**2+.01**2)*(var_a+var_b+.03**2))
                out.append(((source,bg),{'L1':float(np.abs(difference).mean()),'L2':float(np.sqrt(np.mean(difference*difference))),
                    'SSIM':float(1-ssim[5:-5,5:-5].mean())}))
            return out
        print(name,'computing L1/L2/SSIM from archived maps only',flush=True)
        extra={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for number,values in enumerate(pool.map(extra_distances,sorted({r['source_id'] for r in rows}))):
                extra.update(values)
                if number%300==0:print(name,'saved-map sources',number,flush=True)
        for row in score_rows:row.update(extra[(row['source_id'],row['background'])])
        (args.run/(name+'_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in score_rows))
        y=np.array([r['inconsistent'] for r in score_rows],int);source=np.array([r['source_id'] for r in score_rows])
        unique,index=np.unique(source,return_inverse=True);rng=np.random.default_rng(20261011)
        cluster_counts=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(1000)])
        pair_weights=cluster_counts[:,index]
        samples={};ranking={};maximum_sklearn_error=0.
        for method in METHODS:
            values=np.array([r[method] for r in score_rows],float)
            ranking[method]={'auroc':float(roc_auc_score(y,values)),'ap':float(average_precision_score(y,values))}
            if method in PUBLISHED:
                for metric_index,metric in enumerate(('auroc','ap')):
                    assert round(ranking[method][metric],4)==PUBLISHED[method][dataset_index*2+metric_index],(name,method,metric)
            valid,auc,ap=weighted_metrics(y,values,pair_weights);assert valid.sum()>=950
            samples[method]={'auroc':auc[valid],'ap':ap[valid]}
            for metric in ('auroc','ap'):
                ranking[method][metric+'_ci95']=np.quantile(samples[method][metric],[.025,.975]).tolist()
            # Meaningful numerical cross-check against actual repeated pair
            # rows, rather than comparing two weighted implementations.
            for repeat in (0,1,7,29,73):
                take=np.repeat(np.arange(len(y)),pair_weights[repeat])
                error=max(abs(float(roc_auc_score(y[take],values[take]))-auc[repeat]),
                          abs(float(average_precision_score(y[take],values[take]))-ap[repeat]))
                assert error<2e-12;maximum_sklearn_error=max(maximum_sklearn_error,error)
        contrasts={method:{metric:{'difference':ranking['H-RCAM'][metric]-ranking[method][metric],
            'ci95':np.quantile(samples['H-RCAM'][metric]-samples[method][metric],[.025,.975]).tolist()}
            for metric in ('auroc','ap')} for method in METHODS if method!='H-RCAM'}
        summary[name]={'pairs':len(rows),'sources':len(unique),'inconsistent_pairs':int(y.sum()),
            'prevalence':float(y.mean()),'ranking':ranking,'paired_H_minus_baseline':contrasts,
            'bootstrap_seed':20261011,'bootstrap_resamples':int(valid.sum()),'weighted_vs_replicated_sklearn_max_error':maximum_sklearn_error}
        after={key:sha(path) for key,path in inputs.items()};assert before==after
        provenance[name]={'input_paths':{key:str(path) for key,path in inputs.items()},'hashes_before':before,'hashes_after':after}
        del original,follow,pair_weights
        print(name,'all 13 methods and confidence intervals complete',flush=True)
    write_json(args.run/'summary.json',summary);write_json(args.run/'PROVENANCE.json',provenance)
    write_json(args.run/'CALCULATION_STATUS.json',{'state':'complete','seconds':time.time()-started,
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'numerical_threads':4,'ssim_workers':4,'new_model_inference':False})


if __name__=='__main__':main()
