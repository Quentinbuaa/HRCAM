"""Saved-map baseline comparisons and source bootstrap for natural backgrounds."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

METHODS=['JSD','NCD','CD','KLD','SNO','BAT','SpatialW1','FAS','BAI','HRCAM']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''): h.update(block)
    return h.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def estimate(rows,scores,repeats=1000):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    y=np.array([r['inconsistent'] for r in rows],int)
    sources=np.array([r['source_id'] for r in rows])
    functions={'auroc':roc_auc_score,'ap':average_precision_score}
    coverage={'valid_pairs':len(rows),'valid_sources':len(np.unique(sources)),
              'inconsistent_pairs':int(y.sum()),'inconsistency_prevalence':float(y.mean())}
    if len(np.unique(y))<2:
        return {'coverage':coverage,'state':'single_outcome_class','ranking':{
            name:{'auroc':None,'ap':None} for name in scores}}
    ranking={name:{key:float(fn(y,s)) for key,fn in functions.items()} for name,s in scores.items()}
    groups=[np.flatnonzero(sources==s) for s in np.unique(sources)]
    rng=np.random.default_rng(20261104)
    draws={name:{key:[] for key in functions} for name in scores}
    gaps=[]
    for repeat in range(repeats):
        take=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[take]))<2: continue
        for name,s in scores.items():
            for key,fn in functions.items(): draws[name][key].append(float(fn(y[take],s[take])))
        if 'HRCAM' in scores:
            h=scores['HRCAM'][take]
            gaps.append(float(np.median(h[y[take]==1])-np.median(h[y[take]==0])))
    successful=len(next(iter(draws.values()))['auroc'])
    if successful<int(.95*repeats): raise RuntimeError('Insufficient bootstrap resamples')
    for name in scores:
        for key in functions:
            ranking[name][key+'_ci95']=np.quantile(draws[name][key],[.025,.975]).tolist()
    differences={}
    reference='HRCAM' if 'HRCAM' in scores else 'fixed_HRCAM'
    if reference in scores:
        for name in scores:
            if name==reference: continue
            differences[reference+'_minus_'+name]={key:{
                'difference':ranking[reference][key]-ranking[name][key],
                'ci95':np.quantile(np.array(draws[reference][key])-draws[name][key],[.025,.975]).tolist()}
                for key in functions}
    gap=None
    if 'HRCAM' in scores:
        h=scores['HRCAM']; gap={'difference':float(np.median(h[y==1])-np.median(h[y==0])),
            'ci95':np.quantile(gaps,[.025,.975]).tolist(),
            'consistent_median':float(np.median(h[y==0])),
            'inconsistent_median':float(np.median(h[y==1]))}
    return {'state':'complete','coverage':coverage,'ranking':ranking,
        'paired_differences':differences,'HRCAM_group_median_gap':gap,
        'bootstrap_resamples':successful,'bootstrap_seed':20261104}


def hscore(rows):
    import numpy as np
    j=np.array([r['jsd'] for r in rows]); c=np.array([r['d4']/2 for r in rows])
    return 2*j*c/(j+c+1e-12)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    cpus=set(os.sched_getaffinity(0))&set(range(24,32))
    if not cpus: raise RuntimeError('Approved CPUs unavailable')
    os.sched_setaffinity(0,cpus)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='4'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path.insert(0,str(args.root))
    import numpy as np
    import scipy,sklearn
    from metrics import heatmap_metrics,apply_minmax
    from transport_metrics import distances
    output=args.run/'analysis';output.mkdir(exist_ok=False)
    started=time.time(); all_results={};csv_rows=[]
    for name in ('pets_v1','cub_v1'):
        folder=args.run/name
        rows=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
        maps=np.load(folder/'heatmaps.npy',mmap_mode='r')
        with np.load(args.run/'prepared'/(name+'_sources.npz')) as z:
            masks,source_ids=z['masks'],z['source_ids']
        positions={int(s):i for i,s in enumerate(source_ids)}
        config=json.loads((args.root/'runs'/name/'frozen_analysis.json').read_text())
        own=sorted([r for r in rows if r['cam_mode']=='own' and r['valid_cam']],key=lambda r:r['source_id'])
        score_rows=[];clipped_counts=np.zeros(4,int)
        for number,row in enumerate(own):
            pos=positions[row['source_id']];first,second=maps[pos,0],maps[pos,1]
            basic=heatmap_metrics(first,second,masks[pos]);assert basic['valid_cam']
            np.testing.assert_allclose(basic['jsd'],row['jsd'],atol=2e-14,rtol=0)
            np.testing.assert_allclose(basic['d4']/2,row['ncd'],atol=2e-14,rtol=0)
            transport=distances(first,second,masks[pos],grid=14)
            normalized,clipped=apply_minmax(np.array([[basic[k] for k in ['d1','d2','d3','d4']]]),config['normalization']['own'])
            clipped_counts+=clipped[0].astype(int)
            j,c=row['jsd'],row['ncd']
            score_rows.append({'source_id':row['source_id'],'background':'natural','inconsistent':row['inconsistent'],
                'donor_row':row['donor_row'],'JSD':j,'NCD':c,'CD':float(normalized[0,0]),
                'KLD':float(normalized[0,1]),'SNO':float(normalized[0,2]),'BAT':transport['BAT'],
                'SpatialW1':transport['spatial_w1'],'FAS':basic['foreground_shift'],
                'BAI':basic['background_increase'],'HRCAM':2*j*c/(j+c+1e-12),
                'CD_raw':basic['d1'],'KLD_raw':basic['d2'],'SNO_raw':basic['d3']})
            if number%300==0: print(name,'baseline distances',number,'/',len(own),flush=True)
        (output/(name+'_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in score_rows))
        scores={method:np.array([r[method] for r in score_rows]) for method in METHODS}
        print(name,'bootstrap all nine baselines plus H-RCAM',flush=True)
        natural_own=estimate(score_rows,scores)
        raw_scores={k:np.array([r[k+'_raw'] for r in score_rows]) for k in ('CD','KLD','SNO')}
        from sklearn.metrics import roc_auc_score,average_precision_score
        y=np.array([r['inconsistent'] for r in score_rows],int)
        raw_ranking={method:{'auroc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s))} for method,s in raw_scores.items()}
        fixed=sorted([r for r in rows if r['cam_mode']=='fixed' and r['valid_cam']],key=lambda r:r['source_id'])
        natural_fixed=estimate(fixed,{'HRCAM':hscore(fixed),'JSD':np.array([r['jsd'] for r in fixed]),
                                     'NCD':np.array([r['ncd'] for r in fixed])})
        own_lookup={r['source_id']:r for r in own}; fixed_lookup={r['source_id']:r for r in fixed}
        shared=sorted(own_lookup.keys()&fixed_lookup.keys())
        own_matched=[own_lookup[s] for s in shared];fixed_matched=[fixed_lookup[s] for s in shared]
        target_comparison=estimate(own_matched,{'own_HRCAM':hscore(own_matched),'fixed_HRCAM':hscore(fixed_matched)})
        original=[json.loads(line) for line in (args.root/'runs'/name/'holdout_records.jsonl').read_text().splitlines()]
        context=[]
        for mode in ('own','fixed'):
            for bg in ('black','gray','white','pooled_constant'):
                subset=[r for r in original if r['cam_mode']==mode and r['mask_eligible'] and r['valid_cam']
                        and (bg=='pooled_constant' or r['background']==bg)]
                # Existing conditions are descriptive context; do not treat
                # independently computed estimates as paired background contrasts.
                h=hscore(subset); yy=np.array([r['inconsistent'] for r in subset],int)
                context.append({'cam_mode':mode,'background':bg,'n':len(subset),
                    'inconsistency_prevalence':float(yy.mean()),'auroc':float(roc_auc_score(yy,h)),
                    'ap':float(average_precision_score(yy,h))})
        source_rows={r['source_id']:r for r in rows}
        all_results[name]={'natural_own':natural_own,'natural_fixed':natural_fixed,
            'paired_target_comparison':target_comparison,'constant_background_context':context,
            'coverage':{mode:{'eligible_sources':len(source_rows),'valid_pairs':sum(r['cam_mode']==mode and r['valid_cam'] for r in rows),
                'invalid_maps':sum(r['cam_mode']==mode and not r['valid_cam'] for r in rows)} for mode in ('own','fixed')},
            'source_accuracy':float(np.mean([r['original_correct'] for r in source_rows.values()])),
            'transformed_accuracy':float(np.mean([r['transformed_correct'] for r in source_rows.values()])),
            'all_eligible_inconsistency_prevalence':float(np.mean([r['inconsistent'] for r in source_rows.values()])),
            'raw_unclipped_ranking':raw_ranking,'frozen_calibration_clip_counts':dict(zip(['CD','KLD','SNO','D4'],clipped_counts.tolist()))}
        for mode,result in [('own',natural_own),('fixed',natural_fixed)]:
            for method,entry in result['ranking'].items():
                csv_rows.append({'dataset':name,'target':mode,'background':'natural','method':method,
                    'n':result['coverage']['valid_pairs'],'auroc':entry['auroc'],'auroc_ci_low':entry['auroc_ci95'][0],
                    'auroc_ci_high':entry['auroc_ci95'][1],'ap':entry['ap'],'ap_ci_low':entry['ap_ci95'][0],'ap_ci_high':entry['ap_ci95'][1]})
    write_json(output/'summary.json',all_results)
    with (output/'table_with_ci.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    lines=['# RQ3.4: Natural-background replacement','',
        'Existing ResNet-18 models; original masks; one distinct photographic donor per source.','',
        '| Dataset | Target | Valid pairs | Inconsistency | H-RCAM AUROC (95% CI) | AP (95% CI) |',
        '|---|---|---:|---:|---:|---:|']
    for name,result in all_results.items():
        for mode in ('own','fixed'):
            entry=result['natural_'+mode];h=entry['ranking']['HRCAM']
            lines.append('| {} | {} | {} | {:.2%} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} [{:.4f}, {:.4f}] |'.format(
                name,mode,entry['coverage']['valid_pairs'],entry['coverage']['inconsistency_prevalence'],
                h['auroc'],*h['auroc_ci95'],h['ap'],*h['ap_ci95']))
    lines+=['','## All original baselines: own-class targets','',
        '| Method | Pet AUROC | Pet AP | CUB AUROC | CUB AP |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        vals=[all_results[name]['natural_own']['ranking'][method][key] for name in ('pets_v1','cub_v1') for key in ('auroc','ap')]
        lines.append('| '+('H-RCAM' if method=='HRCAM' else method)+' | '+' | '.join('{:.4f}'.format(v) for v in vals)+' |')
    lines+=['','## Original constant-color context (descriptive)','',
        '| Dataset | Target | Background | n | Inconsistency | H-RCAM AUROC | AP |','|---|---|---|---:|---:|---:|---:|']
    for name,result in all_results.items():
        for entry in result['constant_background_context']:
            lines.append('| {} | {} | {} | {} | {:.2%} | {:.4f} | {:.4f} |'.format(name,entry['cam_mode'],entry['background'],entry['n'],entry['inconsistency_prevalence'],entry['auroc'],entry['ap']))
    lines+=['','CD/KLD/SNO use the original frozen calibration transform, matching the original table policy.',
        'Raw unclipped estimates and clipping counts are retained in summary.json. No new calibration was fitted.',
        'CIs use 1000 paired source resamples. Full differences and median-gap intervals are in summary.json.',
        'Different backgrounds can produce different inconsistency labels and prevalence; AP is not directly prevalence invariant.',
        'These composites test photographic replacement under supplied masks, not imperfect segmentation.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    write_json(output/'status.json',{'state':'complete','elapsed_seconds':time.time()-started,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'gpu_used':False,'script_sha256':sha(__file__),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__})
    print('Analysis complete',flush=True)


if __name__=='__main__': main()
