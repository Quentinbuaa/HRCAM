"""Report score/rank/discrimination sensitivity on all-valid and common pairs."""
import argparse
import json
import os
from pathlib import Path
import time
from common import CONFIGS,BACKGROUNDS,bind,sha,write_json,read_rows


def point(rows,scores):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=np.array([r['inconsistent'] for r in rows],int)
    state='empty' if not rows else 'complete' if len(np.unique(y))==2 else 'single_outcome_class'
    return {'state':state,'pairs':len(rows),'sources':len({r['source_id'] for r in rows}),
        'inconsistent':int(y.sum()),'prevalence':float(y.mean()) if len(y) else None,
        'ranking':{key:{'auroc':float(roc_auc_score(y,s)) if state=='complete' else None,
                        'ap':float(average_precision_score(y,s)) if state=='complete' else None} for key,s in scores.items()}}


def infer(rows,scores):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    result=point(rows,scores)
    if result['state']!='complete':return result
    y=np.array([r['inconsistent'] for r in rows],int);source=np.array([r['source_id'] for r in rows])
    groups=[np.flatnonzero(source==s) for s in np.unique(source)];rng=np.random.default_rng(20261105)
    draws={config:{metric:[] for metric in ('auroc','ap')} for config in CONFIGS}
    for _ in range(1000):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        yy=y[indices]
        if len(np.unique(yy))<2:continue
        for config,s in scores.items():
            draws[config]['auroc'].append(float(roc_auc_score(yy,s[indices])))
            draws[config]['ap'].append(float(average_precision_score(yy,s[indices])))
    successful=len(draws[CONFIGS[0]]['auroc']);assert successful>=950
    for config in CONFIGS:
        for metric in ('auroc','ap'):
            result['ranking'][config][metric+'_ci95']=np.quantile(draws[config][metric],[.025,.975]).tolist()
    result['paired_differences']={config+'_minus_GradCAM_layer4':{metric:{
        'difference':result['ranking'][config][metric]-result['ranking'][CONFIGS[0]][metric],
        'ci95':np.quantile(np.array(draws[config][metric])-draws[CONFIGS[0]][metric],[.025,.975]).tolist()}
        for metric in ('auroc','ap')} for config in CONFIGS[1:]}
    result.update(bootstrap_resamples=successful,bootstrap_seed=20261105)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind();os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from scipy.stats import spearmanr
    output=args.run/'analysis';output.mkdir(exist_ok=False);started=time.time();summary={}
    for name in ('pets_v1','cub_v1'):
        rows=read_rows(args.run/name/'records.jsonl');result={'targets':{}}
        for mode in ('own','fixed'):
            valid={config:{(r['source_id'],r['background']):r for r in rows if r['cam_mode']==mode
                   and r['configuration']==config and r['mask_eligible'] and r['valid_cam']} for config in CONFIGS}
            all_valid={};coverage={}
            for config in CONFIGS:
                eligible=[r for r in rows if r['cam_mode']==mode and r['configuration']==config and r['mask_eligible']]
                selected=[valid[config][key] for key in sorted(valid[config])]
                all_valid[config]=point(selected,{metric:np.array([r[metric] for r in selected]) for metric in ('HRCAM','JSD','NCD')})
                coverage[config]={'mask_eligible_pairs':len(eligible),'valid_pairs':len(selected),
                    'invalid_pairs':len(eligible)-len(selected),'valid_sources':len({r['source_id'] for r in selected}),
                    'invalid_consistent':sum(not r['valid_cam'] and not r['inconsistent'] for r in eligible),
                    'invalid_inconsistent':sum(not r['valid_cam'] and r['inconsistent'] for r in eligible)}
            keys=sorted(set.intersection(*(set(valid[config]) for config in CONFIGS)))
            reference=[valid[CONFIGS[0]][key] for key in keys]
            scores={config:np.array([valid[config][key]['HRCAM'] for key in keys]) for config in CONFIGS}
            print(name,mode,'common pairs',len(keys),'source-cluster intervals',flush=True)
            common=infer(reference,scores);agreement={}
            for config in CONFIGS[1:]:
                delta=scores[config]-scores[CONFIGS[0]]
                rho=float(spearmanr(scores[config],scores[CONFIGS[0]]).statistic) if np.std(scores[config])>0 else None
                agreement[config]={'spearman_with_original':rho,'mean_absolute_score_difference':float(np.mean(np.abs(delta))),
                    'median_absolute_score_difference':float(np.median(np.abs(delta))),
                    'p95_absolute_score_difference':float(np.quantile(np.abs(delta),.95)),
                    'maximum_absolute_score_difference':float(np.max(np.abs(delta)))}
            common_rows=[{key:r[key] for key in ('source_id','background','inconsistent','label')}
                         | {config:float(scores[config][i]) for config in CONFIGS} for i,r in enumerate(reference)]
            (output/(name+'_'+mode+'_common_scores.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in common_rows))
            result['targets'][mode]={'all_valid':all_valid,'coverage':coverage,'common':common,'agreement':agreement}
            expected={'pets_v1':{'own':.8953,'fixed':.7973},'cub_v1':{'own':.8066,'fixed':.7658}}[name][mode]
            assert round(all_valid[CONFIGS[0]]['ranking']['HRCAM']['auroc'],4)==expected
        result['consistent_pair_identity_checks']=0
        lookup={(r['source_id'],r['background'],r['configuration'],r['cam_mode']):r for r in rows}
        for row in rows:
            if row['cam_mode']=='own' and not row['inconsistent']:
                other=lookup[(row['source_id'],row['background'],row['configuration'],'fixed')]
                assert row['valid_cam']==other['valid_cam']
                if row['valid_cam']:assert row['HRCAM']==other['HRCAM']
                result['consistent_pair_identity_checks']+=1
        summary[name]=result
    write_json(output/'summary.json',summary)
    write_json(output/'STATUS.json',{'state':'complete','seconds':time.time()-started,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'script_sha256':sha(__file__),'gpu_used':False})
    print('Method/layer analysis complete',flush=True)


if __name__=='__main__':main()
