"""Three-convention comparison, source-cluster intervals, and coverage."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time
from common import bind,sha,write_json,read_rows,comparisons

MODES=('own','fixed','ground_truth')


def point(rows,scores):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    if not rows:return {'state':'empty','n_pairs':0,'n_sources':0,'ranking':{}}
    y=np.array([r['inconsistent'] for r in rows],int)
    result={'state':'complete' if len(np.unique(y))==2 else 'single_outcome_class',
        'n_pairs':len(rows),'n_sources':len({r['source_id'] for r in rows}),
        'n_inconsistent':int(y.sum()),'prevalence':float(y.mean())}
    result['ranking']={name:{'auroc':float(roc_auc_score(y,s)) if result['state']=='complete' else None,
        'ap':float(average_precision_score(y,s)) if result['state']=='complete' else None} for name,s in scores.items()}
    return result


def inference(rows,scores,output):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    result=point(rows,scores)
    if result['state']!='complete':return result
    y=np.array([r['inconsistent'] for r in rows],int);source=np.array([r['source_id'] for r in rows])
    groups=[np.flatnonzero(source==s) for s in np.unique(source)]
    samples={name:{key:[] for key in ('auroc','ap')} for name in scores}
    median_draws={mode:[] for mode in MODES};rng=np.random.default_rng(20261011)
    for repeat in range(1000):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        yy=y[indices]
        if len(np.unique(yy))<2:continue
        for name,s in scores.items():
            samples[name]['auroc'].append(float(roc_auc_score(yy,s[indices])))
            samples[name]['ap'].append(float(average_precision_score(yy,s[indices])))
        for mode in MODES:
            h=scores[mode+'_HRCAM'][indices]
            median_draws[mode].append(float(np.median(h[yy==1])-np.median(h[yy==0])))
    successful=len(next(iter(samples.values()))['auroc']);assert successful>=950
    for name in scores:
        for key in ('auroc','ap'):
            result['ranking'][name][key+'_ci95']=np.quantile(samples[name][key],[.025,.975]).tolist()
    contrasts=[('ground_truth_HRCAM','own_HRCAM'),('ground_truth_HRCAM','fixed_HRCAM'),('fixed_HRCAM','own_HRCAM')]
    contrasts += [(mode+'_HRCAM',mode+'_'+component) for mode in MODES for component in ('JSD','NCD')]
    result['paired_differences']={first+'_minus_'+second:{key:{
        'difference':result['ranking'][first][key]-result['ranking'][second][key],
        'ci95':np.quantile(np.array(samples[first][key])-samples[second][key],[.025,.975]).tolist()}
        for key in ('auroc','ap')} for first,second in contrasts}
    result['group_median_gaps']={mode:{
        'difference':float(np.median(scores[mode+'_HRCAM'][y==1])-np.median(scores[mode+'_HRCAM'][y==0])),
        'ci95':np.quantile(median_draws[mode],[.025,.975]).tolist()} for mode in MODES}
    result.update(bootstrap_seed=20261011,bootstrap_resamples=successful)
    np.savez_compressed(output,**{name+'_'+key:np.array(values) for name,entry in samples.items() for key,values in entry.items()})
    return result


def analyze_dataset(args,name,output):
    import numpy as np
    rows=read_rows(args.run/name/'records.jsonl')
    coverage={};all_valid={}
    for mode in MODES:
        selected=[r for r in rows if r['cam_mode']==mode];eligible=[r for r in selected if r['mask_eligible']]
        valid=[r for r in eligible if r['valid_cam']]
        coverage[mode]={'all_pairs':len(selected),'mask_eligible_pairs':len(eligible),
            'mask_eligible_sources':len({r['source_id'] for r in eligible}),
            'valid_pairs':len(valid),'valid_sources':len({r['source_id'] for r in valid}),
            'invalid_pairs':len(eligible)-len(valid),'invalid_consistent':sum(not r['valid_cam'] and not r['inconsistent'] for r in eligible),
            'invalid_inconsistent':sum(not r['valid_cam'] and r['inconsistent'] for r in eligible),
            'by_source_correctness':{str(correct):{'eligible_pairs':sum(r['original_correct']==correct for r in eligible),
                'valid_pairs':sum(r['original_correct']==correct for r in valid),
                'eligible_sources':len({r['source_id'] for r in eligible if r['original_correct']==correct}),
                'valid_sources':len({r['source_id'] for r in valid if r['original_correct']==correct})} for correct in (True,False)}}
        if mode=='ground_truth':
            coverage[mode].update(invalid_source_map_sources=len({r['source_id'] for r in eligible if not r['source_gt_map_valid']}),
                invalid_followup_pairs=sum(not r['follow_gt_map_valid'] for r in eligible))
        all_valid[mode]=point(valid,{method:np.array([r[field] for r in valid],float)
            for method,field in [('HRCAM','HRCAM'),('JSD','jsd'),('NCD','ncd')]})
    matched,scores,keys=comparisons(rows)
    print(name,'three-way common pairs',len(keys),'computing paired source intervals',flush=True)
    common=inference(matched,scores,output/(name+'_bootstrap_samples.npz'))
    matched2,scores2,keys2=comparisons(rows,('own','fixed'))
    two_way=point(matched2,scores2)
    assert round(two_way['ranking']['own_HRCAM']['auroc'],4)=={'pets_v1':.8953,'cub_v1':.8067}[name]
    assert round(two_way['ranking']['fixed_HRCAM']['auroc'],4)=={'pets_v1':.7973,'cub_v1':.7658}[name]
    assert round(all_valid['own']['ranking']['HRCAM']['auroc'],4)=={'pets_v1':.8953,'cub_v1':.8066}[name]
    lookup={(r['source_id'],r['background'],r['cam_mode']):r for r in rows}
    identity={'source_correct_GT_fixed_score_checks':0,'source_correct_GT_fixed_score_max_error':0.,
              'consistent_own_fixed_score_checks':0,'consistent_own_fixed_score_max_error':0.}
    for row in rows:
        if row['cam_mode']!='ground_truth' or not row['mask_eligible']:continue
        own=lookup[(row['source_id'],row['background'],'own')];fixed=lookup[(row['source_id'],row['background'],'fixed')]
        if row['original_correct']:
            assert row['valid_cam']==fixed['valid_cam']
            if row['valid_cam']:
                error=max(abs(row[k]-fixed[k]) for k in ('jsd','ncd','HRCAM'))
                assert error<2e-12
                identity['source_correct_GT_fixed_score_checks']+=1
                identity['source_correct_GT_fixed_score_max_error']=max(identity['source_correct_GT_fixed_score_max_error'],error)
        if not row['inconsistent']:
            assert own['valid_cam']==fixed['valid_cam']
            if own['valid_cam']:
                error=max(abs(own[k]-fixed[k]) for k in ('jsd','ncd','HRCAM'));assert error==0.
                identity['consistent_own_fixed_score_checks']+=1
                identity['consistent_own_fixed_score_max_error']=max(identity['consistent_own_fixed_score_max_error'],error)
    strata={}
    for correct in (True,False):
        select=np.array([r['original_correct']==correct for r in matched])
        strata['source_correct' if correct else 'source_incorrect']=point([r for r,keep in zip(matched,select) if keep],
            {key:value[select] for key,value in scores.items()})
    backgrounds={}
    for bg in ('black','gray','white'):
        select=np.array([r['background']==bg for r in matched])
        backgrounds[bg]=point([r for r,keep in zip(matched,select) if keep],{key:value[select] for key,value in scores.items()})
    common_rows=[]
    for i,row in enumerate(matched):
        common_rows.append({key:row[key] for key in ('source_id','background','label','original_correct','inconsistent')}
                           | {key:float(value[i]) for key,value in scores.items()})
    (output/(name+'_common_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in common_rows))
    return {'coverage':coverage,'three_way_common':common,'two_way_original_Table8_reproduction':two_way,
        'all_valid_by_convention':all_valid,'source_correctness_strata_common':strata,
        'backgrounds_common':backgrounds,'identity_checks':identity,
        'three_way_excluded_from_original_two_way':len(keys2)-len(keys),
        'three_way_excluded_consistent':sum(not r['inconsistent'] for r in matched2)-sum(not r['inconsistent'] for r in matched),
        'three_way_excluded_inconsistent':sum(r['inconsistent'] for r in matched2)-sum(r['inconsistent'] for r in matched)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind();os.environ['CUDA_VISIBLE_DEVICES']=''
    output=args.run/'analysis';output.mkdir(exist_ok=False);started=time.time()
    results={name:analyze_dataset(args,name,output) for name in ('pets_v1','cub_v1')}
    write_json(output/'summary.json',results)
    write_json(output/'STATUS.json',{'state':'complete','seconds':time.time()-started,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'script_sha256':sha(__file__),'python':sys.version,'gpu_used':False})
    print('Three-convention analysis complete',flush=True)


if __name__=='__main__':main()
