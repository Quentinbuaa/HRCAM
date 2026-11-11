import argparse,json,os,time,copy
from pathlib import Path
from common import bind,sha,write_json,read_rows,weighted_metrics,weighted_median

RULES=('JSD','NCD','Minimum','Maximum','Arithmetic mean','Geometric mean','Harmonic (H-RCAM)')
EXTERNAL=('flowers102_external_v1','caltech101_external_v1','voc2012_external_v1','uecfoodpix_external_v1')
NAMES=('pets_v1','cub_v1',*EXTERNAL)
LABELS=('Oxford-IIIT Pet','CUB-200-2011','Oxford Flowers-102','Caltech-101','PASCAL VOC 2012','UECFoodPixComplete')
QUANTILES=(.05,.25,.5,.75,.95)

def describe(rows):
    import numpy as np
    from scipy.stats import pearsonr,spearmanr
    j=np.array([r['JSD'] for r in rows],float);c=np.array([r['NCD'] for r in rows],float)
    h=np.array([r['H-RCAM'] for r in rows],float)
    ratio=np.divide(j,c,out=np.full(len(j),np.nan),where=c>0)
    maximum=np.maximum(j,c);balance=np.divide(np.minimum(j,c),maximum,out=np.full(len(j),np.nan),where=maximum>0)
    finite=ratio[np.isfinite(ratio)];b=balance[np.isfinite(balance)]
    return {'pairs':len(rows),'pearson_JSD_NCD':float(pearsonr(j,c).statistic),
        'spearman_JSD_NCD':float(spearmanr(j,c).statistic),
        'spearman_H_JSD':float(spearmanr(h,j).statistic),'spearman_H_NCD':float(spearmanr(h,c).statistic),
        'JSD_smaller_fraction':float(np.mean(j<c)),'NCD_smaller_fraction':float(np.mean(c<j)),
        'equal_fraction':float(np.mean(j==c)),'JSD_zero_count':int(np.sum(j==0)),'NCD_zero_count':int(np.sum(c==0)),
        'both_zero_count':int(np.sum(maximum==0)),'H_zero_fraction':float(np.mean(h==0)),
        'JSD_over_NCD':{'valid':len(finite),'zero_denominator':int(np.sum(c==0)),
            'quantiles':np.quantile(finite,QUANTILES).tolist(),'fraction_le_0_1':float(np.mean(finite<=.1))},
        'min_over_max':{'valid':len(b),'zero_denominator':int(np.sum(maximum==0)),
            'quantiles':np.quantile(b,QUANTILES).tolist(),'fraction_le_0_1':float(np.mean(b<=.1))}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind()
    import numpy as np
    from sklearn.metrics import roc_auc_score
    started=time.time();t3=args.root/'runs/table3_revision_20261105_v1';external=args.root/'runs/table_layouts_baselines_20261105_v1_retry1'
    assert all(json.loads((folder/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed' for folder in (t3,external))
    fixture=json.loads(Path(__file__).with_name('TABLE7_FIXTURE.json').read_text())
    prior=json.loads((t3/'summary.json').read_text());median_prior=json.loads((args.root/'runs/table2_mean_ci_20261105_v1/summary.json').read_text())
    summary={};component_rows={};provenance={};original_entries=0
    for name,label in zip(NAMES,LABELS):
        path=t3/(name+'_scores.jsonl') if name in NAMES[:2] else external/('resnet18__'+name+'_scores.jsonl')
        before=sha(path);input_rows=read_rows(path)
        component_rows[name]=[{'source_id':r['source_id'],'background':r['background'],'inconsistent':r['inconsistent'],
            'JSD':r['JSD'],'NCD':r['NCD'],'H-RCAM':r['H-RCAM']} for r in input_rows]
        (args.run/(name+'_component_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in component_rows[name]))
        if name in NAMES[:2]:
            rows=[]
            for r in component_rows[name]:
                j,c=r['JSD'],r['NCD'];h=2*j*c/(j+c+1e-12);assert abs(h-r['H-RCAM'])<1e-15
                rows.append({**r,'Minimum':min(j,c),'Maximum':max(j,c),'Arithmetic mean':(j+c)/2,
                    'Geometric mean':float(np.sqrt(j*c)),'Harmonic (H-RCAM)':h})
            (args.run/(name+'_fusion_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
            y=np.array([r['inconsistent'] for r in rows],int);ids=np.array([r['source_id'] for r in rows])
            unique,index=np.unique(ids,return_inverse=True);assert np.all(np.bincount(index)==3)
            rng=np.random.default_rng(20261011);counts=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(1000)])
            weights=counts[:,index];ranking={};samples={}
            for rule in RULES:
                values=np.array([r[rule] for r in rows],float)
                valid,auc,_=weighted_metrics(y,values,weights);assert valid.sum()==1000;samples[rule]=auc
                delta=weighted_median(values[y==1],weights[:,y==1])-weighted_median(values[y==0],weights[:,y==0])
                entry={'delta_median':float(np.median(values[y==1])-np.median(values[y==0])),
                    'delta_median_ci95':np.quantile(delta,[.025,.975]).tolist(),
                    'auroc':float(roc_auc_score(y,values)),'auroc_ci95':np.quantile(auc,[.025,.975]).tolist()}
                for metric in ('delta_median','auroc'):
                    assert round(entry[metric],4)==fixture['datasets'][name][rule][metric];original_entries+=1
                if rule in ('JSD','NCD','Harmonic (H-RCAM)'):
                    old_name='H-RCAM' if rule.startswith('Harmonic') else rule
                    assert np.max(np.abs(np.array(entry['auroc_ci95'])-prior[name]['ranking'][old_name]['auroc_ci95']))<2e-12
                    if old_name=='H-RCAM':assert entry['delta_median_ci95']==median_prior[name]['methods']['H-RCAM']['delta_median_ci95']
                ranking[rule]=entry
            for rule in RULES[:-1]:
                ranking[rule]['H_minus_rule_auroc']={'difference':ranking[RULES[-1]]['auroc']-ranking[rule]['auroc'],
                    'ci95':np.quantile(samples[RULES[-1]]-samples[rule],[.025,.975]).tolist()}
            summary[name]={'dataset':label,'pairs':len(rows),'sources':len(unique),'inconsistent_pairs':int(y.sum()),
                'bootstrap_seed':20261011,'bootstrap_resamples':1000,'rules':ranking}
            print(name,'fusion intervals calculated',flush=True)
        assert sha(path)==before;provenance[name]={'source':str(path),'sha256':before}
    assert original_entries==28
    correlations={name:{'dataset':label,**describe(component_rows[name])} for name,label in zip(NAMES,LABELS)}
    for key,names,label in [('pooled_development',NAMES[:2],'Pooled development datasets'),('pooled_all_six',NAMES,'Pooled all six datasets')]:
        correlations[key]={'dataset':label,**describe([r for name in names for r in component_rows[name]])}
    write_json(args.run/'fusion_summary.json',summary);write_json(args.run/'component_summary.json',correlations)
    write_json(args.run/'PROVENANCE.json',provenance)
    write_json(args.run/'CALCULATION_STATUS.json',{'state':'complete','seconds':time.time()-started,
        'submitted_point_entries_reproduced':28,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'numerical_threads':4,'new_model_inference':False})

if __name__=='__main__':main()
