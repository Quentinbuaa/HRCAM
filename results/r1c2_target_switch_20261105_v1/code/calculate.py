import argparse,concurrent.futures,json,os,time
from pathlib import Path
from common import bind,sha,write_json,read_rows,weighted_metrics
from diagnostic import COMPONENTS,NAMES,GROUPS,components,load_maps,group_value

def describe(values):
    import numpy as np
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
        'quantiles':np.quantile(a,[.05,.25,.5,.75,.95]).tolist(),'min':float(a.min()),'max':float(a.max())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind()
    import numpy as np
    from scipy.stats import pearsonr,spearmanr
    from diagnostic import credits
    started=time.time();base=args.root/'runs/r2c3_fixed_gt_20261105_v1'
    assert json.loads((base/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
    prior=json.loads((base/'analysis/summary.json').read_text());summary={};provenance={}
    for name in NAMES:
        paths=[base/name/'records.jsonl',base/'analysis/summary.json',
            args.root/'runs'/name/'holdout_heatmaps.npz',args.root/'runs'/name/'classifier.pt']
        before={str(p):sha(p) for p in paths};records=read_rows(paths[0]);modes={}
        for mode in ('own','fixed','ground_truth'):
            modes[mode]={(r['source_id'],r['background']):r for r in records
                if r['cam_mode']==mode and r['mask_eligible'] and r['valid_cam']}
        keys=sorted(modes['own'].keys()&modes['fixed'].keys())
        assert set(keys)==modes['own'].keys()&modes['fixed'].keys()&modes['ground_truth'].keys()
        assert len(keys)==prior[name]['three_way_common']['n_pairs']
        positions,original,maps=load_maps(args.root,name);by_source={}
        for key in keys:by_source.setdefault(key[0],[]).append(key)
        def score_source(source):
            pos=positions[source];a=original[pos];out=[];max_error=0.;negative_map_error=0.
            for key in by_source[source]:
                own=modes['own'][key];fixed=modes['fixed'][key]
                for field in ('label','original_prediction','transformed_prediction','inconsistent','original_correct','transformed_correct'):
                    assert own[field]==fixed[field]
                bg=('black','gray','white').index(key[1]);b=maps[pos,bg,1];c=maps[pos,bg,0]
                fixed_replay=components(a,b);own_replay=fixed_replay if np.array_equal(b,c) else components(a,c)
                gap=components(b,c)
                cached_own={'JSD':own['jsd'],'NCD':own['ncd'],'HRCAM':own['HRCAM']}
                cached_fixed={'JSD':fixed['jsd'],'NCD':fixed['ncd'],'HRCAM':fixed['HRCAM']}
                error=max(abs(replay[component]-cached[component]) for replay,cached in ((own_replay,cached_own),(fixed_replay,cached_fixed)) for component in COMPONENTS)
                assert error<2e-12,(name,key,error);max_error=max(max_error,error)
                if not own['inconsistent']:
                    assert np.array_equal(b,c);negative_map_error=max(negative_map_error,float(np.max(np.abs(b-c))))
                    assert cached_own==cached_fixed and all(v==0 for v in gap.values())
                row={'source_id':key[0],'background':key[1],'inconsistent':own['inconsistent'],
                    'label':own['label'],'source_prediction':own['original_prediction'],'follow_prediction':own['transformed_prediction'],
                    'source_correct':own['original_correct'],'follow_correct':own['transformed_correct']}
                assert row['inconsistent']==(row['source_prediction']!=row['follow_prediction'])
                assert row['source_correct']==(row['source_prediction']==row['label'])
                assert row['follow_correct']==(row['follow_prediction']==row['label'])
                for component in COMPONENTS:
                    row['own_'+component]=cached_own[component];row['fixed_'+component]=cached_fixed[component];row['target_'+component]=gap[component]
                out.append(row)
            return out,max_error,negative_map_error
        rows=[];max_replay=0.;map_identity_error=0.
        print(name,'replaying cached triads and scoring same-image target changes',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for i,(items,error,maperror) in enumerate(pool.map(score_source,sorted(by_source))):
                rows+=items;max_replay=max(max_replay,error);map_identity_error=max(map_identity_error,maperror)
                if i%300==0:print(name,'map source',i,flush=True)
        rows.sort(key=lambda r:(r['source_id'],r['background']))
        y=np.array([r['inconsistent'] for r in rows],bool);ids=np.array([r['source_id'] for r in rows])
        unique,index=np.unique(ids,return_inverse=True)
        assert len(unique)==prior[name]['three_way_common']['n_sources'] and int(y.sum())==prior[name]['three_way_common']['n_inconsistent']
        rng=np.random.default_rng(20261011)
        counts=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(1000)])
        weights=counts[:,index];posweights=weights[:,y];negweights=weights[:,~y]
        positives=[r for r in rows if r['inconsistent']]
        result={'pairs':len(rows),'sources':len(unique),'n_inconsistent':int(y.sum()),'n_consistent':int((~y).sum()),
            'map_replay_max_error':max_replay,'consistent_target_map_error':map_identity_error,
            'bootstrap_seed':20261011,'bootstrap_resamples':1000,'components':{}}
        for component in COMPONENTS:
            own=np.array([r['own_'+component] for r in rows]);fixed=np.array([r['fixed_'+component] for r in rows])
            target=np.array([r['target_'+component] for r in rows]);assert np.array_equal(own[~y],fixed[~y])
            e={};samples={}
            for mode,values in [('own',own),('fixed',fixed)]:
                valid,auc,ap=weighted_metrics(y,values,weights);assert valid.sum()==1000
                _,pointauc,pointap=weighted_metrics(y,values,np.ones((1,len(rows)),int))
                e[mode]={'auroc':float(pointauc[0]),'ap':float(pointap[0]),
                    'auroc_ci95':np.quantile(auc,[.025,.975]).tolist(),'ap_ci95':np.quantile(ap,[.025,.975]).tolist()}
                old=prior[name]['three_way_common']['ranking'][mode+'_'+component]
                for metric in ('auroc','ap'):
                    assert abs(e[mode][metric]-old[metric])<2e-12
                    assert np.max(np.abs(np.array(e[mode][metric+'_ci95'])-old[metric+'_ci95']))<2e-12
                samples[mode]={'auroc':auc,'ap':ap}
            e['own_minus_fixed']={metric:{'difference':e['own'][metric]-e['fixed'][metric],
                'ci95':np.quantile(samples['own'][metric]-samples['fixed'][metric],[.025,.975]).tolist()} for metric in ('auroc','ap')}
            full_own=credits(own[~y],own[y],np.ones((1,int((~y).sum())),int))[0]
            full_fixed=credits(own[~y],fixed[y],np.ones((1,int((~y).sum())),int))[0];full_shift=full_own-full_fixed
            boot_shift=credits(own[~y],own[y],negweights)-credits(own[~y],fixed[y],negweights)
            boot_gain=(posweights*boot_shift).sum(1)/posweights.sum(1)
            error=float(np.max(np.abs(boot_gain-(samples['own']['auroc']-samples['fixed']['auroc']))))
            assert error<2e-12 and abs(full_shift.mean()-e['own_minus_fixed']['auroc']['difference'])<2e-12
            e['rank_accounting']={'mean_positive_credit_shift':float(full_shift.mean()),'bootstrap_identity_max_error':error,
                'positive_credit_rises':int(np.sum(full_shift>0)),'positive_credit_falls':int(np.sum(full_shift<0)),
                'positive_credit_unchanged':int(np.sum(full_shift==0)),'mean_credit_own':float(full_own.mean()),'mean_credit_fixed':float(full_fixed.mean())}
            e['contributions']={}
            for group,levels in GROUPS.items():
                partition={};sum_point=0.;sum_boot=np.zeros(1000)
                for level in levels:
                    flag=np.array([group_value(r,group)==level for r in positives]);n=int(flag.sum())
                    contribution=(posweights[:,flag]*boot_shift[:,flag]).sum(1)/posweights.sum(1)
                    point=float(full_shift[flag].sum()/len(positives));sum_point+=point;sum_boot+=contribution
                    partition[level]={'n_changed_pairs':n,'positive_share':n/len(positives),
                        'mean_credit_shift':float(full_shift[flag].mean()) if n else None,'contribution':point,
                        'ci95':np.quantile(contribution,[.025,.975]).tolist()}
                assert abs(sum_point-e['own_minus_fixed']['auroc']['difference'])<2e-12
                assert np.max(np.abs(sum_boot-boot_gain))<2e-12
                e['contributions'][group]=partition
            delta=own[y]-fixed[y]
            e['changed_score_shift']={**describe(delta),'rises':int(np.sum(delta>1e-12)),'falls':int(np.sum(delta < -1e-12)),
                'unchanged_within_1e_12':int(np.sum(np.abs(delta)<=1e-12))}
            e['target_only']={'consistent':describe(target[~y]),'inconsistent':describe(target[y])}
            e['changed_associations']={'pearson_target_score_shift':float(pearsonr(target[y],delta).statistic),
                'spearman_target_score_shift':float(spearmanr(target[y],delta).statistic),
                'pearson_target_rank_shift':float(pearsonr(target[y],full_shift).statistic),
                'spearman_target_rank_shift':float(spearmanr(target[y],full_shift).statistic)}
            k=0
            for i,row in enumerate(rows):
                if row['inconsistent']:
                    row['credit_own_'+component]=float(full_own[k]);row['credit_fixed_'+component]=float(full_fixed[k]);row['credit_shift_'+component]=float(full_shift[k]);k+=1
                else:row['credit_own_'+component]=row['credit_fixed_'+component]=row['credit_shift_'+component]=None
            result['components'][component]=e
        (args.run/(name+'_diagnostic_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        after={str(p):sha(p) for p in paths};assert before==after
        provenance[name]={'before':before,'after':after};summary[name]=result
        write_json(args.run/'summary.json',summary);write_json(args.run/'PROVENANCE.json',provenance)
        del original,maps,weights;print(name,'all map contrasts, component CIs and rank accounting complete',flush=True)
    write_json(args.run/'CALCULATION_STATUS.json',{'state':'complete','seconds':time.time()-started,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'numerical_threads':4,'map_workers':2,
        'new_model_inference':False,'new_maps_extracted':False})

if __name__=='__main__':main()
