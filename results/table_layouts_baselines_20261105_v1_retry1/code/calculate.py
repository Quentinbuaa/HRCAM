import argparse,concurrent.futures,copy,json,os,time
from pathlib import Path
from common import METHODS,bind,sha,write_json,read_rows,weighted_metrics
from cases import cases,load_maps

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();bind()
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score,average_precision_score
fixture=json.loads(Path(__file__).with_name('TABLE4_FIXTURE.json').read_text());summary={};provenance={};started=time.time();published=0
old_methods=[m for m in METHODS if m not in ('L1','L2','SSIM')]
for case in cases(args.root):
    key=case['key'];folder=case['folder'];prefix=case['prefix'];dense=case['architecture']=='DenseNet-121'
    inputs=[folder/(prefix+'_records.jsonl'),folder/(prefix+'_heatmaps.npy' if dense else 'external_heatmaps.npz')]
    if dense:inputs += [folder/(prefix+'_sources.npz'),case['cache']/(case['name']+'_scores.jsonl'),case['cache']/(case['name']+'_summary.json')]
    else:inputs += [folder/'external_transport.jsonl',folder/'external_summary.json']
    before={str(path):sha(path) for path in inputs}
    records=sorted([r for r in read_rows(inputs[0]) if r['cam_mode']=='own' and r['mask_eligible'] and r['valid_cam']],key=lambda r:(r['source_id'],r['background']))
    if dense:
        rows=read_rows(case['cache']/(case['name']+'_scores.jsonl'))
        old=json.loads((case['cache']/(case['name']+'_summary.json')).read_text())
        assert [(r['source_id'],r['background'],r['inconsistent']) for r in rows]==[(r['source_id'],r['background'],r['inconsistent']) for r in records]
    else:
        cached={(r['source_id'],r['background']):r for r in read_rows(folder/'external_transport.jsonl')}
        old=json.loads((folder/'external_summary.json').read_text());assert old['source_cluster_bootstrap']['seed']==case['seed']
        rows=[]
        for r in records:
            j,c=r['jsd'],r['d4']/2;t=cached[(r['source_id'],r['background'])]
            rows.append({'source_id':r['source_id'],'background':r['background'],'inconsistent':r['inconsistent'],
                'JSD':j,'NCD':c,'CD':r['d1'],'KLD':r['d2'],'SNO':r['d3'],'BAT':t['BAT'],'SpatialW1':t['SpatialW1'],
                'FAS':r['foreground_shift'],'BAI':r['background_increase'],'H-RCAM':2*j*c/(j+c+1e-12)})
    positions,maps=load_maps(case);keys={(r['source_id'],r['background']) for r in rows};sources=sorted({r['source_id'] for r in rows})
    def filt(a):return gaussian_filter(a,sigma=1.5,truncate=3.5,mode='reflect')
    def extra(source):
        pos=positions[source];first,_=maps(pos,0);a=np.asarray(first,float);ma=filt(a);va=filt(a*a)-ma*ma;out=[]
        for bg_index,bg in enumerate(('black','gray','white')):
            if (source,bg) not in keys:continue
            _,second=maps(pos,bg_index);b=np.asarray(second,float)
            assert np.isfinite(a).all() and np.isfinite(b).all() and min(a.min(),b.min())>=0 and max(a.max(),b.max())<=1.000001
            mb=filt(b);vb=filt(b*b)-mb*mb;cov=filt(a*b)-ma*mb;d=a-b
            sim=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
            out.append(((source,bg),{'L1':float(np.abs(d).mean()),'L2':float(np.sqrt(np.mean(d*d))),'SSIM':float(1-sim[5:-5,5:-5].mean())}))
        return out
    print(key,'scoring saved maps',len(sources),'sources',flush=True);distances={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,values in enumerate(pool.map(extra,sources)):
            distances.update(values)
            if i%400==0:print(key,'sources scored',i,flush=True)
    for row in rows:row.update(distances[(row['source_id'],row['background'])])
    (args.run/(key+'_scores.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    y=np.array([r['inconsistent'] for r in rows],int);ids=np.array([r['source_id'] for r in rows]);unique,index=np.unique(ids,return_inverse=True)
    assert np.all(np.bincount(index)==3)
    rng=np.random.default_rng(case['seed']);counts=np.stack([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(1000)])
    weights=counts[:,index];ranking={};samples={}
    for method in METHODS:
        values=np.array([r[method] for r in rows]);point={'auroc':float(roc_auc_score(y,values)),'ap':float(average_precision_score(y,values))}
        valid,auc,ap=weighted_metrics(y,values,weights);assert valid.sum()==1000
        samples[method]={'auroc':auc,'ap':ap}
        for metric in ('auroc','ap'):point[metric+'_ci95']=np.quantile(samples[method][metric],[.025,.975]).tolist()
        if method in old_methods:
            if dense:
                for metric in ('auroc','ap'):
                    assert abs(point[metric]-old['ranking'][method][metric])<2e-12
                    assert np.max(np.abs(np.array(point[metric+'_ci95'])-old['ranking'][method][metric+'_ci95']))<2e-12
                point=copy.deepcopy(old['ranking'][method])
            else:
                for metric in ('auroc','ap'):
                    assert round(point[metric],4)==fixture['datasets'][case['dataset']][method][metric];published+=1
                if method in ('JSD','NCD','H-RCAM'):
                    expected=old['source_cluster_bootstrap']['auroc_intervals'][method]
                    assert np.max(np.abs(np.array(point['auroc_ci95'])-np.array(expected)[[0,2]]))<2e-12
        for repeat in (0,1,7):
            take=np.repeat(np.arange(len(y)),weights[repeat])
            assert abs(roc_auc_score(y[take],values[take])-auc[repeat])<2e-12
            assert abs(average_precision_score(y[take],values[take])-ap[repeat])<2e-12
        ranking[method]=point
    contrasts={m:{metric:{'difference':ranking['H-RCAM'][metric]-ranking[m][metric],
        'ci95':np.quantile(samples['H-RCAM'][metric]-samples[m][metric],[.025,.975]).tolist()}
        for metric in ('auroc','ap')} for m in METHODS if m!='H-RCAM'}
    if dense:
        for method in old_methods:
            if method=='H-RCAM':continue
            for metric in ('auroc','ap'):
                assert np.max(np.abs(np.array(contrasts[method][metric]['ci95'])-old['hrcam_minus_baseline'][method][metric]['ci95']))<2e-12
            contrasts[method]=copy.deepcopy(old['hrcam_minus_baseline'][method])
    summary[key]={'architecture':case['architecture'],'dataset':case['dataset'],'pairs':len(rows),'sources':len(unique),
        'inconsistent_pairs':int(y.sum()),'bootstrap_seed':case['seed'],'bootstrap_resamples':1000,'ranking':ranking,'paired_H_minus_baseline':contrasts}
    after={str(path):sha(path) for path in inputs};assert before==after;provenance[key]={'before':before,'after':after}
    write_json(args.run/'summary.json',summary);write_json(args.run/'PROVENANCE.json',provenance)
    del maps,weights;print(key,'all 13 methods and CIs complete',flush=True)
assert published==80
write_json(args.run/'CALCULATION_STATUS.json',{'state':'complete','seconds':time.time()-started,'cases':10,'published_Table4_points_reproduced':published,
    'cpu_affinity':sorted(os.sched_getaffinity(0)),'numerical_threads':4,'ssim_workers':4,'new_model_inference':False})
