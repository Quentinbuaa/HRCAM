"""Independent GT map/score replay, rank metrics, and core interval audit."""
import argparse
import json
import os
from pathlib import Path
import sys
from common import bind,sha,write_json,read_rows


def rank_metrics(rows,key):
    values=sorted([(float(r[key]),bool(r['inconsistent'])) for r in rows],reverse=True)
    positives=sum(y for _,y in values);negatives=len(values)-positives
    if not positives or not negatives:return {'auroc':None,'ap':None}
    tp=fp=0;wins=ap=0.;i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[j][0]==values[i][0]:j+=1
        p=sum(y for _,y in values[i:j]);n=j-i-p
        wins+=tp*n+.5*p*n;tp+=p;fp+=n
        ap+=p/positives*tp/(tp+fp);i=j
    return {'auroc':wins/(positives*negatives),'ap':ap}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind();sys.path.insert(0,str(args.root))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet18
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
    from fixed_source_sensitivity import SPECS
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    summary=json.loads((args.run/'analysis/summary.json').read_text())
    result={'state':'passed','datasets':{},'point_entries_verified':0,'point_max_error':0.,
            'core_intervals_verified':0,'interval_endpoint_max_error':0.,
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads()}
    for name in ('pets_v1','cub_v1'):
        spec=SPECS[name];original=args.root/'runs'/name
        assert sha(original/'classifier.pt')==spec['model_sha256']
        assert sha(original/'holdout_records.jsonl')==spec['records_sha256']
        rows=read_rows(args.run/name/'records.jsonl');lookup={(r['source_id'],r['background'],r['cam_mode']):r for r in rows}
        maps=np.load(args.run/name/'ground_truth_maps.npy',mmap_mode='r')
        with np.load(args.run/name/'source_metadata.npz') as z:ids,masks,labels=z['source_ids'],z['masks'],z['labels']
        with np.load(args.root/'data'/spec['train']) as z:offset=len(z['labels'])
        with np.load(args.root/'data'/spec['test']) as z:
            images,data_masks,data_labels=(z[k][ids-offset] for k in ('images','masks','labels'))
        np.testing.assert_array_equal(masks,data_masks);np.testing.assert_array_equal(labels,data_labels)
        score_error=0.;score_checks=0;invalid_verified=0
        for i,source in enumerate(ids):
            for bg_index,bg in enumerate(('black','gray','white')):
                row=lookup[(int(source),bg,'ground_truth')]
                a,b=np.asarray(maps[i,0],float).ravel(),np.asarray(maps[i,bg_index+1],float).ravel()
                valid=bool(np.isfinite(a).all() and np.isfinite(b).all() and min(a.min(),b.min())>=0
                           and a.sum()>1e-8 and b.sum()>1e-8 and a.std()>1e-8 and b.std()>1e-8)
                assert valid==row['valid_cam']
                if not valid:
                    assert row['jsd'] is None and row['ncd'] is None and row['HRCAM'] is None
                    invalid_verified+=1;continue
                j=float(jensenshannon(a+1e-8,b+1e-8,base=2)**2)
                c=float(np.clip((1-pearsonr(a,b).statistic)/2,0,1));h=2*j*c/(j+c+1e-12)
                error=max(abs(j-row['jsd']),abs(c-row['ncd']),abs(h-row['HRCAM']))
                assert error<2e-12;score_error=max(score_error,error);score_checks+=1
        model=resnet18(weights=None);model.fc=torch.nn.Linear(512,spec['classes'])
        model.load_state_dict(torch.load(original/'classifier.pt',map_location='cpu',weights_only=True));model=model.cuda().eval()
        spatial=torch.nn.Sequential(*list(model.children())[:-2]);chosen=np.linspace(0,len(ids)-1,16,dtype=int)
        def normalize(array):
            x=torch.as_tensor(np.ascontiguousarray(array.transpose(0,3,1,2)),device='cuda',dtype=torch.float32)/255
            return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        def closed_cam(activation,target):
            weights=model.fc.weight[target]/(activation.shape[2]*activation.shape[3])
            native=(activation*weights[:,:,None,None]).sum(1,keepdim=True).relu()
            cam=F.interpolate(native,size=(224,224),mode='bilinear',align_corners=False)[:,0]
            cam=cam-cam.amin((1,2),keepdim=True)
            return (cam/cam.amax((1,2),keepdim=True).clamp_min(1e-12)).cpu().numpy()
        map_error=0.;foreground_checks=0
        with torch.no_grad():
            for index in chosen:
                start=int(index)//16*16;stop=min(start+16,len(ids));local=int(index)-start
                array=images[start:stop];foreground=masks[start:stop].astype(bool)
                targets=torch.as_tensor(labels[start:stop],device='cuda',dtype=torch.long)
                activation=spatial(normalize(array));pred=model.fc(activation.mean((2,3))).argmax(1)
                np.testing.assert_equal(int(pred[local]),lookup[(int(ids[index]),'black','own')]['original_prediction'])
                replay=closed_cam(activation,targets)[local]
                error=float(np.max(np.abs(replay-maps[index,0])));map_error=max(map_error,error)
                np.testing.assert_allclose(replay,maps[index,0],rtol=0,atol=2e-5)
                for bg_index,(bg,color) in enumerate([('black',0),('gray',128),('white',255)]):
                    altered=array.copy();altered[~foreground]=color
                    assert np.array_equal(altered[foreground],array[foreground]);foreground_checks+=len(array)
                    activation=spatial(normalize(altered));pred=model.fc(activation.mean((2,3))).argmax(1)
                    assert int(pred[local])==lookup[(int(ids[index]),bg,'own')]['transformed_prediction']
                    replay=closed_cam(activation,targets)[local]
                    error=float(np.max(np.abs(replay-maps[index,bg_index+1])));map_error=max(map_error,error)
                    np.testing.assert_allclose(replay,maps[index,bg_index+1],rtol=0,atol=2e-5)
        common=read_rows(args.run/'analysis'/(name+'_common_scores.jsonl'))
        def check_points(selected,reported,mapping):
            for key,field in mapping.items():
                actual=rank_metrics(selected,field)
                for metric,value in actual.items():
                    expected=reported[key][metric]
                    if value is None:assert expected is None;continue
                    error=abs(value-expected);assert error<2e-12
                    result['point_entries_verified']+=1;result['point_max_error']=max(result['point_max_error'],error)
        check_points(common,summary[name]['three_way_common']['ranking'],{key:key for key in summary[name]['three_way_common']['ranking']})
        for mode in ('own','fixed','ground_truth'):
            selected=[r for r in rows if r['cam_mode']==mode and r['mask_eligible'] and r['valid_cam']]
            check_points(selected,summary[name]['all_valid_by_convention'][mode]['ranking'],{'HRCAM':'HRCAM','JSD':'jsd','NCD':'ncd'})
        # Reconstruct H-RCAM intervals and target contrasts independently,
        # retaining all available backgrounds in each sampled source cluster.
        sources=sorted({r['source_id'] for r in common});groups={s:[r for r in common if r['source_id']==s] for s in sources}
        rng=np.random.default_rng(20261011)
        draws={mode:{metric:[] for metric in ('auroc','ap')} for mode in ('own','fixed','ground_truth')}
        medians={mode:[] for mode in draws}
        for repeat in range(1000):
            selected=[r for i in rng.integers(0,len(sources),len(sources)) for r in groups[sources[int(i)]]]
            if len({r['inconsistent'] for r in selected})<2:continue
            for mode in draws:
                metrics=rank_metrics(selected,mode+'_HRCAM')
                for key,value in metrics.items():draws[mode][key].append(value)
                medians[mode].append(float(np.median([r[mode+'_HRCAM'] for r in selected if r['inconsistent']])
                    -np.median([r[mode+'_HRCAM'] for r in selected if not r['inconsistent']])))
        checked=[];reported=summary[name]['three_way_common']
        for mode in draws:
            for metric in ('auroc','ap'):
                checked.append((np.quantile(draws[mode][metric],[.025,.975]),reported['ranking'][mode+'_HRCAM'][metric+'_ci95']))
            checked.append((np.quantile(medians[mode],[.025,.975]),reported['group_median_gaps'][mode]['ci95']))
        for first,second in [('ground_truth','own'),('ground_truth','fixed'),('fixed','own')]:
            for metric in ('auroc','ap'):
                delta=np.array(draws[first][metric])-draws[second][metric]
                checked.append((np.quantile(delta,[.025,.975]),reported['paired_differences'][first+'_HRCAM_minus_'+second+'_HRCAM'][metric]['ci95']))
        for actual,expected in checked:
            error=float(np.max(np.abs(actual-expected)));assert error<2e-12
            result['core_intervals_verified']+=1;result['interval_endpoint_max_error']=max(result['interval_endpoint_max_error'],error)
        result['datasets'][name]={'GT_valid_scores_independently_checked':score_checks,
            'GT_invalid_pairs_independently_checked':invalid_verified,'SciPy_score_max_error':score_error,
            'closed_form_GT_replay_sources':len(chosen),'closed_form_GT_map_max_error':map_error,
            'foreground_replay_checks':foreground_checks,'checkpoint_matches_verified_original':True}
        print(name,'independent GT maps, metrics, table and intervals passed',flush=True)
    write_json(args.run/'INDEPENDENT_AUDIT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
