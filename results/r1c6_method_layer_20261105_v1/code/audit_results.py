"""Independent NumPy CAM weighting, SciPy scores, rank tables and intervals."""
import argparse
import os
from pathlib import Path
import sys
import json
from common import CONFIGS,BACKGROUNDS,bind,sha,write_json,read_rows,upscale


def ranks(rows,key):
    a=sorted([(float(r[key]),bool(r['inconsistent'])) for r in rows],reverse=True)
    positives=sum(y for _,y in a);negatives=len(a)-positives
    if not positives or not negatives:return {'auroc':None,'ap':None}
    tp=fp=0;wins=ap=0.;i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[j][0]==a[i][0]:j+=1
        p=sum(y for _,y in a[i:j]);n=j-i-p
        wins+=tp*n+.5*p*n;tp+=p;fp+=n;ap+=p/positives*tp/(tp+fp);i=j
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
        'intervals_verified':0,'interval_endpoint_max_error':0.,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'torch_threads':torch.get_num_threads()}
    for name in ('pets_v1','cub_v1'):
        spec=SPECS[name];folder=args.run/name;rows=read_rows(folder/'records.jsonl')
        lookup={(r['source_id'],r['background'],r['configuration'],r['cam_mode']):r for r in rows}
        native={}
        for config in CONFIGS:
            with np.load(folder/(config+'_native.npz')) as z:native[config]=z['native']
        with np.load(folder/'source_metadata.npz') as z:ids,masks=z['source_ids'],z['masks']
        assert sha(args.root/'runs'/name/'classifier.pt')==spec['model_sha256']
        component_error=0.;validity_checks=0
        for config in CONFIGS:
            for start in range(0,len(ids),16):
                stop=min(start+16,len(ids))
                source=upscale(torch.from_numpy(native[config][start:stop,0]).cuda()).cpu().numpy()
                for bg_index,bg in enumerate(BACKGROUNDS):
                    for mode_index,mode in enumerate(('own','fixed')):
                        follow=upscale(torch.from_numpy(native[config][start:stop,1+bg_index*2+mode_index]).cuda()).cpu().numpy()
                        for i,s in enumerate(ids[start:stop]):
                            row=lookup[(int(s),bg,config,mode)]
                            a,b=np.asarray(source[i],float).ravel(),np.asarray(follow[i],float).ravel()
                            valid=bool(np.isfinite(a).all() and np.isfinite(b).all() and min(a.min(),b.min())>=0
                                and a.sum()>1e-8 and b.sum()>1e-8 and a.std()>1e-8 and b.std()>1e-8)
                            assert valid==row['valid_cam'];validity_checks+=1
                            if not valid:assert row['HRCAM'] is None;continue
                            j=float(jensenshannon(a+1e-8,b+1e-8,base=2)**2)
                            c=float(np.clip((1-pearsonr(a,b).statistic)/2,0,1));h=2*j*c/(j+c+1e-12)
                            error=max(abs(j-row['JSD']),abs(c-row['NCD']),abs(h-row['HRCAM']))
                            assert error<2e-12;component_error=max(component_error,error)
            print(name,config,'all cached native maps and scores audited',flush=True)
        with np.load(args.root/'data'/spec['train']) as z:offset=len(z['labels'])
        with np.load(args.root/'data'/spec['test']) as z:
            images=z['images'][ids-offset];data_masks=z['masks'][ids-offset]
        np.testing.assert_array_equal(data_masks,masks)
        model=resnet18(weights=None);model.fc=torch.nn.Linear(512,spec['classes'])
        model.load_state_dict(torch.load(args.root/'runs'/name/'classifier.pt',map_location='cpu',weights_only=True));model=model.cuda().eval()
        chosen=np.linspace(0,len(ids)-1,12,dtype=int);native_error=0.;normalized_error=0.;gradient_error=0.
        def normalize(array):
            x=torch.as_tensor(np.ascontiguousarray(array.transpose(0,3,1,2)),device='cuda',dtype=torch.float32)/255
            return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        for index in chosen:
            start=int(index)//16*16;stop=min(start+16,len(ids));local=int(index)-start
            original=images[start:stop];foreground=masks[start:stop].astype(bool)
            for bg_index,bg in enumerate(('source',)+BACKGROUNDS):
                array=original.copy()
                if bg!='source':array[~foreground]=(0,128,255)[bg_index-1]
                assert np.array_equal(array[foreground],original[foreground])
                captured={}
                def capture(key):
                    def hook(module,inputs,output):captured[key]=output
                    return hook
                handles=[model.layer3[0].register_forward_hook(capture('layer3_0')),model.layer4.register_forward_hook(capture('layer4'))]
                try:
                    logits=model(normalize(array));own=logits.argmax(1)
                    fixed=torch.tensor([lookup[(int(s),'black','GradCAM_layer4','own')]['original_prediction'] for s in ids[start:stop]],device='cuda')
                    targets=[own] if bg=='source' else [own,fixed]
                    for mode_index,target in enumerate(targets):
                        layers=('layer3_0','layer4')
                        gradients=torch.autograd.grad(logits.gather(1,target[:,None]).sum(),tuple(captured[k] for k in layers),retain_graph=mode_index+1<len(targets))
                        expected=model.fc.weight[target][:,:,None,None].expand_as(gradients[1])/49
                        err=float((gradients[1]-expected).abs().max());gradient_error=max(gradient_error,err)
                        torch.testing.assert_close(gradients[1],expected,rtol=1e-6,atol=2e-8)
                        cache_column=0 if bg=='source' else 1+(bg_index-1)*2+mode_index
                        for layer,gradient in zip(layers,gradients):
                            a=captured[layer].detach().cpu().numpy();g=gradient.detach().cpu().numpy()
                            for method in ('GradCAM','LayerCAM'):
                                # NumPy reference implements the published
                                # weighting independently of the extractor.
                                weighted=a*g.mean(axis=(2,3),keepdims=True) if method=='GradCAM' else a*np.maximum(g,0)
                                reference=np.maximum(weighted.sum(axis=1),0)[local]
                                config=method+'_'+layer;cached=native[config][int(index),cache_column]
                                err=float(np.max(np.abs(reference-cached)));native_error=max(native_error,err)
                                np.testing.assert_allclose(reference,cached,rtol=2e-5,atol=2e-6)
                                first=upscale(torch.from_numpy(reference[None]).cuda()).cpu().numpy()[0]
                                second=upscale(torch.from_numpy(cached[None]).cuda()).cpu().numpy()[0]
                                err=float(np.max(np.abs(first-second)));normalized_error=max(normalized_error,err)
                                np.testing.assert_allclose(first,second,rtol=0,atol=3e-5)
                finally:
                    for handle in handles:handle.remove()
        for mode in ('own','fixed'):
            reported=summary[name]['targets'][mode]
            for config in CONFIGS:
                selected=[r for r in rows if r['cam_mode']==mode and r['configuration']==config and r['mask_eligible'] and r['valid_cam']]
                for metric in ('HRCAM','JSD','NCD'):
                    actual=ranks(selected,metric)
                    for key,value in actual.items():
                        expected=reported['all_valid'][config]['ranking'][metric][key]
                        if value is None:assert expected is None;continue
                        error=abs(value-expected);assert error<2e-12
                        result['point_entries_verified']+=1;result['point_max_error']=max(result['point_max_error'],error)
            common=read_rows(args.run/'analysis'/(name+'_'+mode+'_common_scores.jsonl'))
            for config in CONFIGS:
                actual=ranks(common,config)
                for key,value in actual.items():
                    expected=reported['common']['ranking'][config][key];error=abs(value-expected);assert error<2e-12
                    result['point_entries_verified']+=1;result['point_max_error']=max(result['point_max_error'],error)
            sources=sorted({r['source_id'] for r in common});groups={s:[r for r in common if r['source_id']==s] for s in sources}
            rng=np.random.default_rng(20261105);draws={c:{metric:[] for metric in ('auroc','ap')} for c in CONFIGS}
            for _ in range(1000):
                selected=[r for i in rng.integers(0,len(sources),len(sources)) for r in groups[sources[int(i)]]]
                if len({r['inconsistent'] for r in selected})<2:continue
                for config in CONFIGS:
                    values=ranks(selected,config)
                    for metric,value in values.items():draws[config][metric].append(value)
            pairs=[]
            for config in CONFIGS:
                for metric in ('auroc','ap'):
                    pairs.append((np.quantile(draws[config][metric],[.025,.975]),reported['common']['ranking'][config][metric+'_ci95']))
                    if config!=CONFIGS[0]:
                        pairs.append((np.quantile(np.array(draws[config][metric])-draws[CONFIGS[0]][metric],[.025,.975]),
                            reported['common']['paired_differences'][config+'_minus_GradCAM_layer4'][metric]['ci95']))
            for actual,expected in pairs:
                error=float(np.max(np.abs(actual-expected)));assert error<2e-12
                result['intervals_verified']+=1;result['interval_endpoint_max_error']=max(result['interval_endpoint_max_error'],error)
        result['datasets'][name]={'all_score_validity_checks':validity_checks,'SciPy_component_max_error':component_error,
            'independent_native_replay_sources':len(chosen),'NumPy_native_map_max_error':native_error,
            'normalized_map_max_error':normalized_error,'closed_form_layer4_gradient_max_error':gradient_error}
        print(name,'independent method/layer audit passed',flush=True)
    write_json(args.run/'INDEPENDENT_AUDIT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
