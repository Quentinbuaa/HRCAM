"""Independent closed-form CAM replay and saved-record table audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''): h.update(block)
    return h.hexdigest()


def rank_metrics(rows, key):
    # Pure Python grouped rank/count calculations, independent of sklearn.
    values=sorted([(float(r[key]),bool(r['inconsistent'])) for r in rows],reverse=True)
    positives=sum(y for _,y in values);negatives=len(values)-positives
    if not positives or not negatives: return {'auroc':None,'ap':None}
    tp=fp=0; wins=ap=0.;i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[j][0]==values[i][0]: j+=1
        p=sum(y for _,y in values[i:j]);n=j-i-p
        # Every positive above this group beats all negatives in it;
        # positive/negative ties contribute half a win.
        wins+=tp*n+.5*p*n
        tp+=p;fp+=n
        ap+=p/positives*tp/(tp+fp)
        i=j
    return {'auroc':wins/(positives*negatives),'ap':ap}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    cpus=set(os.sched_getaffinity(0))&set(range(24,32))
    if not cpus: raise RuntimeError('Approved CPUs unavailable')
    os.sched_setaffinity(0,cpus)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.models import resnet18
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    frozen=json.loads((args.run/'FROZEN_INPUTS.json').read_text())
    prepared=args.run/'prepared'
    for name,expected in frozen['prepared_files'].items(): assert sha(prepared/name)==expected,name
    assignments=json.loads((prepared/'donor_assignment.json').read_text())
    donors=np.load(prepared/'donors.npy',mmap_mode='r')
    assert len({r['donor_row'] for r in assignments})==1956
    summary=json.loads((args.run/'analysis/summary.json').read_text())
    result={'state':'passed','datasets':{},'table_entries_verified':0,'table_max_error':0.,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads()}
    expected_hashes={'pets_v1':'f3edfdb3b780e21a407a1aefec16634aef331fddaaacad0064d828f998d6e50c',
                     'cub_v1':'9784845363e53b3a77f3892b27ff0969152afd0e7200dc304f4b1cecede5dc18'}
    for name in ('pets_v1','cub_v1'):
        assert sha(args.root/'runs'/name/'classifier.pt')==expected_hashes[name]
        with np.load(prepared/(name+'_sources.npz')) as z:
            data={key:z[key] for key in ('images','masks','labels','source_ids')}
        maps=np.load(args.run/name/'heatmaps.npy',mmap_mode='r')
        rows=[json.loads(line) for line in (args.run/name/'records.jsonl').read_text().splitlines()]
        lookup={(r['source_id'],r['cam_mode']):r for r in rows}
        selected=[i for i,r in enumerate(assignments) if r['dataset']==name]
        assert [assignments[i]['source_id'] for i in selected]==data['source_ids'].tolist()
        for i,pos in enumerate(selected):
            crop=np.asarray(donors[pos]);mask=data['masks'][i].astype(bool);image=data['images'][i]
            assert hashlib.sha256(crop.tobytes()).hexdigest()==assignments[pos]['crop_sha256']
            composite=image.copy();composite[~mask]=crop[~mask]
            assert np.array_equal(composite[mask],image[mask])
        score_error=0.
        for i,source in enumerate(data['source_ids']):
            for col,mode in [(1,'own'),(2,'fixed')]:
                row=lookup[(int(source),mode)]
                if not row['valid_cam']:continue
                a,b=np.asarray(maps[i,0],float).ravel(),np.asarray(maps[i,col],float).ravel()
                j=float(jensenshannon(a+1e-8,b+1e-8,base=2)**2)
                c=float((1-pearsonr(a,b).statistic)/2)
                score_error=max(score_error,abs(j-row['jsd']),abs(c-row['ncd']))
                assert abs(j-row['jsd'])<2e-13 and abs(c-row['ncd'])<2e-13
        state=torch.load(args.root/'runs'/name/'classifier.pt',map_location='cpu',weights_only=True)
        model=resnet18(weights=None);model.fc=torch.nn.Linear(512,state['fc.weight'].shape[0])
        model.load_state_dict(state);model=model.cuda().eval()
        spatial=torch.nn.Sequential(*list(model.children())[:-2])
        chosen=np.linspace(0,len(data['source_ids'])-1,16,dtype=int)
        def normalized(array):
            x=torch.as_tensor(np.ascontiguousarray(array.transpose(0,3,1,2)),device='cuda',dtype=torch.float32)/255
            return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
        def closed_maps(activation,target):
            # For GAP + linear head, each spatial gradient is w_c/(H*W).
            weights=model.fc.weight[target]/(activation.shape[2]*activation.shape[3])
            native=(activation*weights[:,:,None,None]).sum(1,keepdim=True).relu()
            out=F.interpolate(native,size=(224,224),mode='bilinear',align_corners=False)[:,0]
            out-=out.amin((1,2),keepdim=True)
            return (out/out.amax((1,2),keepdim=True).clamp_min(1e-12)).cpu().numpy()
        train_file,test_file={'pets_v1':('pets_trainval.npz','pets_test.npz'),
                             'cub_v1':('cub_train.npz','cub_test.npz')}[name]
        with np.load(args.root/'data'/train_file) as z:offset=len(z['labels'])
        with np.load(args.root/'runs'/name/'holdout_heatmaps.npz') as z:original_ids=z['source_ids']
        with np.load(args.root/'data'/test_file) as z:original_images=z['images'][original_ids-offset]
        original_positions={int(s):i for i,s in enumerate(original_ids)}
        replay=np.empty((len(chosen),3,224,224),np.float32)
        predictions=[]
        # Match source and follow-up batch contexts independently. The cached
        # source maps use the original split order, including its short batch.
        with torch.no_grad():
            for sample,index in enumerate(chosen):
                source_id=int(data['source_ids'][index]);original_index=original_positions[source_id]
                original_start=original_index//16*16
                f0=spatial(normalized(original_images[original_start:original_start+16]))
                targets0=model.fc(f0.mean((2,3))).argmax(1)
                replay[sample,0]=closed_maps(f0,targets0)[original_index-original_start]
                first=int(targets0[original_index-original_start])
                follow_start=int(index)//16*16;follow_stop=min(follow_start+16,len(data['source_ids']))
                source=data['images'][follow_start:follow_stop]
                mask=data['masks'][follow_start:follow_stop].astype(bool)
                bg=np.asarray(donors[np.array(selected[follow_start:follow_stop])])
                composite=source.copy();composite[~mask]=bg[~mask]
                f1=spatial(normalized(composite));targets1=model.fc(f1.mean((2,3))).argmax(1)
                fixed_targets=torch.tensor([lookup[(int(s),'own')]['original_prediction']
                    for s in data['source_ids'][follow_start:follow_stop]],device='cuda')
                replay[sample,1]=closed_maps(f1,targets1)[int(index)-follow_start]
                replay[sample,2]=closed_maps(f1,fixed_targets)[int(index)-follow_start]
                predictions.append((first,int(targets1[int(index)-follow_start])))
        map_error=float(np.max(np.abs(replay-maps[chosen])))
        np.testing.assert_allclose(replay,maps[chosen],rtol=0,atol=2e-5)
        for i,index in enumerate(chosen):
            row=lookup[(int(data['source_ids'][index]),'own')]
            assert predictions[i][0]==row['original_prediction'] and predictions[i][1]==row['transformed_prediction']
        score_rows=[json.loads(line) for line in (args.run/'analysis'/(name+'_scores.jsonl')).read_text().splitlines()]
        for method,reported in summary[name]['natural_own']['ranking'].items():
            reconstructed=rank_metrics(score_rows,method)
            for metric in ('auroc','ap'):
                error=abs(reconstructed[metric]-reported[metric])
                assert error<2e-12
                result['table_max_error']=max(result['table_max_error'],error)
                result['table_entries_verified']+=1
        fixed=[]
        for row in rows:
            if row['cam_mode']=='fixed' and row['valid_cam']:
                j,c=row['jsd'],row['ncd']
                fixed.append({**row,'HRCAM':2*j*c/(j+c+1e-12),'JSD':j,'NCD':c})
        for method,reported in summary[name]['natural_fixed']['ranking'].items():
            reconstructed=rank_metrics(fixed,method)
            for metric in ('auroc','ap'):
                error=abs(reconstructed[metric]-reported[metric]);assert error<2e-12
                result['table_max_error']=max(result['table_max_error'],error);result['table_entries_verified']+=1
        result['datasets'][name]={'all_donor_hashes_verified':len(selected),'all_foregrounds_pixel_exact':len(selected),
            'checkpoint_matches_original_verified_sha256':True,'closed_form_CAM_replay_sources':len(chosen),
            'closed_form_CAM_max_error':map_error,'independent_SciPy_score_max_error':score_error,
            'sampled_predictions_all_match':True}
        print(name,'independent audit passed',flush=True)
    (args.run/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
