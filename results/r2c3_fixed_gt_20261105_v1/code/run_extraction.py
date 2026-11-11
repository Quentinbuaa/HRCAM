"""Extract GT-targeted maps while replaying the original source batches."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from common import bind,sha,write_json,read_rows,enrich


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind();sys.path.insert(0,str(args.root))
    import numpy as np
    import torch
    import torchvision
    from torchvision.models import resnet18
    import experiment as exp
    from fixed_source_sensitivity import SPECS,components
    exp.MEAN,exp.STD=[.485,.456,.406],[.229,.224,.225]
    started=time.time();metadata={}
    for name in ('pets_v1','cub_v1'):
        spec=SPECS[name];original=args.root/'runs'/name;out=args.run/name;out.mkdir(exist_ok=False)
        tracked={file:original/file for file in ['classifier.pt','holdout_records.jsonl','splits.json','holdout_heatmaps.npz']}
        before={file:sha(path) for file,path in tracked.items()}
        assert before['classifier.pt']==spec['model_sha256'] and before['holdout_records.jsonl']==spec['records_sha256']
        baseline=read_rows(original/'holdout_records.jsonl')
        lookup={(r['source_id'],r['background'],r['cam_mode']):r for r in baseline}
        with np.load(original/'holdout_heatmaps.npz') as z:
            ids,masks,old_maps,old_source=z['source_ids'],z['masks'],z['maps'],z['original_maps']
        assert ids.tolist()==json.loads((original/'splits.json').read_text())['holdout']
        with np.load(args.root/'data'/spec['train']) as z:offset=len(z['labels'])
        with np.load(args.root/'data'/spec['test']) as z:
            images,labels,data_masks=(z[k][ids-offset] for k in ('images','labels','masks'))
        assert np.array_equal(masks,data_masks)
        assert ((labels>=0)&(labels<spec['classes'])).all()
        exp.seed_all(20260911 if name=='pets_v1' else 20260914)
        torch.backends.cudnn.allow_tf32=True
        model=resnet18(weights=None);model.fc=torch.nn.Linear(512,spec['classes'])
        state=torch.load(original/'classifier.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state);model=model.cuda().eval();cam=exp.GradCAM(model)
        cache=np.lib.format.open_memmap(out/'ground_truth_maps.npy',mode='w+',dtype='float32',shape=(len(ids),4,224,224))
        probabilities=np.empty((len(ids),4,spec['classes']),np.float32)
        records=[enrich(r) for r in baseline]
        checks={'original_map_replay_max_error':0.,'follow_own_map_replay_max_error':0.,
                'GT_equals_fixed_source_map_max_error':0.,'GT_equals_own_follow_map_max_error':0.,
                'source_correct_map_checks':0,'GT_equals_fixed_follow_checks':0,
                'GT_equals_own_follow_checks':0,'foreground_pixel_checks':0}
        def valid_maps(array):
            a=np.asarray(array,np.float64).reshape(len(array),-1)
            return np.isfinite(a).all(1)&(a.min(1)>=0)&(a.sum(1)>1e-8)&(a.std(1)>1e-8)
        for start in range(0,len(ids),16):
            stop=min(start+16,len(ids));array=images[start:stop];foreground=masks[start:stop].astype(bool)
            target=torch.as_tensor(labels[start:stop],device='cuda',dtype=torch.long)
            logits0,own0,gt0=cam(exp.tensors(array),target)
            pred0=logits0.argmax(1).cpu().numpy();correct=pred0==labels[start:stop]
            probabilities[start:stop,0]=logits0.softmax(1).cpu().numpy()
            checks['original_map_replay_max_error']=max(checks['original_map_replay_max_error'],float(np.max(np.abs(own0-old_source[start:stop]))))
            np.testing.assert_allclose(own0,old_source[start:stop],rtol=0,atol=2e-5)
            if correct.any():
                error=float(np.max(np.abs(gt0[correct]-own0[correct])))
                checks['GT_equals_fixed_source_map_max_error']=max(checks['GT_equals_fixed_source_map_max_error'],error)
                np.testing.assert_array_equal(gt0[correct],own0[correct])
                checks['source_correct_map_checks']+=int(correct.sum())
            cache[start:stop,0]=gt0;source_valid=valid_maps(gt0)
            for bg_index,(bg,color) in enumerate([('black',0),('gray',128),('white',255)]):
                altered=np.where(foreground[:,:,:,None],array,color).astype(np.uint8)
                assert np.array_equal(altered[foreground],array[foreground]);checks['foreground_pixel_checks']+=len(array)
                logits1,own1,gt1=cam(exp.tensors(altered),target)
                pred1=logits1.argmax(1).cpu().numpy();probabilities[start:stop,bg_index+1]=logits1.softmax(1).cpu().numpy()
                checks['follow_own_map_replay_max_error']=max(checks['follow_own_map_replay_max_error'],float(np.max(np.abs(own1-old_maps[start:stop,bg_index,0]))))
                np.testing.assert_allclose(own1,old_maps[start:stop,bg_index,0],rtol=0,atol=2e-5)
                if correct.any():
                    error=float(np.max(np.abs(gt1[correct]-old_maps[start:stop,bg_index,1][correct])))
                    checks['GT_equals_fixed_source_map_max_error']=max(checks['GT_equals_fixed_source_map_max_error'],error)
                    np.testing.assert_allclose(gt1[correct],old_maps[start:stop,bg_index,1][correct],rtol=0,atol=2e-5)
                    checks['GT_equals_fixed_follow_checks']+=int(correct.sum())
                follow_correct=pred1==labels[start:stop]
                if follow_correct.any():
                    error=float(np.max(np.abs(gt1[follow_correct]-own1[follow_correct])))
                    checks['GT_equals_own_follow_map_max_error']=max(checks['GT_equals_own_follow_map_max_error'],error)
                    np.testing.assert_array_equal(gt1[follow_correct],own1[follow_correct])
                    checks['GT_equals_own_follow_checks']+=int(follow_correct.sum())
                cache[start:stop,bg_index+1]=gt1
                valid,j,c=components(gt0,gt1);follow_valid=valid_maps(gt1)
                np.testing.assert_array_equal(valid,source_valid&follow_valid)
                for i,source in enumerate(ids[start:stop]):
                    base=lookup[(int(source),bg,'own')]
                    assert base['original_prediction']==int(pred0[i]) and base['transformed_prediction']==int(pred1[i])
                    assert base['label']==int(labels[start+i]) and base['original_correct']==bool(correct[i])
                    identity={key:base[key] for key in ['source_id','role','background','label','original_prediction',
                        'transformed_prediction','original_correct','transformed_correct','inconsistent','mask_fraction','mask_eligible']}
                    row={**identity,'cam_mode':'ground_truth','target_class':int(labels[start+i]),
                        'source_gt_map_valid':bool(source_valid[i]),'follow_gt_map_valid':bool(follow_valid[i]),
                        'valid_cam':bool(valid[i]),'jsd':float(j[i]) if valid[i] else None,
                        'd4':float(2*c[i]) if valid[i] else None}
                    records.append(enrich(row))
            if start%160==0:print(name,'GT extraction',stop,'/',len(ids),'elapsed',round(time.time()-started,1),flush=True)
        cam.close();cache.flush();del cache
        np.savez_compressed(out/'source_metadata.npz',source_ids=ids,labels=labels,masks=masks)
        np.savez_compressed(out/'probabilities.npz',source_ids=ids,probabilities=probabilities)
        (out/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
        for key,value in model.cpu().state_dict().items():assert torch.equal(value,state[key]),key
        after={file:sha(path) for file,path in tracked.items()};assert before==after
        metadata[name]={'dataset':spec['dataset'],'all_sources':len(ids),'all_records':len(records),
            'eligible_sources':len({r['source_id'] for r in records if r['mask_eligible']}),
            'original_files_before':before,'original_files_after':after,'checks':checks,
            'prepared_test_sha256':sha(args.root/'data'/spec['test']),
            'map_layout':['source_GT','black_GT','gray_GT','white_GT'],
            'records_sha256':sha(out/'records.jsonl'),'maps_sha256':sha(out/'ground_truth_maps.npy')}
        write_json(out/'extraction_metadata.json',metadata[name])
        del model,state,old_maps,old_source,images,records,probabilities
        gc.collect();torch.cuda.empty_cache()
    write_json(args.run/'EXTRACTION_STATUS.json',{'state':'complete','datasets':metadata,
        'seconds':time.time()-started,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'threads':torch.get_num_threads(),'torch':torch.__version__,'torchvision':torchvision.__version__,
        'numpy':np.__version__,'python':sys.version,'device':torch.cuda.get_device_name(),
        'script_sha256':sha(__file__),'original_GradCAM_helper_sha256':sha(args.root/'experiment.py'),
        'component_helper_sha256':sha(args.root/'fixed_source_sensitivity.py')})
    print('GT extraction complete',flush=True)


if __name__=='__main__':main()
