"""Two CAM methods at two prespecified ResNet layers; original models/probes."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import json
from common import CONFIGS,BACKGROUNDS,TwoLayerCAM,bind,sha,write_json,read_rows,upscale


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind();sys.path.insert(0,str(args.root))
    import numpy as np
    import torch
    from torchvision.models import resnet18
    import experiment as exp
    from fixed_source_sensitivity import SPECS,components
    exp.MEAN,exp.STD=[.485,.456,.406],[.229,.224,.225]
    started=time.time();metadata={}
    for name in ('pets_v1','cub_v1'):
        spec=SPECS[name];original=args.root/'runs'/name;output=args.run/name;output.mkdir(exist_ok=False)
        tracked={file:original/file for file in ('classifier.pt','holdout_records.jsonl','splits.json','holdout_heatmaps.npz')}
        before={file:sha(path) for file,path in tracked.items()}
        assert before['classifier.pt']==spec['model_sha256'] and before['holdout_records.jsonl']==spec['records_sha256']
        baseline=read_rows(original/'holdout_records.jsonl');lookup={(r['source_id'],r['background'],r['cam_mode']):r for r in baseline}
        with np.load(original/'holdout_heatmaps.npz') as z:
            ids,masks,old_source,old_maps=z['source_ids'],z['masks'],z['original_maps'],z['maps']
        assert ids.tolist()==json.loads((original/'splits.json').read_text())['holdout']
        with np.load(args.root/'data'/spec['train']) as z:offset=len(z['labels'])
        with np.load(args.root/'data'/spec['test']) as z:
            images,labels,data_masks=(z[key][ids-offset] for key in ('images','labels','masks'))
        np.testing.assert_array_equal(masks,data_masks)
        exp.seed_all(20260911 if name=='pets_v1' else 20260914);torch.backends.cudnn.allow_tf32=True
        state=torch.load(original/'classifier.pt',map_location='cpu',weights_only=True)
        model=resnet18(weights=None);model.fc=torch.nn.Linear(512,spec['classes']);model.load_state_dict(state);model=model.cuda().eval()
        extractor=TwoLayerCAM(model)
        native={config:np.empty((len(ids),7,14,14) if 'layer3' in config else (len(ids),7,7,7),np.float32) for config in CONFIGS}
        records=[];replay_error=0.;score_error=0.;foreground_checks=0;consistent_checks=0
        for start in range(0,len(ids),16):
            stop=min(start+16,len(ids));array=images[start:stop];foreground=masks[start:stop].astype(bool)
            logits0,source_native=extractor(exp.normalize(exp.tensors(array)));pred0=logits0.argmax(1)
            source_maps={config:upscale(source_native[0][config]).cpu().numpy() for config in CONFIGS}
            replay_error=max(replay_error,float(np.max(np.abs(source_maps['GradCAM_layer4']-old_source[start:stop]))))
            np.testing.assert_allclose(source_maps['GradCAM_layer4'],old_source[start:stop],rtol=0,atol=2e-5)
            for config in CONFIGS:native[config][start:stop,0]=source_native[0][config].cpu().numpy()
            for bg_index,(bg,color) in enumerate(zip(BACKGROUNDS,(0,128,255))):
                altered=np.where(foreground[:,:,:,None],array,color).astype(np.uint8)
                assert np.array_equal(altered[foreground],array[foreground]);foreground_checks+=len(array)
                logits1,follow_native=extractor(exp.normalize(exp.tensors(altered)),pred0)
                pred1=logits1.argmax(1).cpu().numpy();consistent=pred1==pred0.cpu().numpy()
                for mode_index,mode in enumerate(('own','fixed')):
                    for config in CONFIGS:
                        fmap=upscale(follow_native[mode_index][config]).cpu().numpy()
                        native[config][start:stop,1+2*bg_index+mode_index]=follow_native[mode_index][config].cpu().numpy()
                        if config=='GradCAM_layer4':
                            replay_error=max(replay_error,float(np.max(np.abs(fmap-old_maps[start:stop,bg_index,mode_index]))))
                            np.testing.assert_allclose(fmap,old_maps[start:stop,bg_index,mode_index],rtol=0,atol=2e-5)
                        valid,j,c=components(source_maps[config],fmap)
                        for i,source in enumerate(ids[start:stop]):
                            base=lookup[(int(source),bg,mode)]
                            assert int(pred0[i])==base['original_prediction'] and int(pred1[i])==base['transformed_prediction']
                            if config=='GradCAM_layer4':
                                assert bool(valid[i])==base['valid_cam']
                                if valid[i]:
                                    error=max(abs(j[i]-base['jsd']),abs(c[i]-base['d4']/2));score_error=max(score_error,float(error));assert error<2e-12
                            fields=('source_id','background','label','original_prediction','transformed_prediction',
                                    'original_correct','transformed_correct','inconsistent','mask_fraction','mask_eligible')
                            records.append({**{key:base[key] for key in fields},'cam_mode':mode,'configuration':config,
                                'valid_cam':bool(valid[i]),'JSD':float(j[i]) if valid[i] else None,
                                'NCD':float(c[i]) if valid[i] else None,
                                'HRCAM':float(2*j[i]*c[i]/(j[i]+c[i]+1e-12)) if valid[i] else None})
                for config in CONFIGS:
                    np.testing.assert_array_equal(follow_native[0][config].cpu().numpy()[consistent],
                                                  follow_native[1][config].cpu().numpy()[consistent])
                    consistent_checks+=int(consistent.sum())
            if start%160==0:print(name,'method/layer extraction',stop,'/',len(ids),'elapsed',round(time.time()-started,1),flush=True)
        extractor.close()
        for config in CONFIGS:np.savez_compressed(output/(config+'_native.npz'),source_ids=ids,native=native[config])
        np.savez_compressed(output/'source_metadata.npz',source_ids=ids,labels=labels,masks=masks)
        (output/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
        for key,value in model.cpu().state_dict().items():assert torch.equal(value,state[key]),key
        after={file:sha(path) for file,path in tracked.items()};assert before==after
        metadata[name]={'sources':len(ids),'records':len(records),'source_native_shape':{key:list(value.shape) for key,value in native.items()},
            'original_files_before':before,'original_files_after':after,'original_GradCAM_layer4_map_replay_max_error':replay_error,
            'original_component_replay_max_error':score_error,'foreground_pixel_preservation_checks':foreground_checks,
            'consistent_pair_native_map_identity_checks':consistent_checks,'native_layout':['source','black_own','black_fixed','gray_own','gray_fixed','white_own','white_fixed']}
        write_json(output/'extraction_metadata.json',metadata[name])
        del model,state,native,old_source,old_maps,images,records
        gc.collect();torch.cuda.empty_cache()
    write_json(args.run/'EXTRACTION_STATUS.json',{'state':'complete','datasets':metadata,'seconds':time.time()-started,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'torch':torch.__version__,
        'device':torch.cuda.get_device_name(),'script_sha256':sha(__file__),'common_sha256':sha(Path(__file__).with_name('common.py'))})
    print('Method/layer extraction complete',flush=True)


if __name__=='__main__':main()
