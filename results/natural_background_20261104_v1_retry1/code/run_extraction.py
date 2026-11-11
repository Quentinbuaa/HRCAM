"""Evaluate the frozen natural-background pairs with original ResNet models."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    cpus = set(os.sched_getaffinity(0)) & set(range(24,32))
    if not cpus:
        raise RuntimeError('Approved CPUs unavailable')
    os.sched_setaffinity(0, cpus)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '4'
    sys.path.insert(0, str(args.root))
    import numpy as np
    import torch
    import torchvision
    from torchvision.models import resnet18
    import experiment as exp
    from metrics import heatmap_metrics
    exp.MEAN, exp.STD = [.485,.456,.406], [.229,.224,.225]
    prepared = args.run/'prepared'
    freeze = json.loads((args.run/'FROZEN_INPUTS.json').read_text())
    assert freeze['visual_qa']['state'] == 'passed'
    for name, expected in freeze['prepared_files'].items():
        assert sha(prepared/name) == expected, name
    assert sha(Path(__file__).with_name('PROTOCOL.md')) == freeze['protocol_sha256']
    assignments = json.loads((prepared/'donor_assignment.json').read_text())
    donors = np.load(prepared/'donors.npy', mmap_mode='r')
    meta = json.loads((prepared/'preparation_metadata.json').read_text())
    started = time.time()
    dataset_metadata = {}
    for name, seed in [('pets_v1',20260911),('cub_v1',20260914)]:
        original = args.root/'runs'/name
        output = args.run/name
        output.mkdir(exist_ok=False)
        exp.seed_all(seed)
        checkpoint = original/'classifier.pt'
        before = {key: sha(original/file) for key,file in
                  [('checkpoint','classifier.pt'),('original_records','holdout_records.jsonl'),('splits','splits.json')]}
        assert before == meta['source_files'][name]
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, state['fc.weight'].shape[0])
        model.load_state_dict(state, strict=True)
        model = model.cuda().eval()
        cam = exp.GradCAM(model)
        baseline = [json.loads(line) for line in (original/'holdout_records.jsonl').read_text().splitlines()]
        original_predictions = {r['source_id']:r['original_prediction'] for r in baseline}
        with np.load(prepared/(name+'_sources.npz')) as z:
            data = {key:z[key] for key in ('images','masks','labels','source_ids')}
        with np.load(original/'holdout_heatmaps.npz') as z:
            baseline_maps, baseline_ids = z['original_maps'], z['source_ids']
        baseline_positions = {int(s):i for i,s in enumerate(baseline_ids)}
        # Reproduce the archived source batches, including the final short
        # batch. cuDNN may select a different kernel for a different batch size.
        train_file,test_file = {'pets_v1':('pets_trainval.npz','pets_test.npz'),
                                'cub_v1':('cub_train.npz','cub_test.npz')}[name]
        with np.load(args.root/'data'/train_file) as z:
            train_offset=len(z['labels'])
        with np.load(args.root/'data'/test_file) as z:
            replay_images=z['images'][baseline_ids-train_offset]
        original_batch_error=0.
        for replay_start in range(0,len(baseline_ids),16):
            replay_stop=min(replay_start+16,len(baseline_ids))
            replay_logits,replay_maps,_=cam(exp.tensors(replay_images[replay_start:replay_stop]))
            reference=baseline_maps[replay_start:replay_stop]
            original_batch_error=max(original_batch_error,float(np.max(np.abs(reference-replay_maps))))
            np.testing.assert_allclose(reference,replay_maps,rtol=0,atol=2e-5)
            for k,s in enumerate(baseline_ids[replay_start:replay_stop]):
                assert int(replay_logits[k].argmax())==original_predictions[int(s)]
        del replay_images
        print(name,'original-batch source replay max error',original_batch_error,flush=True)
        selected = [i for i,row in enumerate(assignments) if row['dataset']==name]
        assert [assignments[i]['source_id'] for i in selected] == data['source_ids'].tolist()
        count = len(selected)
        maps = np.lib.format.open_memmap(output/'heatmaps.npy',mode='w+',dtype='float32',shape=(count,3,224,224))
        records, maximum_error, regrouped_error = [], 0., 0.
        original_probabilities, transformed_probabilities = [], []
        for start in range(0,count,16):
            stop=min(start+16,count)
            images = data['images'][start:stop]
            masks = data['masks'][start:stop].astype(bool)
            donor = np.asarray(donors[np.array(selected[start:stop])])
            altered = np.where(masks[:,:,:,None],images,donor).astype(np.uint8)
            assert np.array_equal(altered[masks],images[masks])
            assert np.array_equal(altered[~masks],donor[~masks])
            logits0, regrouped_maps, _ = cam(exp.tensors(images))
            # Reuse the verified archived source templates exactly. The new
            # photographic follow-ups have their own fixed 16-image batching.
            original_maps=np.stack([baseline_maps[baseline_positions[int(s)]] for s in data['source_ids'][start:stop]])
            regrouped_error=max(regrouped_error,float(np.max(np.abs(original_maps-regrouped_maps))))
            pred0 = logits0.argmax(1)
            logits1, own_maps, fixed_maps = cam(exp.tensors(altered),pred0)
            pred1 = logits1.argmax(1).cpu().numpy()
            prob0 = logits0.softmax(1).cpu().numpy()
            prob1 = logits1.softmax(1).cpu().numpy()
            original_probabilities.append(prob0); transformed_probabilities.append(prob1)
            maps[start:stop] = np.stack([original_maps,own_maps,fixed_maps],axis=1)
            consistent = (pred0.cpu().numpy()==pred1)
            np.testing.assert_array_equal(own_maps[consistent],fixed_maps[consistent])
            for i in range(stop-start):
                pos = start+i; source_id = int(data['source_ids'][pos])
                first,second,label = int(pred0[i]),int(pred1[i]),int(data['labels'][pos])
                assert first == original_predictions[source_id], 'Original prediction changed'
                reference = baseline_maps[baseline_positions[source_id]]
                error = float(np.max(np.abs(reference-original_maps[i])))
                maximum_error=max(maximum_error,error)
                np.testing.assert_allclose(reference,original_maps[i],rtol=0,atol=2e-5)
                for mode, follow in [('own',own_maps[i]),('fixed',fixed_maps[i])]:
                    basic = heatmap_metrics(original_maps[i],follow,masks[i])
                    row = {'source_id':source_id,'role':'holdout','background':'natural',
                        'cam_mode':mode,'label':label,'donor_row':assignments[selected[pos]]['donor_row'],
                        'assignment_position':selected[pos],'original_prediction':first,
                        'transformed_prediction':second,'original_correct':first==label,
                        'transformed_correct':second==label,'inconsistent':first!=second,
                        'mask_fraction':float(masks[i].mean()),'mask_eligible':True,
                        'original_confidence':float(prob0[i,first]),
                        'confidence_drop':float(prob0[i,first]-prob1[i,first]),**basic}
                    row['ncd'] = basic['d4']/2 if basic['valid_cam'] else None
                    row = {k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
                    records.append(row)
            if start%80==0:
                print(name,'extracted',stop,'/',count,'elapsed',round(time.time()-started,1),flush=True)
        cam.close(); maps.flush(); del maps
        (output/'records.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in records))
        np.savez_compressed(output/'probabilities.npz', source_ids=data['source_ids'],
            original=np.concatenate(original_probabilities), transformed=np.concatenate(transformed_probabilities))
        for key,value in model.cpu().state_dict().items():
            assert torch.equal(value,state[key]), key
        after = {key:sha(original/file) for key,file in
                 [('checkpoint','classifier.pt'),('original_records','holdout_records.jsonl'),('splits','splits.json')]}
        assert before==after
        dataset_metadata[name] = {'sources':count,'records':len(records),
            'original_map_replay_max_error':maximum_error,'original_predictions_all_match':True,
            'original_batch_replay_max_error':original_batch_error,
            'regrouped_source_map_difference_max':regrouped_error,
            'source_maps':'Reused archived maps after original-order, original-batch-size replay',
            'foreground_pixel_preservation_checked':count,'checkpoint_unchanged':True,
            'original_files_before':before,'original_files_after':after,
            'heatmap_layout':['original_own','natural_own','natural_fixed'],
            'records_sha256':sha(output/'records.jsonl'),'heatmaps_sha256':sha(output/'heatmaps.npy')}
        write_json(output/'extraction_metadata.json',dataset_metadata[name])
        del model,state,data,baseline_maps
        gc.collect(); torch.cuda.empty_cache()
    write_json(args.run/'extraction_status.json',{'state':'complete','datasets':dataset_metadata,
        'elapsed_seconds':time.time()-started,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'torch_threads':torch.get_num_threads(),'torch':torch.__version__,
        'torchvision':torchvision.__version__,'python':sys.version,'numpy':np.__version__,
        'device':torch.cuda.get_device_name(),'script_sha256':sha(__file__),
        'experiment_helper_sha256':sha(args.root/'experiment.py'),
        'metrics_helper_sha256':sha(args.root/'metrics.py'),
        'frozen_inputs_sha256':sha(args.run/'FROZEN_INPUTS.json')})
    print('Natural-background extraction complete',flush=True)


if __name__ == '__main__':
    main()
