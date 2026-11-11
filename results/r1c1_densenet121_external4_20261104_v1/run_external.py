"""Durable four-dataset DenseNet Table-4 extension; no manuscript edits."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

SPECS = {
 "flowers102_external_v1": dict(dataset="Oxford Flowers-102", train="flowers_trainval.npz", test="flowers_test_selected.npz", classes=102, ntrain=2040, ntest=1020, records_sha="eab605408d2d4b96b1707f7ea17350ba892914b000c5b66e6c46f1b117cbd401"),
 "caltech101_external_v1": dict(dataset="Caltech-101", train="caltech101_train.npz", test="caltech101_test_selected.npz", classes=101, ntrain=7667, ntest=1010, records_sha="3f700f8d2036cd774d5ba9dee3666d37a65115d561ca46d9d5645573a4408b18"),
 "voc2012_external_v1": dict(dataset="PASCAL VOC 2012", train="voc2012_train.npz", test="voc2012_val_selected.npz", classes=20, ntrain=933, ntest=927, records_sha="031202c4ec4f142094e4a1e3102f11fa9e7f555f2b7cfa7434f9d81786ae6312"),
 "uecfoodpix_external_v1": dict(dataset="UECFoodPixComplete", train="uecfoodpix_train.npz", test="uecfoodpix_test_selected.npz", classes=102, ntrain=5766, ntest=586, records_sha="068852c1056825baa95b809828ce958c82164af14cd48c316ef296197d59e236"),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    selected=set(os.sched_getaffinity(0)) & set(range(24,32))
    if not selected: raise RuntimeError("Approved CPUs unavailable")
    os.sched_setaffinity(0,selected)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[key]="4"
    helper_path=args.root/"runs/r1c1_densenet121_20261104_v1_retry1/run_architecture.py"
    spec=importlib.util.spec_from_file_location("frozen_architecture_helpers",helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import numpy as np
    import torch
    import torchvision
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from torchvision.models import densenet121,DenseNet121_Weights
    import torch.nn.functional as F
    assert "5080" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(state="running_neural",started=time.time(),cpu_affinity=sorted(os.sched_getaffinity(0)),pid=os.getpid())
    helper.write_json(args.output/"status.json",status)
    for file in ("run_external.py","EXTERNAL_PROTOCOL.md","complete_external_baselines.py"):
        (args.output/file).write_bytes(Path(__file__).with_name(file).read_bytes())
    helper.self_test()
    try:
        for name,settings in SPECS.items():
            output=args.output/name;output.mkdir()
            original=args.root/"runs"/name
            tracked={"train":args.root/"data"/settings["train"],"test":args.root/"data"/settings["test"],
                     "original_records":original/"external_records.jsonl","original_model":original/"classifier.pt",
                     "source_selection":original/"selected_ids.json"}
            before={key:helper.sha256(path) for key,path in tracked.items()}
            assert before["original_records"]==settings["records_sha"]
            torch.manual_seed(20261011);torch.cuda.manual_seed_all(20261011);np.random.seed(20261011)
            weights=DenseNet121_Weights.IMAGENET1K_V1
            model=densenet121(weights=weights,progress=False)
            model.features=torch.nn.Sequential(model.features,torch.nn.ReLU(inplace=False))
            for parameter in model.parameters():parameter.requires_grad_(False)
            model.classifier=torch.nn.Linear(1024,settings["classes"])
            model=model.cuda().eval()
            frozen={key:value.detach().cpu().clone() for key,value in model.features.state_dict().items()}
            print(name+": loading exact original training arrays",flush=True)
            with np.load(tracked["train"]) as archive:
                training_images,training_labels,training_names=archive["images"],archive["labels"],archive["names"]
            assert len(training_labels)==settings["ntrain"]
            assert np.array_equal(np.unique(training_labels),np.arange(settings["classes"]))
            feature_list=[]
            with torch.inference_mode():
                for start in range(0,len(training_images),16):
                    activation=model.features(helper.normalize_images(training_images[start:start+16]))
                    feature_list.append(F.adaptive_avg_pool2d(activation,1).flatten(1).cpu().numpy())
                    if start % 640==0:print(name+": features "+str(start)+"/"+str(len(training_images)),flush=True)
            features=np.concatenate(feature_list)
            head=LogisticRegression(C=1.,solver="lbfgs",max_iter=1000)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always");head.fit(features,training_labels)
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                raise RuntimeError(name+": head convergence failure; preserve run for numerical diagnosis")
            assert np.array_equal(head.classes_,np.arange(settings["classes"]))
            with torch.no_grad():
                model.classifier.weight.copy_(torch.as_tensor(head.coef_,device="cuda",dtype=torch.float32))
                model.classifier.bias.copy_(torch.as_tensor(head.intercept_,device="cuda",dtype=torch.float32))
            torch.save(model.state_dict(),output/"classifier.pt")
            with np.load(tracked["test"]) as archive:
                images,labels,masks,names=(archive[key] for key in ("images","labels","masks","names"))
            assert len(labels)==settings["ntest"] and len(np.unique(names))==len(names)
            assert set(training_names).isdisjoint(set(names))
            del training_images,features,feature_list
            gc.collect()
            verification=helper.model_verification(model,images)
            ids=np.arange(len(labels))
            np.savez_compressed(output/"external_sources.npz",source_ids=ids,labels=labels,masks=masks,names=names)
            (output/"selected_ids.json").write_bytes(tracked["source_selection"].read_bytes())
            cache=np.lib.format.open_memmap(output/"external_heatmaps.npy",mode="w+",dtype="float32",shape=(len(images),7,224,224))
            records=[]
            for start in range(0,len(images),16):
                stop=min(start+16,len(images));array=images[start:stop];foreground=masks[start:stop].astype(bool)
                with torch.no_grad():activation=model.features(helper.normalize_images(array))
                logits0,source_maps=helper.gradcam_from_features(activation,model.classifier)
                source_map=source_maps[0].cpu().numpy();pred0=logits0.argmax(1)
                cache[start:stop,0]=source_map
                for bg_index,(background,color) in enumerate(helper.BACKGROUNDS):
                    altered=np.where(foreground[:,:,:,None],array,color).astype(np.uint8)
                    assert np.array_equal(altered[foreground],array[foreground])
                    with torch.no_grad():activation=model.features(helper.normalize_images(altered))
                    logits1,maps=helper.gradcam_from_features(activation,model.classifier,pred0)
                    pred1=logits1.argmax(1)
                    same=(pred0==pred1).cpu().numpy()
                    if same.any():np.testing.assert_array_equal(maps[0].cpu().numpy()[same],maps[1].cpu().numpy()[same])
                    for mode_index,mode in enumerate(("own","fixed")):
                        follow_map=maps[mode_index].cpu().numpy();cache[start:stop,1+bg_index*2+mode_index]=follow_map
                        valid,j,c=helper.components(source_map,follow_map)
                        for i in range(len(array)):
                            p0,p1,label=int(pred0[i]),int(pred1[i]),int(labels[start+i])
                            area=float(foreground[i].mean())
                            records.append(dict(source_id=start+i,role="external",background=background,cam_mode=mode,
                                label=label,original_prediction=p0,transformed_prediction=p1,original_correct=p0==label,
                                transformed_correct=p1==label,inconsistent=p0!=p1,mask_fraction=area,mask_eligible=.05<=area<=.95,
                                valid_cam=bool(valid[i]),jsd=float(j[i]) if valid[i] else None,ncd=float(c[i]) if valid[i] else None))
                if start % 160==0:print(name+": maps "+str(stop)+"/"+str(len(images)),flush=True)
            cache.flush();del cache
            for key,value in model.features.state_dict().items():assert torch.equal(value.detach().cpu(),frozen[key])
            (output/"external_records.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in records))
            print(name+": core source-cluster intervals",flush=True)
            result=helper.summarize(records,1000,20261104)
            original_rows=[json.loads(line) for line in tracked["original_records"].read_text().splitlines()]
            assert {row["source_id"] for row in original_rows}==set(ids.tolist())
            assert before=={key:helper.sha256(path) for key,path in tracked.items()}
            metadata=dict(dataset=settings["dataset"],classes=settings["classes"],training_sources=settings["ntrain"],
                evaluation_sources=settings["ntest"],head=dict(C=1,solver="lbfgs",max_iter=1000,n_iter=head.n_iter_.tolist()),
                seed=20261011,batch=16,bootstrap_seed=20261104,bootstrap=1000,backbone_unchanged=True,
                verification=verification,cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),
                torch=torch.__version__,torchvision=torchvision.__version__,sklearn=sklearn.__version__,
                source_hashes=before,source_paths={key:str(path) for key,path in tracked.items()},
                script_sha256=helper.sha256(Path(__file__)),helper_sha256=helper.sha256(helper_path))
            helper.write_json(output/"metadata.json",metadata);helper.write_json(output/"summary.json",result)
            print(name+": neural stage complete",flush=True)
            del model,frozen,images,masks,records;torch.cuda.empty_cache();gc.collect()
        status.update(state="running_full_baselines",neural_ended=time.time());helper.write_json(args.output/"status.json",status)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES="")
        subprocess.run(["/home/dell/miniconda3/envs/py314/bin/python",str(Path(__file__).with_name("complete_external_baselines.py")),
                        "--root",str(args.root),"--run",str(args.output)],env=environment,check=True)
        status.update(state="complete",ended=time.time());helper.write_json(args.output/"status.json",status)
        print("Completed four-dataset DenseNet Table 4 extension",flush=True)
    except Exception as exc:
        status.update(state="failed",ended=time.time(),error=repr(exc));helper.write_json(args.output/"status.json",status);raise


if __name__=="__main__":main()
