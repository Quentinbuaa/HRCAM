"""Resource limits, I/O and deterministic native-map reconstruction."""
import hashlib
import json
import os
from pathlib import Path

CONFIGS=('GradCAM_layer4','GradCAM_layer3_0','LayerCAM_layer4','LayerCAM_layer3_0')
BACKGROUNDS=('black','gray','white')


def bind():
    cpus=set(os.sched_getaffinity(0))&set(range(24,32))
    if not cpus:raise RuntimeError('Approved CPUs unavailable')
    os.sched_setaffinity(0,cpus)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def upscale(native):
    import torch.nn.functional as F
    if native.ndim==3:native=native[:,None]
    cam=F.interpolate(native,size=(224,224),mode='bilinear',align_corners=False)[:,0]
    cam=cam-cam.amin((1,2),keepdim=True)
    return cam/cam.amax((1,2),keepdim=True).clamp_min(1e-12)


class TwoLayerCAM:
    def __init__(self,model):
        self.model=model;self.activations={}
        self.handles=[model.layer3[0].register_forward_hook(self.capture('layer3_0')),
                      model.layer4.register_forward_hook(self.capture('layer4'))]

    def capture(self,name):
        def hook(module,inputs,output):self.activations[name]=output
        return hook

    def __call__(self,normalized,fixed=None):
        import torch
        logits=self.model(normalized);own=logits.argmax(1)
        targets=[own] if fixed is None else [own,fixed]
        result=[]
        for index,target in enumerate(targets):
            layers=('layer3_0','layer4')
            gradients=torch.autograd.grad(logits.gather(1,target[:,None]).sum(),
                tuple(self.activations[layer] for layer in layers),retain_graph=index+1<len(targets))
            native={}
            for layer,g in zip(layers,gradients):
                a=self.activations[layer]
                native['GradCAM_'+layer]=(a*g.mean((2,3),keepdim=True)).sum(1).relu().detach()
                native['LayerCAM_'+layer]=(a*g.relu()).sum(1).relu().detach()
            result.append(native)
        return logits.detach(),result

    def close(self):
        for handle in self.handles:handle.remove()
