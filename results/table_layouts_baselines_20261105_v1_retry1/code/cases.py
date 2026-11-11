EXTERNAL=(('flowers102_external_v1','Oxford Flowers-102'),('caltech101_external_v1','Caltech-101'),
          ('voc2012_external_v1','PASCAL VOC 2012'),('uecfoodpix_external_v1','UECFoodPixComplete'))


def cases(root):
    out=[]
    for name,label in (('pets_v1','Oxford-IIIT Pet'),('cub_v1','CUB-200-2011'),*EXTERNAL):
        base=root/'runs'/('r1c1_densenet121_20261104_v1_retry1' if name in ('pets_v1','cub_v1') else 'r1c1_densenet121_external4_20261104_v1')
        out.append({'key':'densenet121__'+name,'architecture':'DenseNet-121','dataset':label,'name':name,
            'folder':base/name,'cache':base/'full_baselines_v1','prefix':'holdout' if name in ('pets_v1','cub_v1') else 'external','seed':20261104})
    for name,label in EXTERNAL:
        out.append({'key':'resnet18__'+name,'architecture':'ResNet-18','dataset':label,'name':name,
            'folder':root/'runs'/name,'prefix':'external','seed':20261012})
    return out


def load_maps(case):
    import numpy as np
    folder=case['folder'];prefix=case['prefix']
    if case['architecture']=='DenseNet-121':
        maps=np.load(folder/(prefix+'_heatmaps.npy'),mmap_mode='r')
        with np.load(folder/(prefix+'_sources.npz')) as z:ids=z['source_ids']
        return {int(s):i for i,s in enumerate(ids)},lambda i,bg:(maps[i,0],maps[i,1+2*bg])
    with np.load(folder/'external_heatmaps.npz') as z:ids,original,follow=z['source_ids'],z['original_maps'],z['maps']
    return {int(s):i for i,s in enumerate(ids)},lambda i,bg:(original[i],follow[i,bg,0])
