"""Read saved-map headers; avoid shadowing the standard-library inspect module."""
from pathlib import Path
import json,os,zipfile
os.sched_setaffinity(0,set(range(24,32)))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import numpy as np
root=Path('/home/dell/rcam_reproduction_20260911');runs=root/'runs'
external=('flowers102_external_v1','caltech101_external_v1','voc2012_external_v1','uecfoodpix_external_v1')
folders=[runs/name for name in external]
for base in runs.glob('r1c1_densenet*'):
    if not base.is_dir():continue
    print('DENSE ROOT',str(base),'files',[p.name for p in base.iterdir() if p.is_dir()],flush=True)
    folders += [base/name for name in ('pets_v1','cub_v1',*external) if (base/name).is_dir()]
for folder in folders:
    print('FOLDER',str(folder),flush=True)
    for path in folder.glob('*heatmaps*'):
        if path.suffix=='.npy':print(path.name,np.load(path,mmap_mode='r').shape,flush=True)
        elif path.suffix=='.npz':
            with zipfile.ZipFile(path) as z:
                headers={}
                for name in z.namelist():
                    with z.open(name) as f:
                        version=np.lib.format.read_magic(f)
                        headers[name]=np.lib.format._read_array_header(f,version)[0]
                print(path.name,headers,flush=True)
    for filename in ('external_summary.json','summary.json'):
        path=folder/filename
        if path.exists():
            data=json.loads(path.read_text());print(filename,'keys',list(data),flush=True)
            if 'own' in data:print('own keys',list(data['own']),flush=True)
            if 'ranking' in data:print('ranking keys',list(data['ranking']),flush=True)
