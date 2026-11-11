"""Create sheets for visual inspection of the entire fixed candidate shard."""
import os
from pathlib import Path
import sys
import io
cpus=set(os.sched_getaffinity(0))&set(range(24,32))
if not cpus: raise RuntimeError('Approved CPUs unavailable')
os.sched_setaffinity(0,cpus)
ROOT=Path('/home/dell/rcam_reproduction_20260911')
sys.path.insert(0,str(ROOT/'deps_background_py310'))
import pyarrow.parquet as pq
from PIL import Image,ImageOps,ImageDraw
table=pq.read_table(ROOT/'data/natural_background_bg20k_v1/train-00000-of-00006.parquet',use_threads=False)
out=ROOT/'runs/natural_background_20261104_v1/donor_pool_qa'
out.mkdir(exist_ok=False)
for start in range(0,len(table),200):
    sheet=Image.new('RGB',(2000,1200),'white');draw=ImageDraw.Draw(sheet)
    for i in range(start,min(start+200,len(table))):
        try:
            raw=table['image'][i].as_py()['bytes']
            with Image.open(io.BytesIO(raw)) as im:
                crop=ImageOps.fit(im.convert('RGB'),(96,96),method=Image.Resampling.BILINEAR)
            cell=i-start;x=(cell%20)*100;y=(cell//20)*120
            sheet.paste(crop,(x,y));draw.text((x,y+98),str(i),fill='black')
        except Exception:
            draw.text((x,y+98),'BAD '+str(i),fill='red')
    sheet.save(out/('pool_'+str(start//200+1).zfill(2)+'.jpg'),quality=93)
print('Created complete donor-shard QA sheets')
