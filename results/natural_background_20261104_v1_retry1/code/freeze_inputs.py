"""Freeze inputs after visual QA, before any model evaluation."""
import hashlib
import json
import os
from pathlib import Path
import time
import shutil

cpus=set(os.sched_getaffinity(0))&set(range(24,32))
if not cpus: raise RuntimeError('Approved CPUs unavailable')
os.sched_setaffinity(0,cpus)
ROOT=Path('/home/dell/rcam_reproduction_20260911')
RUN=ROOT/'runs/natural_background_20261104_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()


def main():
    destination=RUN/'FROZEN_INPUTS.json'
    if destination.exists(): raise FileExistsError('Refuse to replace frozen inputs')
    review=json.loads((RUN/'qa_review.json').read_text())
    assert review['state']=='passed' and review['classifier_outputs_seen'] is False
    prepared=RUN/'prepared'
    checks=json.loads((prepared/'preparation_checks.json').read_text())
    checks.update(visual_review_pending=False,visual_review_state='passed',model_evaluated=False)
    (prepared/'preparation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    files={p.name:sha(p) for p in sorted(prepared.iterdir()) if p.is_file()}
    assignment=json.loads((prepared/'donor_assignment.json').read_text())
    assert len(assignment)==1956
    assert len({r['donor_row'] for r in assignment})==1956
    assert len({r['crop_sha256'] for r in assignment})==1956
    excluded=set(json.loads((prepared/'donor_visual_exclusions.json').read_text())['excluded_rows'])
    assert excluded.isdisjoint(r['donor_row'] for r in assignment)
    result={'state':'frozen_before_inference','time':time.time(),'prepared_files':files,
        'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),
        'visual_qa':review,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'no_donor_reuse':True,'unique_crops':1956,'excluded_donor_rows':len(excluded)}
    code=RUN/'code';code.mkdir(exist_ok=False)
    for source in Path(__file__).parent.iterdir():
        if source.is_file() and source.suffix in ('.py','.md','.json'):
            shutil.copy2(source,code/source.name)
    result['code_files']={p.name:sha(p) for p in sorted(code.iterdir())}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print('Frozen',len(assignment),'assignments; manifest SHA256',sha(prepared/'donor_assignment.json'))


if __name__=='__main__': main()
