"""Fetch one prespecified BG-20k parquet shard; no model evaluation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path('/home/dell/rcam_reproduction_20260911')
if hasattr(os, 'sched_getaffinity'):
    selected = set(os.sched_getaffinity(0)) & set(range(24, 32))
    if not selected:
        raise RuntimeError('No approved CPUs')
    os.sched_setaffinity(0, selected)
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def main():
    base = 'https://hf-mirror.com'
    repo = 'unography/BG-20k-1200px'
    out = ROOT / 'data/natural_background_bg20k_v1'
    out.mkdir(parents=True, exist_ok=True)
    metadata = fetch_json(base + '/api/datasets/' + repo)
    revision = metadata['sha']
    tree = fetch_json(base + '/api/datasets/' + repo + '/tree/' + revision + '?recursive=true&limit=100')
    filename = 'data/train-00000-of-00006.parquet'
    item = next(row for row in tree if row['path'] == filename)
    target = out / Path(filename).name
    expected = item['lfs']['oid']
    url = base + '/datasets/' + repo + '/resolve/' + revision + '/' + filename + '?download=true'
    info = {'repository': repo, 'revision': revision, 'file': item, 'url': url,
            'selection': 'First shard fixed in protocol before evaluation',
            'cpu_affinity': sorted(os.sched_getaffinity(0))}
    (out / 'acquisition.json').write_text(json.dumps(info, indent=2) + '\n')
    if target.exists() and sha(target) == expected:
        print('Verified existing shard', flush=True)
        return
    temporary = target.with_suffix('.parquet.partial')
    started = time.time()
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {'Range': 'bytes=' + str(offset) + '-'} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as response:
                append = offset > 0 and response.status == 206
                downloaded = offset if append else 0
                with temporary.open('ab' if append else 'wb') as f:
                    last = time.time()
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)
                        if time.time() - last > 15:
                            print('downloaded', downloaded, '/', item['size'], 'bytes', flush=True)
                            last = time.time()
            if temporary.stat().st_size != item['size'] or sha(temporary) != expected:
                raise RuntimeError('Shard size/hash mismatch')
            temporary.rename(target)
            info.update(state='complete', sha256=expected, seconds=time.time()-started)
            (out / 'acquisition.json').write_text(json.dumps(info, indent=2) + '\n')
            print(json.dumps(info), flush=True)
            return
        except Exception as exc:
            print('Attempt', attempt + 1, repr(exc), flush=True)
    raise RuntimeError('Donor download failed; inspect preserved partial file/log')


if __name__ == '__main__':
    main()
