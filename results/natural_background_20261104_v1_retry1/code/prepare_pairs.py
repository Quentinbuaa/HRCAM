"""Prepare/freeze donor assignments without loading a classifier."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

SAFE_CPUS = set(range(24, 32))


def bind():
    cpus = set(os.sched_getaffinity(0)) & SAFE_CPUS
    if not cpus:
        raise RuntimeError('Approved CPUs unavailable')
    os.sched_setaffinity(0, cpus)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-donor-rows', default='')
    p.add_argument('--visual-exclusions', type=Path)
    args = p.parse_args()
    bind()
    sys.path.insert(0, str(args.root / 'deps_background_py310'))
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw, ImageOps
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    acquisition = args.root / 'data/natural_background_bg20k_v1/acquisition.json'
    metadata = json.loads(acquisition.read_text())
    shard = acquisition.parent / 'train-00000-of-00006.parquet'
    assert sha(shard) == metadata['file']['lfs']['oid']
    table = pq.read_table(shard, use_threads=False)
    print('Parquet schema', table.schema, 'rows', len(table), flush=True)
    sources, source_files = [], {}
    for name, train, test in [('pets_v1', 'pets_trainval.npz', 'pets_test.npz'),
                              ('cub_v1', 'cub_train.npz', 'cub_test.npz')]:
        baseline = args.root / 'runs' / name
        split = json.loads((baseline / 'splits.json').read_text())
        with np.load(args.root / 'data' / train) as z:
            offset = len(z['labels'])
        with np.load(args.root / 'data' / test) as z:
            ids = np.array(sorted(split['holdout']))
            images, masks, labels = (z[k][ids-offset] for k in ['images', 'masks', 'labels'])
        fraction = masks.reshape(len(masks), -1).mean(1)
        eligible = (fraction >= .05) & (fraction <= .95)
        original_rows = [json.loads(line) for line in (baseline / 'holdout_records.jsonl').read_text().splitlines()]
        expected = {r['source_id'] for r in original_rows if r['mask_eligible']}
        assert set(ids[eligible].tolist()) == expected
        selected = np.flatnonzero(eligible)
        assert len(selected) == {'pets_v1': 993, 'cub_v1': 963}[name]
        np.savez_compressed(output / (name + '_sources.npz'), source_ids=ids[selected],
                            images=images[selected], masks=masks[selected], labels=labels[selected])
        sources.extend((name, int(ids[i])) for i in selected)
        source_files[name] = {'checkpoint': sha(baseline/'classifier.pt'),
                              'original_records': sha(baseline/'holdout_records.jsonl'),
                              'splits': sha(baseline/'splits.json')}
    required = len(sources)
    excluded = {int(x) for x in args.exclude_donor_rows.split(',') if x.strip()}
    if args.visual_exclusions:
        visual = json.loads(args.visual_exclusions.read_text())
        assert visual['classifier_outputs_seen'] is False
        excluded.update(visual['excluded_rows'])
        write_json(output/'donor_visual_exclusions.json', visual)
    crops, donors, rejected, seen = [], [], [], set()
    order = np.random.default_rng(20261104).permutation(len(table))
    for index in order:
        index = int(index)
        if index in excluded:
            rejected.append({'row': index, 'reason': 'visual donor QA exclusion; see donor_visual_exclusions.json'})
            continue
        item = table['image'][index].as_py()
        try:
            raw = item['bytes']
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                width, height = image.size
                if min(width, height) < 224:
                    raise ValueError('Dimension below 224')
                crop = np.asarray(ImageOps.fit(image.convert('RGB'), (224, 224),
                    method=Image.Resampling.BILINEAR, centering=(.5, .5)), dtype=np.uint8)
            # ImageOps.fit is equivalent in geometry to aspect-preserving resize
            # and center crop; its single resampling operation is recorded.
            digest = hashlib.sha256(crop.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError('Exact duplicate RGB crop')
            seen.add(digest)
        except Exception as exc:
            rejected.append({'row': index, 'reason': str(exc)})
            continue
        name, source_id = sources[len(crops)]
        donors.append({'dataset': name, 'source_id': source_id, 'donor_row': index,
                       'width': width, 'height': height, 'encoded_image_sha256': hashlib.sha256(raw).hexdigest(),
                       'crop_sha256': digest, 'path_in_mirror': item.get('path')})
        crops.append(crop)
        if len(crops) == required:
            break
    assert len(crops) == required
    np.save(output / 'donors.npy', np.stack(crops))
    write_json(output / 'donor_assignment.json', donors)
    write_json(output / 'donor_rejections.json', rejected)
    write_json(output / 'preparation_metadata.json', {
        'acquisition': metadata, 'acquisition_sha256': sha(acquisition),
        'source_files': source_files, 'seed': 20261104, 'shard_rows': len(table),
        'assigned_donors': required, 'unique_crop_hashes': len(seen),
        'crop_operation': 'Pillow ImageOps.fit RGB 224x224 bilinear center .5,.5',
        'script_sha256': sha(__file__), 'cpu_affinity': sorted(os.sched_getaffinity(0))})
    selected = np.linspace(0, required-1, 64, dtype=int)
    source_cache = {}
    for name in source_files:
        with np.load(output/(name+'_sources.npz')) as z:
            source_cache[name] = {key: z[key] for key in ('images', 'masks', 'source_ids')}
    positions = {name: {int(s): i for i, s in enumerate(z['source_ids'])} for name,z in source_cache.items()}
    preservation = 0
    # Check every composite, including pairs not displayed in sampled visual QA.
    for i, row in enumerate(donors):
        z = source_cache[row['dataset']]; j = positions[row['dataset']][row['source_id']]
        original, mask = z['images'][j], z['masks'][j].astype(bool)
        composite = np.where(mask[:,:,None], original, crops[i])
        assert np.array_equal(composite[mask], original[mask])
        assert np.array_equal(composite[~mask], crops[i][~mask])
        preservation += 1
    for page in range(4):
        sheet = Image.new('RGB', (1200, 540), 'white'); draw = ImageDraw.Draw(sheet)
        for cell, i in enumerate(selected[page*16:(page+1)*16]):
            row = donors[i]; z = source_cache[row['dataset']]
            j = positions[row['dataset']][row['source_id']]
            original, mask = z['images'][j], z['masks'][j].astype(bool)
            composite = np.where(mask[:,:,None], original, crops[i])
            x, y = (cell%4)*300, (cell//4)*135
            for col, arr in enumerate([original, crops[i], composite]):
                sheet.paste(Image.fromarray(arr).resize((96,96)), (x+col*100,y))
            draw.text((x,y+98), row['dataset']+' source '+str(row['source_id']), fill='black')
            draw.text((x,y+112), 'donor row '+str(row['donor_row'])+' | source / donor / composite', fill='black')
        sheet.save(output/('qa_page_'+str(page+1)+'.jpg'), quality=94)
    write_json(output/'preparation_checks.json', {'all_foregrounds_pixel_exact': preservation,
        'backgrounds_match_donors': preservation, 'qa_sample_assignment_positions': selected.tolist(),
        'model_evaluated': False, 'visual_review_pending': True})
    print('Prepared', required, 'pairs; visual QA required before inference', flush=True)


if __name__ == '__main__':
    main()
