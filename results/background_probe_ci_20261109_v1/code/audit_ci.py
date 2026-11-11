from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import time

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'tmp/hrcam_stats_deps'))
import numpy as np

run = root / 'rcam_reproduction/results/background_probe_ci_20261109_v1'
summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
draws = np.load(run / 'source_draw_counts.npz')
samples = np.load(run / 'bootstrap_auroc_samples.npz')

def rank_auc(sample):
    """Independent ascending average-rank formula on replicated observations."""
    ordered = sorted(sample)
    total = len(ordered)
    positive = sum(label for _, label in ordered)
    negative = total - positive
    assert positive and negative
    rank_sum = 0.0
    i = 0
    while i < total:
        j = i + 1
        while j < total and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += sum(ordered[k][1] for k in range(i, j)) * ((i + 1 + j) / 2)
        i = j
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)

def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    i = math.floor(position)
    fraction = position - i
    return ordered[i] if fraction == 0 else ordered[i] * (1 - fraction) + ordered[i + 1] * fraction

started = time.time()
maximum = 0.0
checked = 0
for name in ['pets_v1', 'cub_v1']:
    path = root / 'rcam_reproduction/results/table3_revision_20261105_v1' / (name + '_scores.jsonl')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    sources = draws[name + '_source_ids'].tolist()
    groups = {source: [] for source in sources}
    for row in rows:
        groups[row['source_id']].append(row)
    for probe in ['Black', 'Gray', 'White', 'Constant color']:
        source_rows = [[(r['H-RCAM'], int(r['inconsistent'])) for r in groups[source] if probe == 'Constant color' or r['background'] == probe.lower()] for source in sources]
        reconstructed = []
        expected = samples[name + '__' + probe.replace(' ', '_')]
        for i, counts in enumerate(draws[name + '_counts'].tolist()):
            sample = [item for rows_for_source, count in zip(source_rows, counts) for _ in range(count) for item in rows_for_source]
            point = rank_auc(sample)
            maximum = max(maximum, abs(point - float(expected[i])))
            reconstructed.append(point)
            checked += 1
        interval = [percentile(reconstructed, q) for q in [0.025, 0.975]]
        reported = summary['datasets'][name]['results'][probe]['auroc_ci95']
        assert max(abs(a - b) for a, b in zip(interval, reported)) < 2e-12
    print(name, 'all color and pooled intervals independently reconstructed', flush=True)
assert maximum < 2e-12
for name, digest in summary['input_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
audit = {'state': 'passed', 'bootstrap_AUROCs_reconstructed': checked, 'intervals_independently_checked': 8, 'new_per_color_intervals': 6, 'pooled_replay_checks': 2, 'max_error': maximum, 'method': 'Replicated observations, ascending average-rank AUROC, scalar linear-interpolated percentiles', 'seconds': time.time() - started}
(run / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
print(json.dumps(audit), flush=True)
