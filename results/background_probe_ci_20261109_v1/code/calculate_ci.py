from pathlib import Path
import csv
import ctypes
import hashlib
import json
import os
import sys
import time

root = Path(__file__).resolve().parents[3]
run = root / 'rcam_reproduction/results/background_probe_ci_20261109_v1'
assert not (run / 'summary.json').exists(), 'Inspect the existing calculation before repeating.'
run.mkdir(parents=True, exist_ok=True)
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
cpus = []
if os.name == 'nt':
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process = kernel.GetCurrentProcess()
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert kernel.GetProcessAffinityMask(process, ctypes.byref(current), ctypes.byref(system))
    cpus = [i for i in range(64) if current.value & (1 << i)][:2]
    assert kernel.SetProcessAffinityMask(process, sum(1 << i for i in cpus))
sys.path.insert(0, str(root / 'tmp/hrcam_stats_deps'))
import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def auc(y, scores, weights):
    order = np.argsort(-scores, kind='stable')
    values, labels = scores[order], y[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    positive = np.add.reduceat(weights[:, order] * labels[None, :], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - labels)[None, :], starts, axis=1)
    tp = np.cumsum(positive, axis=1)
    p, n = tp[:, -1], negative.sum(axis=1)
    assert np.all((p > 0) & (n > 0))
    return ((tp - 0.5 * positive) * negative).sum(axis=1) / (p * n)

started = time.time()
old = root / 'rcam_reproduction/results/table3_revision_20261105_v1'
natural = root / 'rcam_reproduction/results/natural_background_20261104_v1_retry1/analysis/summary.json'
original_summary = json.loads((old / 'summary.json').read_text(encoding='utf-8'))
natural_summary = json.loads(natural.read_text(encoding='utf-8'))
with (root / 'rcam_reproduction/results/table6_revision_20261105_v1/Table6_values.csv').open(encoding='utf-8', newline='') as f:
    table_rows = list(csv.DictReader(f))
inputs = [old / 'summary.json', natural, *[old / (name + '_scores.jsonl') for name in ['pets_v1', 'cub_v1']]]
hashes = {str(path.relative_to(root)).replace('\\', '/'): sha(path) for path in inputs}
all_counts, all_samples, output_rows = {}, {}, []
checks = {}
for name, label in [('pets_v1', 'Oxford-IIIT Pet'), ('cub_v1', 'CUB-200-2011')]:
    rows = [json.loads(line) for line in (old / (name + '_scores.jsonl')).read_text(encoding='utf-8').splitlines() if line]
    sources, indices = np.unique([r['source_id'] for r in rows], return_inverse=True)
    assert len(rows) == 3 * len(sources)
    assert all(sum(r['source_id'] == source for r in rows) == 3 for source in sources)
    generator = np.random.default_rng(20261011)
    counts = np.stack([np.bincount(generator.integers(0, len(sources), len(sources)), minlength=len(sources)) for _ in range(1000)])
    all_counts[name + '_source_ids'] = sources
    all_counts[name + '_counts'] = counts.astype(np.uint16)
    checks[name] = {'sources': len(sources), 'pairs': len(rows), 'bootstrap_seed': 20261011, 'resamples': 1000}
    results = {}
    for probe in ['Black', 'Gray', 'White', 'Constant color']:
        take = np.array([probe == 'Constant color' or r['background'] == probe.lower() for r in rows])
        chosen = [r for r, keep in zip(rows, take) if keep]
        y = np.array([int(r['inconsistent']) for r in chosen])
        scores = np.array([r['H-RCAM'] for r in chosen], dtype=float)
        point = float(auc(y, scores, np.ones((1, len(chosen)), dtype=np.int64))[0])
        samples = auc(y, scores, counts[:, indices[take]])
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
        all_samples[name + '__' + probe.replace(' ', '_')] = samples
        if probe == 'Constant color':
            saved = original_summary[name]['ranking']['H-RCAM']
            assert abs(point - saved['auroc']) < 2e-15
            assert np.max(np.abs(np.array(interval) - saved['auroc_ci95'])) < 2e-15
            origin = 'reused Table3; exact replay verified'
        else:
            origin = 'new source-cluster bootstrap from cached Table3 scores'
        results[probe] = {'auroc': point, 'auroc_ci95': interval, 'origin': origin}
    nat = natural_summary[name]['natural_own']
    results['Natural'] = {
        'auroc': nat['ranking']['HRCAM']['auroc'],
        'auroc_ci95': nat['ranking']['HRCAM']['auroc_ci95'],
        'origin': 'reused independently audited natural-background summary; fixed donor assignment',
    }
    for row in [r for r in table_rows if r['dataset'] == label]:
        result = results[row['probe']]
        assert f"{result['auroc']:.4f}" == f"{float(row['auroc']):.4f}"
        low, high = result['auroc_ci95']
        output_rows.append({**row, 'auroc_ci_low': low, 'auroc_ci_high': high, 'ci_origin': result['origin']})
    checks[name]['pooled_Table3_interval_reproduced'] = True
    checks[name]['results'] = results
    print(name, {k: [round(x, 4) for x in v['auroc_ci95']] for k, v in results.items()}, flush=True)

assert all(sha(root / name) == value for name, value in hashes.items())
np.savez_compressed(run / 'source_draw_counts.npz', **all_counts)
np.savez_compressed(run / 'bootstrap_auroc_samples.npz', **all_samples)
with (run / 'Table7_with_auroc_ci.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)
summary = {
    'state': 'calculated_pending_independent_audit',
    'purpose': 'Six missing per-color AUROC CIs for current Table7; cached scores only',
    'new_model_inference': False, 'new_experiments': False, 'new_resampling': True,
    'cpu_affinity': cpus, 'numerical_threads': 1, 'numpy': np.__version__, 'python': sys.version,
    'new_intervals': 6, 'reused_intervals': 4,
    'bootstrap': '1000 source-image draws with replacement; all within-probe pairs of a source retained; percentile 2.5/97.5; same draws across color probes and their aggregate',
    'natural_ci_scope': 'Existing fixed donor-selection/assignment protocol; no new donor assignments',
    'datasets': checks, 'input_sha256': hashes,
    'seconds': time.time() - started,
}
(run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print('Six missing intervals calculated; both pooled Table3 intervals reproduced exactly. Audit required before manuscript integration.', flush=True)
