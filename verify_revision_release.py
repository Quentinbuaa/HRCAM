"""Verify released cached scores with Python's standard library (no inference).

This checks hashes, AUROC/AP point estimates, group differences, and descriptive
correlations. It checks saved CI endpoints, but does not rerun the bootstrap.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
CHECKS = Counter()
MAX_ERROR = 0.0


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(run, filename):
    with (RESULTS / run / filename).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def equal(actual, expected, kind, tolerance=5e-12):
    global MAX_ERROR
    error = abs(actual - expected)
    assert math.isfinite(actual) and error <= tolerance, (kind, actual, expected)
    MAX_ERROR = max(MAX_ERROR, error)
    CHECKS[kind] += 1


def rank_metrics(records, key):
    ordered = sorted((float(r[key]), bool(r['inconsistent'])) for r in records)
    assert all(math.isfinite(score) for score, _ in ordered), key
    positive = sum(label for _, label in ordered)
    negative = len(ordered) - positive
    assert positive and negative
    groups = []
    for score, label in ordered:
        if not groups or score != groups[-1][0]:
            groups.append([score, 0, 0])
        groups[-1][1 + int(not label)] += 1
    wins = negative_before = 0
    for _, p, n in groups:
        wins += p * negative_before + 0.5 * p * n
        negative_before += n
    tp = seen = 0
    ap = 0.0
    for _, p, n in reversed(groups):
        tp += p
        seen += p + n
        ap += (p / positive) * (tp / seen)
    return {'auroc': wins / (positive * negative), 'ap': ap}


def check_ranking(records, ranking, rename=None):
    rename = rename or {}
    for method, expected in ranking.items():
        observed = rank_metrics(records, rename.get(method, method))
        for metric in ('auroc', 'ap'):
            if metric in expected:
                equal(observed[metric], expected[metric], metric)


def check_ci_tree(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (key.endswith('ci95') or key == 'ci95') and value is not None:
                assert len(value) == 2, (key, value)
                if all(v is not None for v in value):
                    assert all(math.isfinite(v) for v in value) and value[0] <= value[1]
                    CHECKS['saved_ci_endpoint_checks'] += 1
            check_ci_tree(value)
    elif isinstance(obj, list):
        for item in obj:
            check_ci_tree(item)


def ranks(values):
    out = [None] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        for index in order[start:stop]:
            out[index] = (start + 1 + stop) / 2
        start = stop
    return out


def pearson(x, y):
    mx, my = fmean(x), fmean(y)
    a = [v - mx for v in x]
    b = [v - my for v in y]
    return math.fsum(v*w for v, w in zip(a, b)) / math.sqrt(
        math.fsum(v*v for v in a) * math.fsum(v*v for v in b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional verification JSON output')
    args = parser.parse_args()
    manifest = read(ROOT / 'REVISION_RELEASE_MANIFEST.json')
    for name, expected in manifest['files'].items():
        path = ROOT / name
        assert path.is_file(), name
        data = path.read_bytes()
        assert len(data) == expected['bytes'], name
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
        CHECKS['sha256_files'] += 1

    table3 = 'table3_revision_20261105_v1'
    summary = read(RESULTS / table3 / 'summary.json')
    group_summary = read(RESULTS / 'table2_mean_ci_20261105_v1/summary.json')
    for dataset, expected in summary.items():
        records = rows(table3, dataset + '_scores.jsonl')
        equal(len(records), expected['pairs'], 'pair_counts', 0)
        equal(len({r['source_id'] for r in records}), expected['sources'], 'source_counts', 0)
        assert set(Counter(r['source_id'] for r in records).values()) == {3}
        check_ranking(records, expected['ranking'])
        for method, gaps in group_summary[dataset]['methods'].items():
            g0 = [r[method] for r in records if not r['inconsistent']]
            g1 = [r[method] for r in records if r['inconsistent']]
            for name, fun in [('median', median), ('mean', fmean), ('max', max), ('min', min)]:
                equal(fun(g1) - fun(g0), gaps['delta_' + name], 'group_differences')

    expanded = 'table_layouts_baselines_20261105_v1_retry1'
    for name, expected in read(RESULTS / expanded / 'summary.json').items():
        records = rows(expanded, name + '_scores.jsonl')
        equal(len(records), expected['pairs'], 'pair_counts', 0)
        equal(len({r['source_id'] for r in records}), expected['sources'], 'source_counts', 0)
        check_ranking(records, expected['ranking'])

    method_run = 'r1c6_method_layer_20261105_v1'
    for dataset, expected in read(RESULTS / method_run / 'analysis/summary.json').items():
        for convention, target in expected['targets'].items():
            records = rows(method_run, 'analysis/' + dataset + '_' + convention + '_common_scores.jsonl')
            equal(len(records), target['common']['pairs'], 'pair_counts', 0)
            check_ranking(records, target['common']['ranking'])

    target_run = 'r2c3_fixed_gt_20261105_v1'
    for dataset, expected in read(RESULTS / target_run / 'analysis/summary.json').items():
        records = rows(target_run, 'analysis/' + dataset + '_common_scores.jsonl')
        equal(len(records), expected['three_way_common']['n_pairs'], 'pair_counts', 0)
        check_ranking(records, expected['three_way_common']['ranking'])

    natural_run = 'natural_background_20261104_v1_retry1'
    for dataset, expected in read(RESULTS / natural_run / 'analysis/summary.json').items():
        records = rows(natural_run, 'analysis/' + dataset + '_scores.jsonl')
        check_ranking(records, expected['natural_own']['ranking'], {'H-RCAM': 'HRCAM'})

    fusion_run = 'table7_components_20261105_v1'
    for dataset, expected in read(RESULTS / fusion_run / 'fusion_summary.json').items():
        records = rows(fusion_run, dataset + '_fusion_scores.jsonl')
        check_ranking(records, expected['rules'])
    components = read(RESULTS / fusion_run / 'component_summary.json')
    pooled, development = [], []
    for dataset in components:
        if dataset.startswith('pooled_'):
            continue
        records = rows(fusion_run, dataset + '_component_scores.jsonl')
        pooled.extend(records)
        if dataset in ('pets_v1', 'cub_v1'):
            development.extend(records)
        x, y = [r['JSD'] for r in records], [r['NCD'] for r in records]
        equal(pearson(x, y), components[dataset]['pearson_JSD_NCD'], 'correlations')
        equal(pearson(ranks(x), ranks(y)), components[dataset]['spearman_JSD_NCD'], 'correlations')
    for key, records in [('pooled_all_six', pooled), ('pooled_development', development)]:
        x, y = [r['JSD'] for r in records], [r['NCD'] for r in records]
        equal(pearson(x, y), components[key]['pearson_JSD_NCD'], 'correlations')
        equal(pearson(ranks(x), ranks(y)), components[key]['spearman_JSD_NCD'], 'correlations')

    probes = read(RESULTS / 'background_probe_ci_20261109_v1/summary.json')
    for dataset, expected in probes['datasets'].items():
        colors = rows(table3, dataset + '_scores.jsonl')
        natural = rows(natural_run, 'analysis/' + dataset + '_scores.jsonl')
        for probe, values in expected['results'].items():
            if probe == 'Natural':
                selected, method = natural, 'HRCAM'
            else:
                selected = colors if probe == 'Constant color' else [
                    row for row in colors if row['background'] == probe.lower()]
                method = 'H-RCAM'
            equal(rank_metrics(selected, method)['auroc'], values['auroc'], 'background_auroc')

    for name in manifest['files']:
        if name.endswith('summary.json'):
            check_ci_tree(read(ROOT / name))
    result = {'state': 'passed', 'checks': dict(CHECKS), 'maximum_numeric_error': MAX_ERROR,
              'new_model_inference': False, 'new_resampling': False,
              'ci_check_scope': 'Saved finite ordered endpoints; bootstrap not regenerated.'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
