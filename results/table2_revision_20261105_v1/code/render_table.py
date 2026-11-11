"""Render the standalone Markdown draft and full-precision CSV."""
import argparse
import csv
import json
from pathlib import Path
from common import METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();run=args.run;summary=json.loads((run/'summary.json').read_text())
    assert json.loads((run/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
    lines=['# Table 2. Group-divergence statistics','',
        'Group-divergence statistics for the baseline measures and H-RCAM. '
        'Bracketed values are 95% source-cluster confidence intervals.','']
    csv_rows=[]
    for name,title in [('pets_v1','Oxford-IIIT Pet'),('cub_v1','CUB-200-2011')]:
        lines += ['## '+title,'',
            '| Method | Δmedian [95% CI] | Δmean | Δmax | Δmin | U (descriptive) | A_U [95% CI] |',
            '|---|---:|---:|---:|---:|---:|---:|']
        maximum=max(summary[name]['methods'][m]['A_U'] for m in METHODS)
        for method in METHODS:
            e=summary[name]['methods'][method];label='**H-RCAM**' if method=='H-RCAM' else 'SSIM distance' if method=='SSIM' else method
            median='{:.4f} [{:.6f}, {:.6f}]'.format(e['delta_median'],*e['delta_median_ci95'])
            point=f"{e['A_U']:.4f}"
            if e['A_U']==maximum:point='**'+point+'**'
            rank=point+' [{:.4f}, {:.4f}]'.format(*e['A_U_ci95'])
            cells=[label,median]+[f'{e[k]:.4f}' for k in ('delta_mean','delta_max','delta_min')]+[f"{e['U']:.1f}",rank]
            lines.append('| '+' | '.join(cells)+' |')
            csv_rows.append({'dataset':name,'method':method,**{k:e[k] for k in ('delta_median','delta_mean','delta_max','delta_min','U','A_U')},
                'delta_median_ci_low':e['delta_median_ci95'][0],'delta_median_ci_high':e['delta_median_ci95'][1],
                'A_U_ci_low':e['A_U_ci95'][0],'A_U_ci_high':e['A_U_ci95'][1]})
        lines.append('')
    lines += ['**Notes.**','',
        '1. Each Δ is the statistic of the prediction-inconsistent group minus the same statistic of the '
        'prediction-consistent group. The median-gap CI estimates uncertainty in Δmedian. '
        'Mean, maximum and minimum differences remain descriptive. Effect magnitudes should be compared '
        'within a measure, because the measures use different scales.',
        '2. U uses the prediction-consistent group first and is retained only as a descriptive rank statistic. '
        'A_U = 1 − U/(n₀n₁), where n₀ and n₁ are the consistent and inconsistent pair counts. '
        'A_U is the probability that an inconsistent pair has a larger score than a consistent pair, '
        'with half-credit for ties; it equals the AUROC in the revised Table 3. '
        'The A_U estimates and intervals match Table 3 exactly. Higher A_U indicates better separation.',
        '3. The original ResNet-18 classifiers, layer4 Grad-CAM and own-class targets are retained. '
        'Black, gray and white replacements are pooled. Pet: 2,979 pairs from 993 sources '
        '(n₀ = 2,515; n₁ = 464). CUB: 2,889 pairs from 963 sources (n₀ = 1,061; n₁ = 1,828).',
        '4. Percentile 95% CIs use 1,000 source-image-cluster bootstrap resamples (seed 20261011). '
        'Sources are sampled with replacement and all three background pairs of each sampled source are '
        'retained together. Both group medians are recomputed within each resample. All methods use the same '
        'draws, matching Table 3. Pair-level p-values have been removed; these effect-size intervals are '
        'not a numerical conversion of those p-values.',
        '5. L1, L2 and SSIM distance are the three added baselines, using exactly the definitions in '
        '[the revised Table 3](Table3_revised.md). L1 is mean absolute difference; L2 is root mean squared '
        'difference; SSIM distance is 1−SSIM (11×11 Gaussian window, σ = 1.5, data range 1, '
        'K₁ = 0.01, K₂ = 0.03, population covariance), on the same independently [0,1]-scaled maps.',
        '6. Bold A_U marks the largest observed point estimate within a dataset. It does not establish '
        'superiority over every baseline; paired method comparisons are in '
        '[the Table 3 companion](Table3_paired_comparisons.md). '
        'These development datasets contributed to metric selection. The CIs quantify sampling uncertainty '
        'conditional on the selected metric and do not correct selection-induced optimism.','',
        'All 120 original descriptive/rank values reproduce at the submitted precision. '
        'This is a standalone review draft; the manuscript has not been edited.','']
    (run/'Table2_revised.md').write_text('\n'.join(lines),encoding='utf-8')
    with (run/'Table2_with_ci.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    print('Standalone Markdown Table 2 written',flush=True)


if __name__=='__main__':main()
