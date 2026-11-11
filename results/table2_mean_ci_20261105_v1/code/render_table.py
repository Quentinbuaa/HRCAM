import argparse
import csv
import json
from pathlib import Path
from common import METHODS

p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();run=args.run;summary=json.loads((run/'summary.json').read_text())
assert json.loads((run/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
lines=['# Table 2. Group-divergence statistics','',
    'Group-divergence statistics for the baseline measures and H-RCAM. '
    'Bracketed values are 95% source-cluster confidence intervals. Maximum and minimum differences are descriptive.','']
csv_rows=[]
for name,title in [('pets_v1','Oxford-IIIT Pet'),('cub_v1','CUB-200-2011')]:
    lines+=['## '+title,'',
        '| Measure | Δmedian [95% CI] | Δmean [95% CI] | Δmax | Δmin |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        e=summary[name]['methods'][method];label='**H-RCAM**' if method=='H-RCAM' else 'SSIM distance' if method=='SSIM' else method
        cells=[label]+['{:.4f} [{:.6f}, {:.6f}]'.format(e[k],*e[k+'_ci95']) for k in ('delta_median','delta_mean')]
        cells += [f'{e[k]:.4f}' for k in ('delta_max','delta_min')]
        lines.append('| '+' | '.join(cells)+' |')
        csv_rows.append({'dataset':name,'method':method,**{k:e[k] for k in ('delta_median','delta_mean','delta_max','delta_min')},
            'delta_median_ci_low':e['delta_median_ci95'][0],'delta_median_ci_high':e['delta_median_ci95'][1],
            'delta_mean_ci_low':e['delta_mean_ci95'][0],'delta_mean_ci_high':e['delta_mean_ci95'][1]})
    lines.append('')
lines+=['**Notes.**','',
    '1. Each Δ is the statistic of the prediction-inconsistent group minus the same statistic of the '
    'prediction-consistent group. A CI entirely above zero supports a positive difference for that '
    'particular summary; an interval spanning zero leaves its direction unresolved. Maximum and minimum '
    'differences are descriptive only and are not used for confidence-interval-based conclusions.',
    '2. Percentile 95% CIs use 1,000 source-image-cluster bootstrap resamples (seed 20261011). Sources '
    'are sampled with replacement, retaining all three background pairs of each sampled source together. '
    'Both group medians and means are recomputed within each resample. The same source draws are used '
    'for all measures and for the revised Table 3.',
    '3. Original ResNet-18 classifiers, layer4 Grad-CAM and own-class targets are retained. Black, gray '
    'and white replacements are pooled. Pet: 2,979 pairs from 993 sources (2,515 consistent; 464 inconsistent). '
    'CUB: 2,889 pairs from 963 sources (1,061 consistent; 1,828 inconsistent).',
    '4. L1, L2 and SSIM distance are the added baselines, with the same definitions as '
    '[the revised Table 3](Table3_revised.md): L1 is mean absolute difference, L2 is root mean squared '
    'difference, and SSIM distance is 1−SSIM on the same independently [0,1]-scaled maps. SSIM uses '
    'an 11×11 Gaussian window, σ = 1.5, data range 1, K₁ = 0.01, K₂ = 0.03 and population covariance.',
    '5. The former U, p and A_U columns are removed. A_U equals AUROC, which is reported separately in '
    'Table 3. Gap magnitudes should not be compared across measures with different scales. These are '
    'individual intervals for within-measure group differences; they do not establish H-RCAM superiority '
    'over other measures or provide simultaneous coverage across all entries.',
    '6. These development datasets contributed to component/fusion selection. The intervals quantify '
    'sampling uncertainty conditional on the selected metric and do not correct selection-induced optimism.',
    '7. BAI has zero medians in both groups on both datasets, and its median-gap interval is [0, 0] in '
    'these resamples because of the large fraction of zero scores. This does not imply identical group '
    'distributions or absence of sampling uncertainty in other summaries.','',
    'All 80 original Δ statistics reproduce at the submitted precision. The original caption and '
    'descriptive values are retained. This is a standalone review draft; the manuscript has not been edited.','']
(run/'Table2_revised.md').write_text('\n'.join(lines),encoding='utf-8')
with (run/'Table2_with_ci.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
print('Agreed five-column Table 2 rendered',flush=True)
