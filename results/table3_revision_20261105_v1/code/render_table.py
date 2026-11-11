"""Render the review table and companion paired comparisons from checked data."""
import argparse
import csv
import json
from pathlib import Path
from common import METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--original-only',action='store_true');args=p.parse_args()
    run=args.run;summary=json.loads((run/'summary.json').read_text())
    assert json.loads((run/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
    methods=[m for m in METHODS if not args.original_only or m not in ('L1','L2','SSIM')]
    lines=['# Table 3. Discrimination of prediction inconsistency','',
        'Discrimination of prediction inconsistency by the baseline heatmap-dissimilarity metrics and H-RCAM. '
        'Higher AUROC and AP are better. Values are point estimates [95% confidence intervals].','',
        '| Measure | Oxford-IIIT Pet AUROC | Oxford-IIIT Pet AP | CUB-200-2011 AUROC | CUB-200-2011 AP |',
        '|---|---:|---:|---:|---:|']
    maxima={(name,metric):max(summary[name]['ranking'][m][metric] for m in methods)
            for name in ('pets_v1','cub_v1') for metric in ('auroc','ap')}
    csv_rows=[]
    for method in methods:
        cells=[]
        for name in ('pets_v1','cub_v1'):
            entry=summary[name]['ranking'][method]
            for metric in ('auroc','ap'):
                number=f"{entry[metric]:.4f}"
                if entry[metric]==maxima[(name,metric)]:number='**'+number+'**'
                cells.append(number+' [{:.4f}, {:.4f}]'.format(*entry[metric+'_ci95']))
            csv_rows.append({'dataset':name,'method':method,'auroc':entry['auroc'],'auroc_ci_low':entry['auroc_ci95'][0],
                'auroc_ci_high':entry['auroc_ci95'][1],'ap':entry['ap'],'ap_ci_low':entry['ap_ci95'][0],'ap_ci_high':entry['ap_ci95'][1]})
        label='**H-RCAM**' if method=='H-RCAM' else 'SSIM distance' if method=='SSIM' else method
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','**Notes.**','',
        '1. Original ResNet-18 classifiers and layer4 Grad-CAM, using each image\'s own predicted class. '
        'Black, gray and white background replacements are pooled. Pet: 2,979 pairs from 993 source images '
        '(464 prediction-inconsistent pairs); CUB: 2,889 pairs from 963 sources (1,828 prediction-inconsistent pairs).',
        '2. Intervals are percentile 95% CIs from 1,000 source-image-cluster bootstrap resamples (seed 20261011). '
        'All three background pairs of each sampled source are retained together; all methods share the same source draws.',
        '3. Bold indicates the highest unrounded observed point estimate in a column. It does not establish superiority '
        'over every baseline. Paired H-RCAM-minus-baseline intervals are provided in '
        '[the companion comparison file](Table3_paired_comparisons.md).',
        '4. These two development datasets contributed to component/fusion selection. The intervals quantify sampling '
        'uncertainty conditional on the selected metric and do not correct selection-induced optimism.']
    if not args.original_only:
        lines += ['5. L1, L2 and SSIM distance are the newly added baselines. On the same independently [0,1]-scaled '
            '224x224 maps, L1 is mean absolute difference and L2 is root mean squared difference. Their constant '
            'normalizations preserve the rankings of the usual L1/L2 norms. SSIM distance is 1-SSIM, using an 11x11 '
            'Gaussian window (sigma 1.5), data range 1, K1=0.01, K2=0.03 and population covariance. '
            'No parameters were selected using these results.']
    lines += ['', 'The original ten methods reproduce all 40 submitted Table 3 point estimates to four decimal places. '
        'This is a standalone review draft; the manuscript has not been edited.','']
    (run/'Table3_revised.md').write_text('\n'.join(lines),encoding='utf-8')
    with (run/'Table3_with_ci.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    paired=['# Table 3: paired H-RCAM-minus-baseline comparisons','',
        'Differences and 95% source-cluster bootstrap intervals on the original own-class evaluation sets. '
        'The same source draws are used for both methods.','',
        '| Baseline | Pet AUROC difference | Pet AP difference | CUB AUROC difference | CUB AP difference |',
        '|---|---:|---:|---:|---:|']
    for method in methods:
        if method=='H-RCAM':continue
        cells=[]
        for name in ('pets_v1','cub_v1'):
            for metric in ('auroc','ap'):
                entry=summary[name]['paired_H_minus_baseline'][method][metric]
                cells.append('{:+.6f} [{:+.6f}, {:+.6f}]'.format(entry['difference'],*entry['ci95']))
        paired.append('| '+method+' | '+' | '.join(cells)+' |')
    paired += ['', 'A positive difference favors H-RCAM. An interval containing zero does not resolve the direction '
        'of that difference. These are comparison-wise intervals, not a simultaneous multiple-comparison guarantee. '
        'Development-set selection influence remains relevant.','']
    (run/'Table3_paired_comparisons.md').write_text('\n'.join(paired),encoding='utf-8')
    print('Standalone Markdown Table 3 and paired comparison file written',flush=True)


if __name__=='__main__':main()
