"""Standalone Table 8 proposal and interpretation; no manuscript mutation."""
import argparse
import csv
import json
from pathlib import Path

MODES=('own','fixed','ground_truth')
LABELS={'own':'Own-predicted class','fixed':'Fixed source-predicted class','ground_truth':'Fixed ground-truth class'}
DATASETS={'pets_v1':'Oxford-IIIT Pet','cub_v1':'CUB-200-2011'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run;summary=json.loads((run/'analysis/summary.json').read_text())
    assert json.loads((run/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
    counts={name:summary[name]['three_way_common']['n_pairs'] for name in DATASETS}
    table=['# Proposed Table 8: attribution-target conventions','',
        'All rows use the same three-way matched valid pairs within each dataset: '
        f"{counts['pets_v1']} Pet pairs and {counts['cub_v1']} CUB pairs.",'',
        '| Attribution target | Pet AUROC | Pet AP | CUB AUROC | CUB AP |','|---|---:|---:|---:|---:|']
    tex=[r'\begin{table}[!htbp]',r'\centering',
        r'\caption{H-RCAM discrimination under three attribution-target conventions on a common valid set ('+
        str(counts['pets_v1'])+' Pet and '+str(counts['cub_v1'])+r' CUB pairs). Higher AUROC and AP are better.}',
        r'\label{tab:rq3-target}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Attribution target & Pet AUROC & Pet AP & CUB AUROC & CUB AP \\',r'\midrule']
    csv_rows=[]
    for mode in MODES:
        values=[summary[name]['three_way_common']['ranking'][mode+'_HRCAM'][metric] for name in DATASETS for metric in ('auroc','ap')]
        table.append('| '+LABELS[mode]+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
        tex.append(LABELS[mode]+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
        for name in DATASETS:
            r=summary[name]['three_way_common'];entry=r['ranking'][mode+'_HRCAM']
            csv_rows.append({'dataset':name,'convention':mode,'common_pairs':r['n_pairs'],'common_sources':r['n_sources'],
                'auroc':entry['auroc'],'auroc_ci_low':entry['auroc_ci95'][0],'auroc_ci_high':entry['auroc_ci95'][1],
                'ap':entry['ap'],'ap_ci_low':entry['ap_ci95'][0],'ap_ci_high':entry['ap_ci95'][1]})
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (run/'table8_three_conventions.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    (run/'table8_three_conventions.tex').write_text('\n'.join(tex),encoding='utf-8')
    with (run/'table8_three_conventions_with_ci.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    lines=['# R2C3: fixed-ground-truth diagnostic','',
        '**Completed and independently audited. These are standalone review artifacts; the submitted manuscript is unchanged.**','',
        'Original fitted ResNet-18 models, original Pet/CUB source identities and masks, and the same black/gray/white '
        'replacement probes. No training, component selection, recalibration, natural-background extension, or R1C2/R1C3 '
        'additional study was performed in this run.','',*table[2:],'',
        '## Target definitions','',
        '- Own-class explains f(I) on the source and f(I\') on the follow-up.',
        '- Fixed-source explains f(I) on both images.',
        '- Fixed ground-truth explains the original annotated class y on both images, including source-incorrect cases.','',
        '## Coverage and the original Table 8','',
        '| Dataset | Convention | Mask-eligible pairs | Convention-valid pairs | Invalid pairs | Valid sources |',
        '|---|---|---:|---:|---:|---:|']
    for name,label in DATASETS.items():
        for mode in MODES:
            c=summary[name]['coverage'][mode]
            lines.append(f"| {label} | {LABELS[mode]} | {c['mask_eligible_pairs']} | {c['valid_pairs']} | {c['invalid_pairs']} | {c['valid_sources']} |")
        result=summary[name]
        lines += ['',f"{label}: the original own/fixed-source matched set has {result['two_way_original_Table8_reproduction']['n_pairs']} pairs. "
            f"The three-way comparison excludes another {result['three_way_excluded_from_original_two_way']} pairs "
            f"({result['three_way_excluded_consistent']} consistent, {result['three_way_excluded_inconsistent']} inconsistent)."]
    lines += ['', 'The proposed three-row table recomputes every convention on the three-way intersection. '
        'If these numbers differ from the submitted Table 8, changed coverage must be disclosed; Table 3 remains on its '
        'original own-class valid set. Convention-specific all-valid estimates and the original two-way Table 8 '
        'reproduction are preserved in [summary.json](analysis/summary.json).','',
        '## H-RCAM confidence intervals on the three-way common set','',
        '| Dataset | Convention | AUROC (95% CI) | AP (95% CI) |','|---|---|---:|---:|']
    for name,label in DATASETS.items():
        for mode in MODES:
            r=summary[name]['three_way_common']['ranking'][mode+'_HRCAM']
            lines.append('| {} | {} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} [{:.4f}, {:.4f}] |'.format(
                label,LABELS[mode],r['auroc'],*r['auroc_ci95'],r['ap'],*r['ap_ci95']))
    lines += ['', 'Intervals use 1000 paired source-cluster bootstrap resamples, seed 20261011, retaining the available '
        'black/gray/white pairs together. Pair outcomes and the source draw are shared across conventions and components. '
        'They describe source-sampling uncertainty conditional on these classifiers/probes and the valid-map rules. '
        'No pair-level independence p-values are reported.','',
        '## Source-correctness diagnostic','',
        '| Dataset | Source stratum | Common sources | Common pairs | Own AUROC | Fixed-source AUROC | GT AUROC |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,label in DATASETS.items():
        for stratum,r in summary[name]['source_correctness_strata_common'].items():
            values=[r['ranking'].get(mode+'_HRCAM',{}).get('auroc') for mode in MODES]
            lines.append('| {} | {} | {} | {} | {} |'.format(label,stratum,r['n_sources'],r['n_pairs'],
                ' | '.join('not estimable' if v is None else f'{v:.4f}' for v in values)))
    lines += ['', 'When the source prediction is correct, the GT target equals the fixed-source target. Their maps/scores '
        'therefore agree; the third convention adds information on source-incorrect images. Ground-truth maps can be '
        'invalid for some images, so the source-correctness coverage breakdown is also retained in summary.json. '
        'Own/fixed-source scores agree on prediction-consistent pairs. Ground-truth scores need not agree with them '
        'on prediction-consistent but source-incorrect pairs.','',
        '## Component diagnostics on the same common set','',
        '| Convention | Measure | Pet AUROC | Pet AP | CUB AUROC | CUB AP |','|---|---|---:|---:|---:|---:|']
    for mode in MODES:
        for method in ('JSD','NCD','HRCAM'):
            values=[summary[name]['three_way_common']['ranking'][mode+'_'+method][metric] for name in DATASETS for metric in ('auroc','ap')]
            lines.append('| '+LABELS[mode]+' | '+method+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '## Interpretation for the revision','',
        'The practical argument is lack of reference labels at assessment time. This supports prediction-based targeting; '
        'both own-class and fixed-source conventions are label-free. The additional reason for own-class is that it '
        'explains the actual decision for each image. This is a different question from evidence stability for a fixed class. '
        'Its target choice depends on predictions and, when predictions differ, the comparison also changes the explained class.','',
        'The GT experiment is a labelled evaluation diagnostic, not a requirement for using H-RCAM on unlabelled inputs. '
        'It does not remove the need to acknowledge prediction-dependent targeting, and its observed results should not '
        'be used to select a preferred target convention after the fact. The R1C2 study of why AUROC changes and the R1C3 '
        'predictive-reliability validation remain separate, unexecuted proposals.','',
        '## Verification and reproducibility','',
        'All original checkpoints, records, splits, and cached maps are preserved and hash-checked. Original source and '
        'follow-up own-class maps are replayed in their exact original order/batch sizes. Applicable GT/fixed-source and '
        'GT/own-class map identities are checked during extraction. An independent audit uses closed-form Grad-CAM '
        '(GAP + linear head), SciPy JSD/Pearson formulas, pure-Python AUROC/AP ranks and source-bootstrap interval '
        'reconstruction. See [audit](INDEPENDENT_AUDIT.json), [extraction metadata](EXTRACTION_STATUS.json), and '
        '[analysis](analysis/summary.json).','',
        'Remote host: remote-5080 via Tailscale 100.74.145.98. CPUs 24-31 only; four numerical/PyTorch threads. '
        'Large GT maps remain remote with recorded hashes. No manuscript changes were made.','']
    (run/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Standalone Table 8 extension and report written',flush=True)


if __name__=='__main__':main()
