"""Standalone method/layer sensitivity report and tables."""
import argparse
import csv
import json
from pathlib import Path
from common import CONFIGS,read_rows

LABELS={'GradCAM_layer4':'Grad-CAM / layer4 (original)',
        'GradCAM_layer3_0':'Grad-CAM / layer3[0]',
        'LayerCAM_layer4':'LayerCAM / layer4','LayerCAM_layer3_0':'LayerCAM / layer3[0]'}
DATASETS={'pets_v1':'Oxford-IIIT Pet','cub_v1':'CUB-200-2011'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    run=args.run;summary=json.loads((run/'analysis/summary.json').read_text())
    audit=json.loads((run/'INDEPENDENT_AUDIT.json').read_text());assert audit['state']=='passed'
    table=['# R1C6: attribution-method/layer comparison','',
        'Own-class targets; common valid pairs across all four configurations within each dataset.','',
        '| Configuration | Pet AUROC | Pet AP | CUB AUROC | CUB AP |','|---|---:|---:|---:|---:|']
    tex=[r'\begin{table}[t]',r'\centering',
        r'\caption{H-RCAM sensitivity to attribution method and layer under own-class targeting on common valid pairs. '
        r'The original classifier, source images, masks, probes and score definition are fixed.}',
        r'\label{tab:method-layer-sensitivity}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Configuration & Pet AUROC & Pet AP & CUB AUROC & CUB AP \\',r'\midrule']
    csv_rows=[]
    for config in CONFIGS:
        values=[summary[name]['targets']['own']['common']['ranking'][config][metric] for name in DATASETS for metric in ('auroc','ap')]
        table.append('| '+LABELS[config]+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
        tex.append(LABELS[config]+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (run/'table_method_layer.md').write_text('\n'.join(table)+'\n')
    (run/'table_method_layer.tex').write_text('\n'.join(tex))
    for name in DATASETS:
        for mode in ('own','fixed'):
            common=summary[name]['targets'][mode]['common']
            for config in CONFIGS:
                entry=common['ranking'][config]
                csv_rows.append({'dataset':name,'target':mode,'configuration':config,'common_pairs':common['pairs'],
                    'common_sources':common['sources'],'auroc':entry['auroc'],'auroc_ci_low':entry['auroc_ci95'][0],
                    'auroc_ci_high':entry['auroc_ci95'][1],'ap':entry['ap'],'ap_ci_low':entry['ap_ci95'][0],'ap_ci_high':entry['ap_ci95'][1]})
    with (run/'table_with_ci.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    lines=['# R1C6: attribution-method and layer sensitivity','',
        '**Complete and independently audited. No manuscript changes.**','',*table[2:],'',
        '## Setup and boundaries','',
        'The original fitted ResNet-18 classifiers, Pet/CUB source identities, masks, black/gray/white probes, target '
        'conventions, and H-RCAM score formula are unchanged. Grad-CAM and LayerCAM are evaluated at layer4 (7x7) '
        'and layer3[0] (14x14), a 2x2 method/layer comparison selected before results. No layer search, model training, '
        'recalibration, smoothing or cross-layer fusion was performed.','',
        'LayerCAM is a published nonnegative CAM method compatible with the unchanged H-RCAM inputs. It weights '
        'activations by local positive target gradients, while Grad-CAM uses spatially averaged channel gradients. '
        'The earlier layer equals layer3[-2] in ResNet-18; the authors recommend this type of layer choice to reduce '
        'grid effects from immediately following strided convolutions. See the '
        '[authors\' code/guidance](https://github.com/PengtaoJiang/LayerCAM-jittor) and '
        '[LayerCAM paper](https://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf).','',
        '## Common-set H-RCAM estimates and intervals','',
        '| Dataset | Target | Configuration | Pairs | AUROC (95% CI) | AP (95% CI) |','|---|---|---|---:|---:|---:|']
    for name,label in DATASETS.items():
        for mode in ('own','fixed'):
            c=summary[name]['targets'][mode]['common']
            for config in CONFIGS:
                r=c['ranking'][config]
                lines.append('| {} | {} | {} | {} | {:.4f} [{:.4f}, {:.4f}] | {:.4f} [{:.4f}, {:.4f}] |'.format(
                    label,mode,LABELS[config],c['pairs'],r['auroc'],*r['auroc_ci95'],r['ap'],*r['ap_ci95']))
    lines += ['', 'Confidence intervals use 1000 paired source-cluster resamples (seed 20261105), retaining each source\'s '
        'available probes together. All configurations share the same source draws within a comparison. Paired '
        'configuration-minus-original AUROC/AP differences and their CIs are in [summary.json](analysis/summary.json). '
        'These are comparison-wise intervals, not simultaneous guarantees across all comparisons.','',
        '## Coverage and all-valid estimates','',
        '| Dataset | Target | Configuration | Eligible pairs | Valid pairs | Invalid pairs | All-valid AUROC | AP |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for name,label in DATASETS.items():
        for mode in ('own','fixed'):
            for config in CONFIGS:
                t=summary[name]['targets'][mode];c=t['coverage'][config];r=t['all_valid'][config]['ranking']['HRCAM']
                lines.append('| {} | {} | {} | {} | {} | {} | {:.4f} | {:.4f} |'.format(label,mode,LABELS[config],
                    c['mask_eligible_pairs'],c['valid_pairs'],c['invalid_pairs'],r['auroc'],r['ap']))
    lines += ['', 'The original Grad-CAM-layer4 all-valid estimates reproduce the submitted main/target results. '
        'Common-set estimates can differ if another configuration excludes pairs. Table 3 has not been changed.','',
        '## Score/rank agreement with the original configuration','',
        '| Dataset | Target | Alternative | Spearman | Median absolute H difference | 95th-percentile absolute difference |',
        '|---|---|---|---:|---:|---:|']
    for name,label in DATASETS.items():
        for mode in ('own','fixed'):
            for config,r in summary[name]['targets'][mode]['agreement'].items():
                rho='undefined' if r['spearman_with_original'] is None else f"{r['spearman_with_original']:.4f}"
                lines.append('| {} | {} | {} | {} | {:.6f} | {:.6f} |'.format(label,mode,LABELS[config],rho,
                    r['median_absolute_score_difference'],r['p95_absolute_score_difference']))
    lines += ['', 'These descriptive correlations contain no pair-independent p-values. They measure agreement between '
        'method/layer settings, not the separate R2C4 correlation between JSD and NCD. Similar AUROC does not by itself '
        'establish unchanged score scales, case rankings, or attribution maps.','',
        '## Interpretation and revision use','',
        'This bounded comparison directly examines R1C6\'s method/layer sensitivity concern. It supports conclusions '
        'only within the tested two CAM methods, two layers, two datasets, and fixed classifier/probe settings. '
        'It does not prove invariance across arbitrary explanation methods or layers. Report the observed variation '
        'and retain the original Grad-CAM-layer4 setting as the primary study configuration; do not choose a new '
        'default by the highest retrospective AUROC. Placement/RQ numbering is for the author to decide.','',
        'This experiment does not implement the R2C4 L1/L2/SSIM baselines or JSD/NCD redundancy analysis, R1C2\'s '
        'detailed target-switch analysis, or R1C3\'s predictive-failure validation. Table 8\'s completed GT diagnostic '
        'also does not substitute for a predictive-reliability outcome.','',
        '## Audit and files','',
        f"The independent audit verified {audit['point_entries_verified']} AUROC/AP point entries and "
        f"{audit['intervals_verified']} intervals. Original Grad-CAM-layer4 maps/components were replayed against archived "
        'data; method/layer native maps were independently computed with NumPy reference weights, with a GAP+linear-head '
        'gradient check at layer4. Native maps were reconstructed and score validity/components checked with SciPy. '
        'Pure-Python rank counts reconstructed table metrics and source-bootstrap intervals.','',
        '- [Audit](INDEPENDENT_AUDIT.json), [extraction metadata](EXTRACTION_STATUS.json), [full analysis](analysis/summary.json).',
        '- [Standalone table](table_method_layer.md), [TeX](table_method_layer.tex), [all common-set CIs](table_with_ci.csv).',
        '- Native float32 maps are small and retained with source IDs and masks for deterministic full-map reconstruction.',
        '- Tailscale host 100.74.145.98; CPUs 24-31; four numerical/PyTorch threads. No dependency changes.','']
    (run/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Method/layer report written',flush=True)


if __name__=='__main__':main()
