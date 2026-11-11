"""Prepare the R2C3 table and wording proposal from the audited GT run."""
from pathlib import Path
import csv,hashlib,json,re

root=Path.cwd();base=root/'rcam_reproduction/results/r2c3_fixed_gt_20261105_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=base/'table8_three_conventions_with_ci.csv';summary_path=base/'analysis/summary.json'
assert json.loads((base/'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))['state']=='passed'
summary=json.loads(summary_path.read_text(encoding='utf-8'))
with source.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
lookup={(r['dataset'],r['convention']):r for r in rows};assert len(lookup)==6
names=('pets_v1','cub_v1');labels=('Oxford-IIIT Pet','CUB-200-2011')
modes=(('own','Own-predicted class'),('fixed','Fixed source-predicted class'),('ground_truth','Fixed ground-truth class'))
out=root/'rcam_reproduction/results/table8_revision_20261105_v1';out.mkdir(exist_ok=False)
target=root/'Reviews/Table8_revised.md';response=root/'Reviews/R2C3_Table8_response.md'
assert not target.exists() and not response.exists()
manuscript=root/'Entropy-latest/Entropy-clean.tex';text=manuscript.read_text(encoding='utf-8')
assert sha(manuscript)=='197ef6c2f75e2c3423635415b75d870d5763260f556eef39b6fec189e06b20a0'
old=text.split(r'\label{tab:rq3-target}',1)[1].split(r'\end{table}',1)[0]
fixture={}
for line in old.splitlines():
    cells=[v.strip() for v in line.strip().removesuffix(r'\\').split('&')]
    if len(cells)==5 and cells[0] in ('Own-predicted class','Fixed source-predicted class'):
        fixture[cells[0]]=[float(x) for x in cells[1:]]
assert len(fixture)==2
for mode,label in modes:
    if mode=='ground_truth':continue
    assert [round(float(lookup[(n,mode)][metric]),4) for n in names for metric in ('auroc','ap')]==fixture[label]
caption='H-RCAM discrimination of prediction inconsistency under three attribution-target conventions. Higher AUROC and AP are better.'
lines=['# Table 8. Attribution-target sensitivity','',caption,'',
    '| Attribution target | Oxford-IIIT Pet AUROC | Oxford-IIIT Pet AP | CUB-200-2011 AUROC | CUB-200-2011 AP |',
    '|---|---:|---:|---:|---:|']
for mode,label in modes:
    cells=[]
    for name in names:
        e=lookup[(name,mode)]
        for metric in ('auroc','ap'):
            rank=summary[name]['three_way_common']['ranking'][mode+'_HRCAM']
            assert [float(e[metric]),float(e[metric+'_ci_low']),float(e[metric+'_ci_high'])]==[rank[metric],*rank[metric+'_ci95']]
            cells.append('{:.4f}<br>[{:.4f}, {:.4f}]'.format(float(e[metric]),float(e[metric+'_ci_low']),float(e[metric+'_ci_high'])))
    lines.append('| '+label+' | '+' | '.join(cells)+' |')
lines+=['','**Notes.**','',
    '1. Every row uses the same three-way matched valid pairs: 2,979 Pet pairs from 993 sources and '
    '2,887 CUB pairs from 963 sources. Original ResNet-18, masks, black/gray/white probes and score formula are unchanged.',
    '2. Own-predicted class explains the source prediction on the source and the follow-up prediction on the '
    'follow-up. Fixed source-predicted class explains the source prediction on both images. Fixed ground-truth '
    'class explains the original annotated class on both images, including cases where the source prediction is incorrect.',
    '3. Each estimate is above its 95% source-image-cluster bootstrap CI (1,000 resamples, seed 20261011). '
    'Available background pairs from each sampled source remain together and the same source draws are used across conventions. '
    'These intervals are reused from the independently checked fixed-GT experiment.',
    '4. Both submitted own/fixed-source rows reproduce at four decimals. Two CUB pairs have invalid fixed-source maps '
    'and remain excluded from all three rows; GT targeting causes no additional matched-pair exclusions. '
    'This accounts for the matched CUB own-class AUROC of 0.8067 versus 0.8066 in Table 3.',
    '5. Own-class and fixed-source conventions are both label-free. Own-class is retained to characterize changes '
    'between explanations of actual decisions. Its target selection depends on predictions and changes class when '
    'predictions switch. The fixed-GT row is a labelled evaluation diagnostic, not a requirement for applying H-RCAM.',
    '6. Target conventions answer different questions. Their observed AUROC/AP ordering should not be used to select '
    'the preferred convention after observing the results. When the source prediction is correct, fixed-source and '
    'fixed-GT targets coincide; they can differ for source-incorrect images.','',
    'Standalone review draft for R2C3. The original wording and numeric rows are retained where applicable. '
    'No manuscript edits or new experiments were made to prepare this table. The separate R1C2 target-switch '
    'analysis examines why discrimination changes.','']
target.write_text('\n'.join(lines),encoding='utf-8');(out/target.name).write_bytes(target.read_bytes())
response_text='''# R2C3: Table 8 and attribution-target wording proposal

**Status:** author-review draft. The submitted manuscript is unchanged.
The response below describes the proposed completed revision and should
be used after the table and accompanying wording are integrated.

## Proposed response

We have retained own-predicted-class targeting as the primary convention
because the intended diagnostic compares explanations of the model's actual
decision for each image. We agree that the absence of reference labels alone
does not distinguish it from fixed-source targeting: both are label-free.
We now explicitly acknowledge that own-class target selection depends on
the predictions and changes the explained class when predictions differ.
Accordingly, its score characterizes joint changes in input and explained
decision, rather than isolating fixed-class attribution stability.

We have expanded Table 8 with a fixed-ground-truth convention, using the
original annotated class for both source and follow-up, including initially
incorrect predictions. All three rows use identical matched valid pairs
(2,979 Pet and 2,887 CUB pairs); the original two rows are preserved.
Source-cluster CIs and invalid-map coverage are disclosed. We have added
the target definitions and rationale in Section 3.3 and qualified the
abstract's interpretation. The ground-truth convention is an evaluation
diagnostic requiring labels, not an operational requirement or a
post-result selection of a preferred target.

## Proposed local addition to Section 3.3 — not applied

Retain the current target-map definitions and the sentence explaining that
different predictions imply different explained classes. Add:

> Own-predicted-class targeting is used because the primary analysis
> compares explanations of the actual decision made for each image.
> Both own-class and fixed-source-class targeting can be used without
> reference labels. Under own-class targeting, the target choice is
> prediction-dependent; its construction is therefore not independent of
> the predictions used to define consistency. When predictions change,
> the score can reflect both the background replacement and the change
> in explained class. Fixed-source and fixed-ground-truth comparisons in
> RQ3.3 provide diagnostics with the target held constant.

## Proposed abstract qualifier — not applied

> Under own-predicted-class targeting, H-RCAM characterizes changes between
> explanations of actual model decisions; its discrimination depends on
> the attribution-target convention.

This is an addition/qualification, not a rewrite of the abstract. Placement
should be settled with the author alongside the other abstract revisions.

## Proposed RQ3.3 changes — not applied

Change “two ways” to “three ways” and retain the existing own/fixed-source
definitions. Add:

> Under the fixed-ground-truth convention, both maps explain the original
> annotated class, including cases where the source prediction is
> incorrect.

Use the revised Table 8. Retain the matched-pair coverage explanation and
state that GT introduces no further exclusions. Keep the explanation that
own/fixed-source scores agree for prediction-consistent pairs; GT scores
need not equal them for source-incorrect, prediction-consistent cases.
When source predictions are correct, GT and fixed-source targets agree.

The more detailed R1C2 analysis of changed-class maps and score ordering
is a separate analysis. Neither this GT row nor that analysis alone
validates H-RCAM against independent predictive-reliability outcomes
(R1C3). Table 3's evaluation structure remains unchanged.
'''
response.write_text(response_text,encoding='utf-8');(out/response.name).write_bytes(response.read_bytes())
tex=[r'\documentclass[10pt,a4paper]{article}',r'\usepackage[margin=20mm]{geometry}',r'\usepackage{booktabs}',
    r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',
    r'\newcommand{\cicell}[2]{\shortstack[c]{#1\\[-1pt]{\fontsize{8}{9}\selectfont #2}}}',
    r'\begin{document}',r'\setcounter{table}{7}',r'\begin{table}[htbp]',r'\caption{'+caption+r'}',
    r'\label{tab:rq3-target}',r'\centering',r'\fontsize{9}{10}\selectfont',r'\setlength{\tabcolsep}{4pt}',
    r'\renewcommand{\arraystretch}{1.08}',r'\begin{tabular}{lcccc}',r'\toprule',
    r'& \multicolumn{2}{c}{Oxford-IIIT Pet} & \multicolumn{2}{c}{CUB-200-2011}\\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',r'Attribution target & AUROC & AP & AUROC & AP\\',r'\midrule']
for mode,label in modes:
    cells=[]
    for name in names:
        e=lookup[(name,mode)]
        for metric in ('auroc','ap'):
            cells.append(r'\cicell{'+'{:.4f}'.format(float(e[metric]))+'}{'+'[{:.4f}, {:.4f}]'.format(float(e[metric+'_ci_low']),float(e[metric+'_ci_high']))+'}')
    tex.append(' & '.join([label,*cells])+r'\\[3pt]')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\noindent\footnotesize ',
    r'All rows use identical matched valid pairs: 2,979 Pet and 2,887 CUB pairs, from 993 and 963 sources. '
    r'Two invalid fixed-source CUB maps remain excluded from all rows; GT adds no exclusions. '
    r'95\% CIs use 1,000 paired source-cluster resamples, seed20261011. Both prediction-based conventions are label-free; '
    r'GT requires reference labels. Target conventions answer different questions and are not selected by these rankings. '
    r'Original models, probes and masks unchanged. Standalone review draft; manuscript unchanged.',r'\end{document}']
(out/'table8_preview.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
record={'state':'passed','original_point_entries_preserved':8,'added_GT_point_entries':4,'exactly_reused_audited_intervals':12,
    'source_sha256':{str(p.relative_to(root)):sha(p) for p in (source,summary_path)},
    'review_files':{str(p.relative_to(root)):sha(p) for p in (target,response)},'manuscript_edited':False,'new_experiments':False}
(out/'TABLE8_CHECK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
