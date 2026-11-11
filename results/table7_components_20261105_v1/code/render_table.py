"""Export full data tables, a correlation scatterplot and component-ratio ECDF."""
import argparse,csv,json,math
from pathlib import Path
from common import bind,read_rows
from calculate import RULES,NAMES

p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();bind()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
assert json.loads((args.run/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
fusion=json.loads((args.run/'fusion_summary.json').read_text());corr=json.loads((args.run/'component_summary.json').read_text())
def csvwrite(name,rows):
    with (args.run/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
for name,d in fusion.items():
    for rule,e in d['rules'].items():
        contrast=e.get('H_minus_rule_auroc',{})
        rows.append({'dataset':d['dataset'],'rule':rule,'delta_median':e['delta_median'],
            'delta_median_ci_low':e['delta_median_ci95'][0],'delta_median_ci_high':e['delta_median_ci95'][1],
            'auroc':e['auroc'],'auroc_ci_low':e['auroc_ci95'][0],'auroc_ci_high':e['auroc_ci95'][1],
            'H_minus_rule_auroc':contrast.get('difference'),'difference_ci_low':contrast.get('ci95',[None,None])[0],
            'difference_ci_high':contrast.get('ci95',[None,None])[1]})
csvwrite('Table7_with_ci.csv',rows)
csvwrite('component_correlations.csv',[{'case':name,**{key:e[key] for key in ('dataset','pairs','pearson_JSD_NCD','spearman_JSD_NCD','spearman_H_JSD','spearman_H_NCD')}} for name,e in corr.items()])
ratio_rows=[]
for name,e in corr.items():
    row={key:e[key] for key in ('dataset','pairs','JSD_smaller_fraction','NCD_smaller_fraction','equal_fraction','JSD_zero_count','NCD_zero_count','both_zero_count','H_zero_fraction')}
    for ratio in ('JSD_over_NCD','min_over_max'):
        for key in ('valid','zero_denominator','fraction_le_0_1'):row[ratio+'_'+key]=e[ratio][key]
        for percentile,value in zip((5,25,50,75,95),e[ratio]['quantiles']):row[ratio+'_q'+str(percentile)]=value
    ratio_rows.append(row)
csvwrite('component_ratios.csv',ratio_rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':10,
    'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
allrows={name:read_rows(args.run/(name+'_component_scores.jsonl')) for name in NAMES}
maxj=max(r['JSD'] for rows in allrows.values() for r in rows);maxc=max(r['NCD'] for rows in allrows.values() for r in rows)
xlim=max(.1,math.ceil(maxj*20)/20);ylim=max(.1,math.ceil(maxc*20)/20)
fig,axes=plt.subplots(2,3,figsize=(7.5,5.2),sharex=True,sharey=True)
colors=('#377eb8','#e17c23')
for letter,name,ax in zip('abcdef',NAMES,axes.flat):
    rows=allrows[name];j=np.array([r['JSD'] for r in rows]);c=np.array([r['NCD'] for r in rows]);y=np.array([r['inconsistent'] for r in rows],bool);e=corr[name]
    for group,color in zip((False,True),colors):
        ax.scatter(j[y==group],c[y==group],s=6,color=color,alpha=.32,linewidths=0,rasterized=True,zorder=2)
    diagonal=min(xlim,ylim);ax.plot([0,diagonal],[0,diagonal],ls='--',lw=.9,color='#666666',zorder=1)
    ax.set(xlim=(0,xlim),ylim=(0,ylim));ax.grid(alpha=.15,lw=.5)
    ax.set_title(f'({letter}) '+e['dataset'],loc='left',pad=7)
    ax.text(.035,.965,f"n = {len(rows):,} pairs\nPearson r = {e['pearson_JSD_NCD']:.3f}\nSpearman "+r'$\rho$'+f" = {e['spearman_JSD_NCD']:.3f}",
        transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':3})
for ax in axes[:,0]:ax.set_ylabel('NCD')
for ax in axes[-1,:]:ax.set_xlabel('JSD')
handles=[Line2D([],[],marker='o',ls='',markersize=5,color=colors[0],label='Prediction-consistent'),
    Line2D([],[],marker='o',ls='',markersize=5,color=colors[1],label='Prediction-inconsistent'),
    Line2D([],[],ls='--',lw=.9,color='#666666',label='JSD = NCD')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.012))
fig.subplots_adjust(left=.065,right=.99,bottom=.14,top=.95,wspace=.16,hspace=.3)
for ext in ('png','pdf'):fig.savefig(args.run/('Figure_JSD_NCD_correlation.'+ext),dpi=320)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4.5));palette=plt.get_cmap('tab10').colors
for i,name in enumerate(NAMES):
    values=sorted(min(r['JSD'],r['NCD'])/max(r['JSD'],r['NCD']) for r in allrows[name] if max(r['JSD'],r['NCD'])>0)
    y=np.arange(1,len(values)+1)/len(values)
    ax.step(np.r_[0,values,1],np.r_[0,y,1],where='post',color=palette[i],lw=1.5,label=corr[name]['dataset'])
ax.axvline(.1,ls='--',lw=1,color='#555555');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Smaller component / larger component',ylabel='Cumulative fraction of pairs')
ax.grid(alpha=.18);ax.text(.115,.08,'0.1: at least tenfold imbalance',fontsize=8,color='#555555')
ax.legend(loc='lower right',fontsize=8,framealpha=.95);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(args.run/('Figure_component_balance.'+ext),dpi=320)
plt.close(fig)
print('Tables and both publication figure candidates exported',flush=True)
