import argparse,csv,json
from pathlib import Path
from common import bind,read_rows
from diagnostic import COMPONENTS,NAMES

p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--run',type=Path,required=True)
args=p.parse_args();bind()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
assert json.loads((args.run/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
summary=json.loads((args.run/'summary.json').read_text());labels={'pets_v1':'Oxford-IIIT Pet','cub_v1':'CUB-200-2011'}
def csvwrite(name,rows):
    with (args.run/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
component_rows=[];contribution_rows=[]
for name,d in summary.items():
    for component,e in d['components'].items():
        r={'dataset':name,'component':component}
        for mode in ('own','fixed'):
            for metric in ('auroc','ap'):
                r[mode+'_'+metric]=e[mode][metric];r[mode+'_'+metric+'_ci_low']=e[mode][metric+'_ci95'][0];r[mode+'_'+metric+'_ci_high']=e[mode][metric+'_ci95'][1]
        for metric,v in e['own_minus_fixed'].items():
            r['own_minus_fixed_'+metric]=v['difference'];r[metric+'_difference_ci_low']=v['ci95'][0];r[metric+'_difference_ci_high']=v['ci95'][1]
        component_rows.append(r)
        for group,levels in e['contributions'].items():
            for level,v in levels.items():contribution_rows.append({'dataset':name,'component':component,'partition':group,'stratum':level,
                'changed_pairs':v['n_changed_pairs'],'positive_share':v['positive_share'],'mean_credit_shift':v['mean_credit_shift'],
                'global_auroc_contribution':v['contribution'],'contribution_ci_low':v['ci95'][0],'contribution_ci_high':v['ci95'][1]})
csvwrite('component_target_comparisons.csv',component_rows);csvwrite('rank_contributions.csv',contribution_rows)
data={name:read_rows(args.run/(name+'_diagnostic_scores.jsonl')) for name in NAMES}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7.5,3.5),sharex=True,sharey=True)
maximum=max(max(r['own_HRCAM'],r['fixed_HRCAM']) for rows in data.values() for r in rows)*1.03
colors=('#377eb8','#e17c23')
for name,ax in zip(NAMES,axes):
    rows=data[name];y=np.array([r['inconsistent'] for r in rows],bool);x=np.array([r['fixed_HRCAM'] for r in rows]);v=np.array([r['own_HRCAM'] for r in rows])
    for outcome,color in zip((False,True),colors):ax.scatter(x[y==outcome],v[y==outcome],s=7,alpha=.45,color=color,linewidths=0,rasterized=True)
    ax.plot([0,maximum],[0,maximum],'--',color='#666666',lw=.8);ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Fixed-source H-RCAM',title=labels[name])
    ax.grid(alpha=.15);ax.text(.97,.03,f"Consistent: {(~y).sum():,}\nChanged: {y.sum():,}",transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
axes[0].set_ylabel('Own-class H-RCAM')
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l) for c,l in zip(colors,('Prediction-consistent','Prediction-inconsistent'))],
    loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.005))
fig.subplots_adjust(left=.09,right=.98,top=.89,bottom=.23,wspace=.12)
for ext in ('png','pdf'):fig.savefig(args.run/('Figure_own_fixed_scores.'+ext),dpi=320)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7.5,3.5),sharey=True)
for name,ax in zip(NAMES,axes):
    rows=[r for r in data[name] if r['inconsistent']];x=np.array([r['target_HRCAM'] for r in rows]);y=np.array([r['credit_shift_HRCAM'] for r in rows])
    ax.scatter(x,y,s=8,alpha=.45,color='#7b61a8',linewidths=0,rasterized=True);ax.axhline(0,color='#555555',ls='--',lw=.8)
    ax.set(xlabel='Same-image target discrepancy (H-RCAM)',ylim=(-1.03,1.03),title=labels[name]);ax.grid(alpha=.15)
    rho=summary[name]['components']['HRCAM']['changed_associations']['spearman_target_rank_shift']
    ax.text(.97,.03,f"Changed pairs: {len(rows):,}\nSpearman "+r'$\rho$'+f" = {rho:.3f}",transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
axes[0].set_ylabel('Own minus fixed rank credit\nagainst common consistent pairs')
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(args.run/('Figure_target_gap_rank_shift.'+ext),dpi=320)
plt.close(fig)
print('Component tables, exact rank contributions and both diagnostic figures exported',flush=True)
