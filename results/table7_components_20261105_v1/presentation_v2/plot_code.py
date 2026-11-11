import argparse,json,shutil,tarfile,hashlib,os
from pathlib import Path
from common import bind,read_rows
from calculate import NAMES
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();bind()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
assert json.loads((args.run/'INDEPENDENT_AUDIT.json').read_text())['state']=='passed'
corr=json.loads((args.run/'component_summary.json').read_text())
output=args.run/'presentation_v2';output.mkdir(exist_ok=False)
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
    ax.text(.965,.035,f"n = {len(rows):,} pairs\nPearson r = {e['pearson_JSD_NCD']:.3f}\nSpearman "+r'$\rho$'+f" = {e['spearman_JSD_NCD']:.3f}",
        transform=ax.transAxes,ha='right',va='bottom',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':3})
for ax in axes[:,0]:ax.set_ylabel('NCD')
for ax in axes[-1,:]:ax.set_xlabel('JSD')
handles=[Line2D([],[],marker='o',ls='',markersize=5,color=colors[0],label='Prediction-consistent'),
    Line2D([],[],marker='o',ls='',markersize=5,color=colors[1],label='Prediction-inconsistent'),
    Line2D([],[],ls='--',lw=.9,color='#666666',label='JSD = NCD')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.012))
fig.subplots_adjust(left=.065,right=.99,bottom=.14,top=.95,wspace=.16,hspace=.3)
for ext in ('png','pdf'):fig.savefig(output/('Figure_JSD_NCD_correlation.'+ext),dpi=320)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4.5));palette=plt.get_cmap('tab10').colors
for i,name in enumerate(NAMES):
    values=sorted(min(r['JSD'],r['NCD'])/max(r['JSD'],r['NCD']) for r in allrows[name] if max(r['JSD'],r['NCD'])>0)
    y=np.arange(1,len(values)+1)/len(values)
    ax.step(np.r_[0,values,1],np.r_[0,y,1],where='post',color=palette[i],lw=1.5,label=corr[name]['dataset'])
ax.axvline(.1,ls='--',lw=1,color='#555555');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Smaller component / larger component',ylabel='Cumulative fraction of pairs')
ax.grid(alpha=.18);ax.text(.115,.95,'0.1: at least tenfold imbalance',fontsize=8,color='#555555')
ax.legend(loc='lower right',fontsize=8,framealpha=.95);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(output/('Figure_component_balance.'+ext),dpi=320)
plt.close(fig)
print('Tables and both publication figure candidates exported',flush=True)

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copy2(Path(__file__),output/'plot_code.py')
record={'change':'Move scatter statistics to empty lower-right and ratio threshold label above curves; values/points unchanged',
    'component_summary_sha256':sha(args.run/'component_summary.json'),'independent_audit_sha256':sha(args.run/'INDEPENDENT_AUDIT.json'),
    'cpu_affinity':sorted(os.sched_getaffinity(0)),'new_statistics':False,
    'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
(output/'PRESENTATION_CHECK.json').write_text(json.dumps(record,indent=2)+'\n')
archive=args.run/'presentation_v2.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted(output.iterdir()):tar.add(path,arcname=path.name,recursive=False)
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'size':archive.stat().st_size}))
