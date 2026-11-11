import argparse,json,math,statistics,time
from pathlib import Path
from common import bind,sha,write_json,read_rows,rank_metrics
from calculate import RULES,NAMES,QUANTILES

def pearson(a,b):
    ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b)
    aa=[v-ma for v in a];bb=[v-mb for v in b]
    return math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(math.fsum(x*x for x in aa)*math.fsum(y*y for y in bb))

def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);result=[None]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):result[order[k]]=(i+j+1)/2
        i=j
    return result

def quantiles(values):
    v=sorted(values);out=[]
    for q in QUANTILES:
        pos=(len(v)-1)*q;i=int(pos);f=pos-i
        out.append(v[i] if f==0 else v[i]*(1-f)+v[i+1]*f)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind()
    import numpy as np
    started=time.time();fusion=json.loads((args.run/'fusion_summary.json').read_text());corr=json.loads((args.run/'component_summary.json').read_text())
    report={'state':'passed','fusion_point_entries':0,'fusion_intervals':0,'correlations':0,'ratio_statistics':0,'max_error':0.}
    def close(a,b):
        error=abs(a-b);assert error<2e-12,(a,b,error);report['max_error']=max(report['max_error'],error)
    for name in NAMES[:2]:
        rows=read_rows(args.run/(name+'_fusion_scores.jsonl'));saved=fusion[name]['rules'];groups={}
        for r in rows:groups.setdefault(r['source_id'],[]).append(r)
        sources=sorted(groups);rng=np.random.default_rng(20261011)
        draws={rule:{'auroc':[],'delta_median':[]} for rule in RULES}
        for rule in RULES:
            close(rank_metrics(rows,rule)['auroc'],saved[rule]['auroc']);report['fusion_point_entries']+=1
            a=[r[rule] for r in rows if not r['inconsistent']];b=[r[rule] for r in rows if r['inconsistent']]
            close(statistics.median(b)-statistics.median(a),saved[rule]['delta_median']);report['fusion_point_entries']+=1
        for repeat in range(1000):
            sample=[r for i in rng.integers(0,len(sources),len(sources)) for r in groups[sources[int(i)]]]
            for rule in RULES:
                draws[rule]['auroc'].append(rank_metrics(sample,rule)['auroc'])
                a=[r[rule] for r in sample if not r['inconsistent']];b=[r[rule] for r in sample if r['inconsistent']]
                draws[rule]['delta_median'].append(statistics.median(b)-statistics.median(a))
        for rule in RULES:
            for metric in ('auroc','delta_median'):
                for a,b in zip(np.quantile(draws[rule][metric],[.025,.975]),saved[rule][metric+'_ci95']):close(float(a),b)
                report['fusion_intervals']+=1
            if rule!=RULES[-1]:
                expected=np.quantile(np.array(draws[RULES[-1]]['auroc'])-draws[rule]['auroc'],[.025,.975])
                for a,b in zip(expected,saved[rule]['H_minus_rule_auroc']['ci95']):close(float(a),b)
                close(saved[RULES[-1]]['auroc']-saved[rule]['auroc'],saved[rule]['H_minus_rule_auroc']['difference'])
                report['fusion_point_entries']+=1;report['fusion_intervals']+=1
        print(name,'all fusion points and CIs independently verified',flush=True)
    datasets={name:read_rows(args.run/(name+'_component_scores.jsonl')) for name in NAMES}
    datasets['pooled_development']=[r for name in NAMES[:2] for r in datasets[name]]
    datasets['pooled_all_six']=[r for name in NAMES for r in datasets[name]]
    for name,rows in datasets.items():
        e=corr[name];j=[r['JSD'] for r in rows];c=[r['NCD'] for r in rows];h=[r['H-RCAM'] for r in rows]
        for key,value in [('pearson_JSD_NCD',pearson(j,c)),('spearman_JSD_NCD',pearson(ranks(j),ranks(c))),
            ('spearman_H_JSD',pearson(ranks(h),ranks(j))),('spearman_H_NCD',pearson(ranks(h),ranks(c)))]:
            close(value,e[key]);report['correlations']+=1
        flags={'JSD_smaller_fraction':sum(a<b for a,b in zip(j,c))/len(j),'NCD_smaller_fraction':sum(b<a for a,b in zip(j,c))/len(j),
            'equal_fraction':sum(a==b for a,b in zip(j,c))/len(j),'JSD_zero_count':j.count(0),'NCD_zero_count':c.count(0),
            'both_zero_count':sum(a==0 and b==0 for a,b in zip(j,c)),'H_zero_fraction':h.count(0)/len(h)}
        for key,value in flags.items():close(value,e[key]);report['ratio_statistics']+=1
        for key,values in [('JSD_over_NCD',[a/b for a,b in zip(j,c) if b>0]),
            ('min_over_max',[min(a,b)/max(a,b) for a,b in zip(j,c) if max(a,b)>0])]:
            assert len(values)==e[key]['valid'] and len(rows)-len(values)==e[key]['zero_denominator']
            for a,b in zip(quantiles(values),e[key]['quantiles']):close(a,b);report['ratio_statistics']+=1
            close(sum(v<=.1 for v in values)/len(values),e[key]['fraction_le_0_1']);report['ratio_statistics']+=1
    provenance=json.loads((args.run/'PROVENANCE.json').read_text())
    for e in provenance.values():assert sha(Path(e['source']))==e['sha256']
    assert (report['fusion_point_entries'],report['fusion_intervals'],report['correlations'])==(40,40,32)
    report['seconds']=time.time()-started;write_json(args.run/'INDEPENDENT_AUDIT.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
