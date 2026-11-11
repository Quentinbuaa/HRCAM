import argparse,bisect,json,math,statistics,time
from pathlib import Path
from common import bind,sha,write_json,read_rows,rank_metrics
from diagnostic import COMPONENTS,NAMES,GROUPS,load_maps,group_value

def evaluate(rows):
    out={};positive=[r for r in rows if r['inconsistent']]
    for c in COMPONENTS:
        for mode in ('own','fixed'):
            for metric,value in rank_metrics(rows,mode+'_'+c).items():out[mode+'_'+c+'_'+metric]=value
        for metric in ('auroc','ap'):out['difference_'+c+'_'+metric]=out['own_'+c+'_'+metric]-out['fixed_'+c+'_'+metric]
        negative=sorted(r['own_'+c] for r in rows if not r['inconsistent'])
        assert negative==sorted(r['fixed_'+c] for r in rows if not r['inconsistent'])
        def credit(value):return (bisect.bisect_left(negative,value)+bisect.bisect_right(negative,value))/(2*len(negative))
        shifts=[credit(r['own_'+c])-credit(r['fixed_'+c]) for r in positive]
        assert abs(math.fsum(shifts)/len(shifts)-out['difference_'+c+'_auroc'])<2e-12
        for group,levels in GROUPS.items():
            for level in levels:
                out['contribution_'+c+'_'+group+'_'+level]=math.fsum(v for r,v in zip(positive,shifts) if group_value(r,group)==level)/len(shifts)
    return out

def expected(d):
    points={};intervals={}
    for c,e in d['components'].items():
        for mode in ('own','fixed'):
            for metric in ('auroc','ap'):
                key=mode+'_'+c+'_'+metric;points[key]=e[mode][metric];intervals[key]=e[mode][metric+'_ci95']
        for metric in ('auroc','ap'):
            key='difference_'+c+'_'+metric;points[key]=e['own_minus_fixed'][metric]['difference'];intervals[key]=e['own_minus_fixed'][metric]['ci95']
        for group,levels in e['contributions'].items():
            for level,v in levels.items():
                key='contribution_'+c+'_'+group+'_'+level;points[key]=v['contribution'];intervals[key]=v['ci95']
    return points,intervals

def pearson(a,b):
    ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b);aa=[v-ma for v in a];bb=[v-mb for v in b]
    return math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(math.fsum(x*x for x in aa)*math.fsum(y*y for y in bb))

def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);result=[None]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):result[order[k]]=(i+j+1)/2
        i=j
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();bind()
    import numpy as np
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
    saved=json.loads((args.run/'summary.json').read_text());provenance=json.loads((args.run/'PROVENANCE.json').read_text());started=time.time()
    report={'state':'passed','point_entries':0,'intervals':0,'rank_credit_row_checks':0,'map_pair_checks':0,
        'correlations':0,'max_error':0.,'map_max_error':0.,'datasets':{}}
    def close(a,b):
        error=abs(a-b);assert error<2e-12,(a,b,error);report['max_error']=max(report['max_error'],error)
    for name in NAMES:
        rows=read_rows(args.run/(name+'_diagnostic_scores.jsonl'));d=saved[name]
        for path,value in provenance[name]['before'].items():assert sha(Path(path))==value
        points,intervals=expected(d);actual=evaluate(rows);assert set(points)==set(actual) and len(points)==42
        for key,value in actual.items():close(value,points[key]);report['point_entries']+=1
        positive=[r for r in rows if r['inconsistent']];negative=[r for r in rows if not r['inconsistent']]
        for c in COMPONENTS:
            negatives=sorted(r['own_'+c] for r in negative)
            def credit(value):return (bisect.bisect_left(negatives,value)+bisect.bisect_right(negatives,value))/(2*len(negatives))
            for row in rows:
                if row['inconsistent']:
                    for key,value in [('credit_own_'+c,credit(row['own_'+c])),('credit_fixed_'+c,credit(row['fixed_'+c])),
                        ('credit_shift_'+c,credit(row['own_'+c])-credit(row['fixed_'+c]))]:
                        close(row[key],value);report['rank_credit_row_checks']+=1
                else:assert row['own_'+c]==row['fixed_'+c] and row['target_'+c]==0
            gap=[r['target_'+c] for r in positive];score=[r['own_'+c]-r['fixed_'+c] for r in positive];shift=[r['credit_shift_'+c] for r in positive]
            expected_corr=d['components'][c]['changed_associations']
            for key,value in [('pearson_target_score_shift',pearson(gap,score)),('spearman_target_score_shift',pearson(ranks(gap),ranks(score))),
                ('pearson_target_rank_shift',pearson(gap,shift)),('spearman_target_rank_shift',pearson(ranks(gap),ranks(shift)))]:
                close(value,expected_corr[key]);report['correlations']+=1
            for label,group in [('consistent',negative),('inconsistent',positive)]:
                values=[r['target_'+c] for r in group];v=d['components'][c]['target_only'][label]
                assert len(values)==v['n'];close(statistics.fmean(values),v['mean']);close(statistics.median(values),v['median'])
                close(min(values),v['min']);close(max(values),v['max'])
        groups={}
        for r in rows:groups.setdefault(r['source_id'],[]).append(r)
        sources=sorted(groups);rng=np.random.default_rng(20261011);draws={key:[] for key in points}
        for repeat in range(1000):
            sample=[r for i in rng.integers(0,len(sources),len(sources)) for r in groups[sources[int(i)]]]
            for key,value in evaluate(sample).items():draws[key].append(value)
            if repeat%250==0:print(name,'independent source/rank audit',repeat,flush=True)
        for key,values in draws.items():
            for a,b in zip(np.quantile(values,[.025,.975]),intervals[key]):close(float(a),b)
            report['intervals']+=1
        positions,original,maps=load_maps(args.root,name);lookup={(r['source_id'],r['background']):r for r in rows}
        audit_sources={sources[int(k)] for k in np.linspace(0,len(sources)-1,8,dtype=int)}
        for outcome in (False,True):
            audit_sources.update(sorted({r['source_id'] for r in rows if r['inconsistent']==outcome})[:8])
        def scipy_components(a,b):
            a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel();p=a+1e-8;q=b+1e-8
            j=float(jensenshannon(p/p.sum(),q/q.sum(),base=2)**2);c=float(np.clip((1-pearsonr(a,b).statistic)/2,0,1))
            return {'JSD':j,'NCD':c,'HRCAM':2*j*c/(j+c+1e-12)}
        checked=0
        for source in sorted(audit_sources):
            for bg_index,bg in enumerate(('black','gray','white')):
                if (source,bg) not in lookup:continue
                r=lookup[(source,bg)];i=positions[source];a,b,c=original[i],maps[i,bg_index,1],maps[i,bg_index,0]
                for prefix,first,second in [('own',a,c),('fixed',a,b),('target',b,c)]:
                    for component,value in scipy_components(first,second).items():
                        error=abs(value-r[prefix+'_'+component]);assert error<2e-12
                        report['map_max_error']=max(report['map_max_error'],error)
                checked+=1
        report['map_pair_checks']+=checked;report['datasets'][name]={'pairs':len(rows),'sources':len(sources),'map_pairs_checked':checked}
        for path,value in provenance[name]['after'].items():assert sha(Path(path))==value
        del original,maps;print(name,'all independent diagnostics passed',flush=True)
    assert (report['point_entries'],report['intervals'],report['correlations'])==(84,84,24)
    report['seconds']=time.time()-started;write_json(args.run/'INDEPENDENT_AUDIT.json',report)

if __name__=='__main__':main()
