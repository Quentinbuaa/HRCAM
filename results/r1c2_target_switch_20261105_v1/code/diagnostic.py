"""Map contrasts and exact positive-versus-negative AUROC accounting."""
COMPONENTS=('JSD','NCD','HRCAM')
NAMES=('pets_v1','cub_v1')
GROUPS={'background':('black','gray','white'),'source_correctness':('correct','incorrect'),
    'transition':('correct_to_incorrect','incorrect_to_correct','incorrect_to_incorrect')}

def components(first,second):
    import numpy as np
    a=np.asarray(first,float).ravel();b=np.asarray(second,float).ravel()
    assert np.isfinite(a).all() and np.isfinite(b).all() and a.min()>=0 and b.min()>=0
    assert a.std()>1e-8 and b.std()>1e-8
    if np.array_equal(a,b):return {'JSD':0.,'NCD':0.,'HRCAM':0.}
    p=(a+1e-8)/(a+1e-8).sum();q=(b+1e-8)/(b+1e-8).sum();m=(p+q)/2
    j=float(np.clip((np.sum(p*np.log(p/m))+np.sum(q*np.log(q/m)))/(2*np.log(2)),0,1))
    c=float(np.clip((1-np.corrcoef(a,b)[0,1])/2,0,1))
    return {'JSD':j,'NCD':c,'HRCAM':2*j*c/(j+c+1e-12)}

def load_maps(root,name):
    import numpy as np
    with np.load(root/'runs'/name/'holdout_heatmaps.npz') as z:
        ids,original,follow=z['source_ids'],z['original_maps'],z['maps']
    return {int(s):i for i,s in enumerate(ids)},original,follow

def transition(row):
    if not row['inconsistent']:return 'consistent'
    if row['source_correct']:
        assert not row['follow_correct'];return 'correct_to_incorrect'
    return 'incorrect_to_correct' if row['follow_correct'] else 'incorrect_to_incorrect'

def group_value(row,key):
    if key=='background':return row['background']
    if key=='source_correctness':return 'correct' if row['source_correct'] else 'incorrect'
    return transition(row)

def credits(negative,positive,negative_weights):
    """Return per-positive half-tie credit for every weighted negative draw."""
    import numpy as np
    order=np.argsort(negative,kind='stable');values=negative[order]
    prefix=np.c_[np.zeros(len(negative_weights)),np.cumsum(negative_weights[:,order],axis=1)]
    lo=np.searchsorted(values,positive,side='left');hi=np.searchsorted(values,positive,side='right')
    return (prefix[:,lo]+prefix[:,hi])/(2*prefix[:,-1,None])
