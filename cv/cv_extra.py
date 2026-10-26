"""CV: does the 24k extra pool help the CPU model? Train = train folds (+ extra), eval = held-out prompts."""
import numpy as np, pandas as pd, time, sys
sys.path.insert(0,'.')
from cv_pipeline import *
tr = pd.read_csv('/home/claude/data/ntl/train.csv')
ex = pd.read_parquet('/home/claude/data/ntl_extra/nlm_matched_clean.parquet')
labels = sorted(tr.model.unique()); lab2i = {l: i for i, l in enumerate(labels)}
y = tr.model.map(lab2i).values; groups = tr.prompt_id.values
ye = ex.model.map(lab2i).values
# recover pseudo-prompts for the extra pool (for centring); cap by cosine clustering in chunks
t0=time.time()
rng=np.random.default_rng(0)
# cluster extra in chunks of 3000 to keep pdist cheap
exg=np.empty(len(ex),dtype=object); base=0
order=rng.permutation(len(ex))
for s in range(0,len(ex),3000):
    idx=order[s:s+3000]
    lab=recover_prompts(ex.text.values[idx], n_groups=max(1,len(idx)//18))
    for i,l in zip(idx,lab): exg[i]=f'x{base+l}'
    base+=lab.max()+1
sizes=pd.Series(exg).value_counts(); print('extra clusters', len(sizes), 'size dist', sizes.describe().to_dict(), f'{time.time()-t0:.0f}s', flush=True)
def softmax_lp(lp):
    e=np.exp(lp-lp.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
gkf = GroupKFold(n_splits=5); res={}
for f,(tri,tei) in enumerate(gkf.split(tr,y,groups)):
    if f>=3: break
    te_text=tr.text.values[tei]; yt=y[tei]
    rec=recover_prompts(te_text, n_groups=len(np.unique(groups[tei])))
    trn_text=np.concatenate([tr.text.values[tri], ex.text.values]); ytr=np.concatenate([y[tri], ye]); gtr=np.concatenate([groups[tri], exg])
    w=np.concatenate([np.full(len(tri),1.0), np.full(len(ex),0.5)])
    Xtr,Xte=build_features(trn_text, te_text, max_char=16000, max_word=8000)
    out={}
    m=LogisticRegression(C=10,max_iter=2000).fit(Xtr,ytr,sample_weight=w); out['raw_ex']=m.predict_log_proba(Xte)
    Xtr_c=centre_by_group(Xtr.astype(np.float32),gtr); Xte_c=centre_by_group(Xte.astype(np.float32),rec)
    for C in [10,30]:
        m=LogisticRegression(C=C,max_iter=2000).fit(Xtr_c,ytr,sample_weight=w); out[f'cen_ex_c{C}']=m.predict_log_proba(Xte_c)
    out['blend']=0.75*out['cen_ex_c30']+0.25*out['raw_ex']
    for k,lp in out.items():
        P=sinkhorn(softmax_lp(lp)); lpb=np.log(P+1e-12)
        res.setdefault(k,[]).append(((lp.argmax(1)==yt).mean(), (group_assign(lpb,rec)==yt).mean()))
    print('fold',f,{k:(round(v[-1][0],3),round(v[-1][1],3)) for k,v in res.items()}, f'{time.time()-t0:.0f}s', flush=True)
for k,v in res.items():
    v=np.array(v); print(f'{k:14s} argmax={v[:,0].mean():.4f} hung={v[:,1].mean():.4f}')
