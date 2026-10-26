import numpy as np, pandas as pd, time, sys
sys.path.insert(0,'.')
from cv_pipeline import *
from sklearn.svm import LinearSVC
tr = pd.read_csv('/home/claude/data/ntl/train.csv')
labels = sorted(tr.model.unique()); lab2i = {l: i for i, l in enumerate(labels)}
y = tr.model.map(lab2i).values; groups = tr.prompt_id.values
gkf = GroupKFold(n_splits=5)
res = {}
def softmax_lp(lp):
    e=np.exp(lp-lp.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
t0=time.time()
for f,(tri,tei) in enumerate(gkf.split(tr,y,groups)):
    te_text=tr.text.values[tei]; yt=y[tei]
    rec=recover_prompts(te_text, n_groups=len(np.unique(groups[tei])))
    Xtr,Xte=build_features(tr.text.values[tri], te_text)
    Xtr_c=centre_by_group(Xtr.astype(np.float32),groups[tri]); Xte_c=centre_by_group(Xte.astype(np.float32),rec)
    out={}
    for C in [10,30,100]:
        m=LogisticRegression(C=C,max_iter=3000,n_jobs=-1).fit(Xtr_c,y[tri]); out[f'lr_c{C}']=m.predict_log_proba(Xte_c)
    for C in [0.5,1]:
        s=LinearSVC(C=C,max_iter=5000).fit(Xtr_c,y[tri]); d=s.decision_function(Xte_c); out[f'svc_c{C}']=np.log(softmax_lp(d*3)+1e-12)
    m=LogisticRegression(C=10,max_iter=3000,n_jobs=-1).fit(Xtr,y[tri]); out['lr_raw']=m.predict_log_proba(Xte)
    combos=dict(out)
    for w in [0.2,0.35]:
        combos[f'blend_c30_raw{w}']=(1-w)*out['lr_c30']+w*out['lr_raw']
        combos[f'blend_svc1_raw{w}']=(1-w)*out['svc_c1']+w*out['lr_raw']
    combos['lr30+svc1']=0.5*out['lr_c30']+0.5*out['svc_c1']
    for k,lp in combos.items():
        P=sinkhorn(softmax_lp(lp)); lpb=np.log(P+1e-12)
        res.setdefault(k,[]).append(((lp.argmax(1)==yt).mean(), (group_assign(lpb,rec)==yt).mean()))
    print('fold',f, f'{time.time()-t0:.0f}s', flush=True)
for k,v in sorted(res.items(), key=lambda kv:-np.mean([b for a,b in kv[1]])):
    v=np.array(v); print(f'{k:22s} argmax={v[:,0].mean():.4f}  hung={v[:,1].mean():.4f}')
