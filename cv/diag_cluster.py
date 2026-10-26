import numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
from cv_pipeline import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
tr=pd.read_csv('/home/claude/data/ntl/train.csv')
ps=sorted(tr.prompt_id.unique())[:30]; te=tr[tr.prompt_id.isin(ps)].reset_index(drop=True)
v=TfidfVectorizer(analyzer='word',stop_words='english',sublinear_tf=True,max_features=50000); X=normalize(v.fit_transform(te.text))
S=(X@X.T).toarray(); D=1-S; g=te.prompt_id.to_numpy(dtype=object)
same=D[g[:,None]==g[None,:]]; same=same[same>1e-9]; diff=D[g[:,None]!=g[None,:]]
print('within-prompt cosine dist: pcts', np.percentile(same,[5,50,95,99]).round(3)); print('between-prompt: pcts', np.percentile(diff,[0.1,1,5,50]).round(3))
Z=linkage(squareform(D,checks=False),method='average')
for t in [0.5,0.6,0.7,0.8,0.85,0.9]:
    lab=fcluster(Z,t=t,criterion='distance'); s=pd.Series(lab).value_counts()
    pur=np.mean([pd.Series(g[lab==c]).value_counts().iloc[0]/(lab==c).sum() for c in np.unique(lab)])
    print(f'cut {t}: clusters={len(s)} sizes min/med/max={s.min()}/{s.median()}/{s.max()} purity={pur:.3f}')
# scenario: random subset of models per prompt (6..18) -> forced N/18 clustering
rng=np.random.default_rng(0)
keep=np.concatenate([rng.choice(np.where(g==p)[0], size=rng.integers(6,19), replace=False) for p in ps])
sub=te.iloc[keep].reset_index(drop=True); gs=sub.prompt_id.to_numpy(dtype=object)
lab=recover_prompts(sub.text.values, n_groups=int(round(len(sub)/18)))
pur=np.mean([pd.Series(gs[lab==c]).value_counts().iloc[0]/(lab==c).sum() for c in np.unique(lab)])
print('subset scenario: rows',len(sub),'true prompts',len(ps),'forced clusters',len(np.unique(lab)),'purity',round(pur,3))
