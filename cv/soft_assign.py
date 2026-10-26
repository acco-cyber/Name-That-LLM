"""Is a soft (Sinkhorn-within-cluster) assignment a safer replacement for the hard Hungarian when recovered test clusters
may be impure?  CPU proxy: TF-IDF LR on 120 train prompts, 30 held-out prompts (540 rows) as fake test, 5 folds.
Scenarios: clusters = truth (pure, complete) | truth with pairs of prompts merged (impure, 36 rows) | recovered by the
pipeline's capped avg-linkage at thr 0.15 / 0.08 / 0.04."""
import numpy as np, pandas as pd, scipy.sparse as sp, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import linear_sum_assignment
K = 18
train = pd.read_csv('/home/claude/data/ntl/train.csv'); LABELS = sorted(train.model.unique()); L2I = {l: i for i, l in enumerate(LABELS)}
y_all = train.model.map(L2I).values
def capped_agglo(S, cap=18, thr=0.02):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab
def hungarian(lp, groups, dense_only=True):
    pred = lp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if dense_only and not (12 <= len(idx) <= 18): continue
        if len(idx) > K:   # impure over-size cluster: assign with K copies
            r, c = linear_sum_assignment(-np.tile(lp[idx], (1, int(np.ceil(len(idx) / K))))); pred[idx[r]] = c % K
        else:
            r, c = linear_sum_assignment(-lp[idx]); pred[idx[r]] = c
    return pred
def soft_assign(lp, groups, iters=20, tau=1.0, min_size=2):
    """per cluster: Sinkhorn with column target = n/K each (soft 'each model about once'); then argmax."""
    pred = lp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; n = len(idx)
        if n < min_size: continue
        P = np.exp((lp[idx] - lp[idx].max(1, keepdims=True)) / tau); P /= P.sum(1, keepdims=True)
        col = np.full(K, n / K)
        for _ in range(iters):
            P *= col / np.maximum(P.sum(0), 1e-12); P /= P.sum(1, keepdims=True)
        pred[idx] = P.argmax(1)
    return pred
def sinkhorn_global(lp, iters=50):
    P = np.exp(lp - lp.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); t = np.full(K, len(P) / K)
    for _ in range(iters): P *= t / np.maximum(P.sum(0), 1e-12); P /= P.sum(1, keepdims=True)
    return np.log(P + 1e-12)

rng = np.random.RandomState(0); pids = np.array(sorted(train.prompt_id.unique())); rng.shuffle(pids)
res = {}
for fold in range(5):
    hold = np.isin(train.prompt_id.values, pids[fold::5])
    tr, te = train[~hold], train[hold]; yt, ye = y_all[~hold], y_all[hold]
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000, sublinear_tf=True, lowercase=False)
    vw = TfidfVectorizer(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True, lowercase=False)
    Xtr = sp.hstack([vc.fit_transform(tr.text), vw.fit_transform(tr.text)]).tocsr(); Xte = sp.hstack([vc.transform(te.text), vw.transform(te.text)]).tocsr()
    lp = LogisticRegression(C=20, max_iter=3000).fit(Xtr, yt).predict_log_proba(Xte)
    lp = sinkhorn_global(lp)
    truth_g = pd.factorize(te.prompt_id.values)[0]
    # impure: merge prompt pairs -> clusters of 36 (over-merge), then split arbitrarily into two 18s (what a wrong cut does)
    pair = truth_g // 2
    impure = pair * 2 + (np.arange(len(te)) % 2)        # two mixed 18-row clusters per pair
    v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05); Xc = v.fit_transform(te.text.values); S = (Xc @ Xc.T).toarray()
    scen = {'truth': truth_g, 'impure_pairs': impure}
    for thr in (0.15, 0.08, 0.04): scen[f'rec{thr}'] = capped_agglo(S, 18, thr)
    for name, g in scen.items():
        for meth, pred in [('argmax', lp.argmax(1)), ('hung_dense', hungarian(lp, g)), ('hung_all', hungarian(lp, g, False)),
                           ('soft_t1', soft_assign(lp, g, tau=1.0)), ('soft_t2', soft_assign(lp, g, tau=2.0)), ('soft_t0.5', soft_assign(lp, g, tau=0.5))]:
            res.setdefault((name, meth), []).append(balanced_accuracy_score(ye, pred))
    print('fold', fold, 'done', flush=True)
out = pd.Series({k: np.mean(v) for k, v in res.items()}).unstack()
print(out.round(4).to_string())
