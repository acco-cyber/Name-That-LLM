"""Name That LLM - CPU-only pipeline with grouped CV.

Key ideas
1. Char + word TF-IDF features, prompt-centred (subtract the per-prompt mean of the
   feature vectors so topic content cancels and only style remains).
2. Linear classifier (LogisticRegression) -> log-probabilities.
3. Test prompt recovery: agglomerative clustering on content-word TF-IDF with a hard
   cap of K=18 passages per cluster (each prompt is answered once by every model).
4. Hungarian assignment inside each recovered cluster: every model used at most once.
5. Sinkhorn balancing of the class marginals before the assignment.

CV = GroupKFold over prompt_id so the held-out fold has unseen prompts, exactly like
the hidden test; cluster recovery is run on the held-out fold texts (prompt ids hidden).
"""
import argparse, time, re, sys
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

K = 18


def stylometric(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        words = re.findall(r"\w+", t)
        nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        rows.append([
            np.log(n), np.log(nw), nw / max(len(sents), 1),
            t.count(',') / nw, t.count(';') / nw, t.count(':') / nw, t.count('—') / nw,
            t.count('–') / nw, t.count('-') / nw, t.count('(') / nw, t.count('"') / nw,
            t.count("'") / nw, t.count('’') / nw, t.count('“') / nw, t.count('!') / nw,
            t.count('?') / nw, sum(c.isupper() for c in t) / n, sum(c.isdigit() for c in t) / n,
            np.mean([len(w) for w in words]) if words else 0, len(set(words)) / nw,
            t.count('\n') / nw, t.count('...') / nw, t.count('…') / nw,
        ])
    return np.array(rows, dtype=np.float32)


def build_features(train_text, test_text, max_char=300000, max_word=200000):
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), min_df=2, sublinear_tf=True,
                         lowercase=False, max_features=max_char, dtype=np.float32)
    vw = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                         lowercase=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=max_word, dtype=np.float32)
    Xc_tr = vc.fit_transform(train_text); Xc_te = vc.transform(test_text)
    Xw_tr = vw.fit_transform(train_text); Xw_te = vw.transform(test_text)
    S_tr = stylometric(train_text); S_te = stylometric(test_text)
    mu, sd = S_tr.mean(0), S_tr.std(0) + 1e-6
    S_tr = (S_tr - mu) / sd; S_te = (S_te - mu) / sd
    Xtr = sparse.hstack([Xc_tr, Xw_tr, sparse.csr_matrix(S_tr) * 0.3]).tocsr()
    Xte = sparse.hstack([Xc_te, Xw_te, sparse.csr_matrix(S_te) * 0.3]).tocsr()
    return Xtr, Xte


def centre_by_group(X, groups):
    """Subtract the mean feature vector of each group (prompt) - removes topic content."""
    X = X.toarray() if sparse.issparse(X) else X
    out = X.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] -= X[idx].mean(0, keepdims=True)
    return out


def recover_prompts(texts, n_groups):
    """Cluster passages into prompt groups of size <= K using content words."""
    v = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english',
                        sublinear_tf=True, max_features=50000)
    X = normalize(v.fit_transform(texts))
    D = pdist(X.toarray(), metric='cosine') if X.shape[0] < 6000 else None
    Z = linkage(D, method='average')
    labels = fcluster(Z, t=n_groups, criterion='maxclust') - 1
    # size-cap clusters: split any cluster bigger than K by re-clustering it
    labels = labels.copy(); nxt = labels.max() + 1
    Xd = X.toarray()
    changed = True
    while changed:
        changed = False
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            if len(idx) > K:
                sub = linkage(pdist(Xd[idx], metric='cosine'), method='average')
                m = int(np.ceil(len(idx) / K))
                sl = fcluster(sub, t=m, criterion='maxclust') - 1
                for s in np.unique(sl):
                    if s == 0: continue
                    labels[idx[sl == s]] = nxt; nxt += 1
                changed = True
    return labels


def sinkhorn(P, iters=50):
    P = P.copy()
    target = np.full(P.shape[1], P.shape[0] / P.shape[1])
    for _ in range(iters):
        P *= target / np.maximum(P.sum(0), 1e-12)
        P /= P.sum(1, keepdims=True)
    return P


def group_assign(logp, groups):
    """Hungarian assignment inside each group so each class is used at most once."""
    pred = logp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 1: continue
        cost = -logp[idx]
        if len(idx) > K:  # cannot be one prompt; fall back to argmax
            continue
        r, c = linear_sum_assignment(cost)
        pred[idx[r]] = c
    return pred


def fit_predict(train_text, y, train_groups, test_text, test_groups, C=10.0):
    Xtr, Xte = build_features(train_text, test_text)
    # raw model
    clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1)
    clf.fit(Xtr, y)
    lp_raw = clf.predict_log_proba(Xte)
    # prompt-centred model (dense; keep feature count manageable via SVD-free centring on sparse->dense is heavy)
    # use centring on a reduced view: keep top features by chi2-free heuristic: all features but cast to float32
    Xtr_c = centre_by_group(Xtr.astype(np.float32), train_groups)
    Xte_c = centre_by_group(Xte.astype(np.float32), test_groups)
    clf2 = LogisticRegression(C=C, max_iter=3000, n_jobs=-1)
    clf2.fit(Xtr_c, y)
    lp_c = clf2.predict_log_proba(Xte_c)
    return lp_raw, lp_c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='/home/claude/data/ntl'); ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--C', type=float, default=10.0); ap.add_argument('--max_feats', type=int, default=0)
    a = ap.parse_args()
    tr = pd.read_csv(f'{a.data}/train.csv')
    labels = sorted(tr.model.unique()); lab2i = {l: i for i, l in enumerate(labels)}
    y = tr.model.map(lab2i).values; groups = tr.prompt_id.values
    gkf = GroupKFold(n_splits=a.folds)
    accs = {k: [] for k in ['raw', 'centred', 'blend', 'blend+hung(true groups)', 'blend+hung(recovered)', 'cluster_purity']}
    t0 = time.time()
    for f, (tri, tei) in enumerate(gkf.split(tr, y, groups)):
        te_text = tr.text.values[tei]
        rec = recover_prompts(te_text, n_groups=len(np.unique(groups[tei])))
        # purity of recovered clusters
        pur = np.mean([pd.Series(groups[tei][rec == g]).value_counts().iloc[0] / (rec == g).sum() for g in np.unique(rec)])
        lp_raw, lp_c = fit_predict(tr.text.values[tri], y[tri], groups[tri], te_text, rec, C=a.C)
        yt = y[tei]
        accs['raw'].append((lp_raw.argmax(1) == yt).mean())
        accs['centred'].append((lp_c.argmax(1) == yt).mean())
        lp = 0.5 * lp_raw + 0.5 * lp_c
        P = sinkhorn(np.exp(lp - lp.max(1, keepdims=True)) / np.exp(lp - lp.max(1, keepdims=True)).sum(1, keepdims=True))
        lpb = np.log(P + 1e-12)
        accs['blend'].append((lpb.argmax(1) == yt).mean())
        accs['blend+hung(true groups)'].append((group_assign(lpb, groups[tei]) == yt).mean())
        accs['blend+hung(recovered)'].append((group_assign(lpb, rec) == yt).mean())
        accs['cluster_purity'].append(pur)
        print(f'fold {f}: ' + ' | '.join(f'{k}={v[-1]:.3f}' for k, v in accs.items()), f'({time.time()-t0:.0f}s)', flush=True)
    print('MEAN: ' + ' | '.join(f'{k}={np.mean(v):.4f}' for k, v in accs.items()))


if __name__ == '__main__':
    main()
