import numpy as np, pandas as pd
exec(open('soft_assign.py').read().split("rng = np.random.RandomState(0)")[0])
def soft_cap(lp, groups, iters=30, tau=1.0, min_size=2):
    """per cluster: iterative scaling with an INEQUALITY column cap (each model at most ~once): only columns whose mass
    exceeds cap are scaled down; rows renormalised. cap = max(1, n/K) so over-size (impure) clusters allow repeats."""
    pred = lp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; n = len(idx)
        if n < min_size: continue
        P = np.exp((lp[idx] - lp[idx].max(1, keepdims=True)) / tau); P /= P.sum(1, keepdims=True)
        cap = max(1.0, n / K)
        for _ in range(iters):
            cs = P.sum(0); P *= np.minimum(1.0, cap / np.maximum(cs, 1e-12)); P /= P.sum(1, keepdims=True)
        pred[idx] = P.argmax(1)
    return pred
def hybrid(lp, groups, tau=1.0):
    """hard Hungarian on dense (12-18) clusters, capped-soft elsewhere"""
    pred = soft_cap(lp, groups, tau=tau)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if 12 <= len(idx) <= 18: r, c = linear_sum_assignment(-lp[idx]); pred[idx[r]] = c
    return pred
rng = np.random.RandomState(0); pids = np.array(sorted(train.prompt_id.unique())); rng.shuffle(pids)
res = {}
for fold in range(5):
    hold = np.isin(train.prompt_id.values, pids[fold::5])
    tr, te = train[~hold], train[hold]; yt, ye = y_all[~hold], y_all[hold]
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000, sublinear_tf=True, lowercase=False)
    vw = TfidfVectorizer(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True, lowercase=False)
    Xtr = sp.hstack([vc.fit_transform(tr.text), vw.fit_transform(tr.text)]).tocsr(); Xte = sp.hstack([vc.transform(te.text), vw.transform(te.text)]).tocsr()
    lp = sinkhorn_global(LogisticRegression(C=20, max_iter=3000).fit(Xtr, yt).predict_log_proba(Xte))
    truth_g = pd.factorize(te.prompt_id.values)[0]; pair = truth_g // 2; impure = pair * 2 + (np.arange(len(te)) % 2)
    # partial structure: each prompt keeps a random 4-18 of its rows
    keep = np.zeros(len(te), bool)
    for g in np.unique(truth_g):
        ii = np.where(truth_g == g)[0]; keep[rng.choice(ii, rng.randint(4, 19), replace=False)] = True
    v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05); Xc = v.fit_transform(te.text.values); S = (Xc @ Xc.T).toarray()
    scen = {'truth': truth_g, 'impure_pairs': impure}
    for thr in (0.15, 0.08, 0.04): scen[f'rec{thr}'] = capped_agglo(S, 18, thr)
    for name, g in scen.items():
        for meth, pred in [('hung_dense', hungarian(lp, g)), ('softcap_t1', soft_cap(lp, g, tau=1.0)), ('softcap_t0.5', soft_cap(lp, g, tau=0.5)),
                           ('softcap_t0.3', soft_cap(lp, g, tau=0.3)), ('hybrid_t0.5', hybrid(lp, g, 0.5))]:
            res.setdefault((name, meth), []).append(balanced_accuracy_score(ye, pred))
    # partial-structure scenario: recovered clusters on the subset
    lp_k, ye_k, te_k = lp[keep], ye[keep], te[keep]
    Xk = v.fit_transform(te_k.text.values); Sk = (Xk @ Xk.T).toarray()
    for thr in (0.15, 0.08, 0.04):
        g = capped_agglo(Sk, 18, thr)
        for meth, pred in [('argmax', lp_k.argmax(1)), ('hung_dense', hungarian(lp_k, g)), ('softcap_t1', soft_cap(lp_k, g, tau=1.0)), ('softcap_t0.5', soft_cap(lp_k, g, tau=0.5)), ('hybrid_t0.5', hybrid(lp_k, g, 0.5))]:
            res.setdefault((f'partial_rec{thr}', meth), []).append(balanced_accuracy_score(ye_k, pred))
    print('fold', fold, flush=True)
print(pd.Series({k: np.mean(v) for k, v in res.items()}).unstack().round(4).to_string())
