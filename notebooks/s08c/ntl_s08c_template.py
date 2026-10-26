# %% [markdown]
# # Name That LLM - s08c: CPU-only stylometric pipeline with prompt recovery + Hungarian assignment
#
# No GPU. Five ideas, each verified with GroupKFold CV over `prompt_id` (held-out prompts, like the hidden test):
#
# 1. **Char (1-5, case-sensitive) + word/punct (1-2) TF-IDF** plus 23 stylometric ratios.
# 2. **Prompt centring** - subtract the per-prompt mean feature vector, so topic content cancels and only the model's style remains
#    (train: by `prompt_id`; test: by recovered prompt cluster). CV: 0.356 -> 0.42 argmax accuracy.
# 3. **Prompt recovery on the test set** - every prompt is answered exactly once by each of the 18 models, so average-linkage
#    clustering on content words with a hard cap of 18 per cluster recovers the prompts (purity 1.000 in CV).
# 4. **Sinkhorn balancing** of the class marginals (18 x N/18 per class).
# 5. **Hungarian assignment inside each recovered prompt** - every model used at most once per prompt. CV: 0.42 -> 0.478.
#
# Optional: the public `nlm_matched_clean` passage pool (24k extra labelled passages, MIT) is added to training when mounted.

# %%
import os, re, time, glob, gc
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

T0 = time.time()
K = 18
CFG = dict(
    use_extra=__USE_EXTRA__,      # add the public passage pool to training
    extra_weight=0.5,             # sample weight of pool passages
    max_char=__MAX_CHAR__, max_word=__MAX_WORD__,
    C_lr=__C_LR__, C_svc=1.0,
    w_raw=__W_RAW__,                    # weight of the uncentred model in the log-prob blend
    w_svc=__W_SVC__,                    # weight of the SVC inside the centred part
    sinkhorn_iters=50,
)
print(CFG)
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm', '/home/claude/data/ntl'] if os.path.exists(p + '/train.csv'))
train = pd.read_csv(f'{DATA}/train.csv'); test = pd.read_csv(f'{DATA}/test.csv')
LABELS = sorted(train.model.unique()); lab2i = {l: i for i, l in enumerate(LABELS)}
assert len(LABELS) == K
print(train.shape, test.shape)

extra = None
if CFG['use_extra']:
    cand = glob.glob('/kaggle/input/**/nlm_matched_clean.parquet', recursive=True) + glob.glob('/home/claude/data/ntl_extra/nlm_matched_clean.parquet')
    if cand:
        extra = pd.read_parquet(cand[0])[['text', 'model']]
        extra = extra[extra.model.isin(LABELS) & ~extra.text.isin(set(train.text))].reset_index(drop=True)
        print('extra pool', extra.shape)

# %%
def stylometric(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1); words = re.findall(r"\w+", t); nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        rows.append([np.log(n), np.log(nw), nw / max(len(sents), 1),
                     t.count(',') / nw, t.count(';') / nw, t.count(':') / nw, t.count('—') / nw, t.count('–') / nw,
                     t.count('-') / nw, t.count('(') / nw, t.count('"') / nw, t.count("'") / nw, t.count('’') / nw,
                     t.count('“') / nw, t.count('!') / nw, t.count('?') / nw, sum(c.isupper() for c in t) / n,
                     sum(c.isdigit() for c in t) / n, np.mean([len(w) for w in words]) if words else 0,
                     len(set(words)) / nw, t.count('\n') / nw, t.count('...') / nw, t.count('…') / nw])
    return np.array(rows, dtype=np.float32)

def build_features(train_text, test_text, max_char, max_word):
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), min_df=2, sublinear_tf=True, lowercase=False,
                         max_features=max_char, dtype=np.float32)
    vw = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                         token_pattern=r"(?u)\b\w+\b|[^\w\s]", max_features=max_word, dtype=np.float32)
    Xc_tr = vc.fit_transform(train_text); Xc_te = vc.transform(test_text)
    Xw_tr = vw.fit_transform(train_text); Xw_te = vw.transform(test_text)
    S_tr = stylometric(train_text); S_te = stylometric(test_text)
    mu, sd = S_tr.mean(0), S_tr.std(0) + 1e-6
    S_tr = (S_tr - mu) / sd; S_te = (S_te - mu) / sd
    Xtr = sparse.hstack([Xc_tr, Xw_tr, sparse.csr_matrix(S_tr) * 0.3]).tocsr().astype(np.float32)
    Xte = sparse.hstack([Xc_te, Xw_te, sparse.csr_matrix(S_te) * 0.3]).tocsr().astype(np.float32)
    return Xtr, Xte

def centre_by_group(X, groups):
    X = X.toarray() if sparse.issparse(X) else X.copy()   # one dense copy, centred in place
    codes, inv = np.unique(groups, return_inverse=True)
    for g in range(len(codes)):
        idx = np.where(inv == g)[0]
        X[idx] -= X[idx].mean(0, keepdims=True)
    return X

def cosine_condensed(X):
    """Condensed cosine distance from a sparse row-normalised matrix (sparse matmul, no dense features)."""
    X = normalize(X)
    S = (X @ X.T).toarray().astype(np.float32)
    D = 1.0 - S; np.fill_diagonal(D, 0.0); D = np.maximum(D, 0.0)
    return squareform(D, checks=False)

def recover_prompts(texts, n_groups, cut=__CUT__):
    """Average-linkage clustering on content words. cut=None: force n_groups clusters (train-like structure);
    cut=t: cut the dendrogram at cosine distance t (calibrated on train: within-prompt median 0.80, 95% 0.91;
    between-prompt 1% 0.93). Any cluster larger than K is split."""
    v = TfidfVectorizer(analyzer='word', min_df=1, stop_words='english', sublinear_tf=True, max_features=50000, dtype=np.float32)
    X = v.fit_transform(texts)
    Z = linkage(cosine_condensed(X), method='average')
    labels = (fcluster(Z, t=n_groups, criterion='maxclust') if cut is None else fcluster(Z, t=cut, criterion='distance')) - 1
    nxt = labels.max() + 1; changed = True
    while changed:
        changed = False
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            if len(idx) > K:
                sub = linkage(cosine_condensed(X[idx]), method='average')
                sl = fcluster(sub, t=int(np.ceil(len(idx) / K)), criterion='maxclust') - 1
                for s in np.unique(sl):
                    if s == 0: continue
                    labels[idx[sl == s]] = nxt; nxt += 1
                changed = True
    return labels

def softmax(lp):
    e = np.exp(lp - lp.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def sinkhorn(P, iters):
    P = P.copy(); target = np.full(P.shape[1], P.shape[0] / P.shape[1])
    for _ in range(iters):
        P *= target / np.maximum(P.sum(0), 1e-12); P /= P.sum(1, keepdims=True)
    return P

def group_assign(logp, groups):
    pred = logp.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if 1 < len(idx) <= K:
            r, c = linear_sum_assignment(-logp[idx]); pred[idx[r]] = c
    return pred

# %%
# ---- test prompt recovery -------------------------------------------------------------------
test_text = test.text.values
n_test = len(test_text)
if n_test >= 2 * K:
    n_groups = int(round(n_test / K))
    rec = recover_prompts(test_text, n_groups)   # cut from CFG; n_groups only used when cut is None
    sizes = pd.Series(rec).value_counts()
    print(f'recovered {len(sizes)} prompt clusters; size distribution:', sizes.describe()[['min', '50%', 'max']].to_dict())
else:
    rec = np.arange(n_test)   # the 3-row public test.csv
print(f'[{time.time()-T0:.0f}s] prompts recovered')

# ---- training set (+ pool) ------------------------------------------------------------------
trn_text, y, grp = train.text.values, train.model.map(lab2i).values, train.prompt_id.values
w = np.ones(len(trn_text))
if extra is not None:
    rng = np.random.default_rng(0); order = rng.permutation(len(extra)); exg = np.empty(len(extra), dtype=object); base = 0
    for s in range(0, len(extra), 3000):      # pseudo-prompt groups for centring the pool
        idx = order[s:s + 3000]
        lab = recover_prompts(extra.text.values[idx], n_groups=max(1, len(idx) // K))
        for i, l in zip(idx, lab): exg[i] = f'x{base + l}'
        base += lab.max() + 1
    trn_text = np.concatenate([trn_text, extra.text.values]); y = np.concatenate([y, extra.model.map(lab2i).values])
    grp = np.concatenate([grp, exg]); w = np.concatenate([w, np.full(len(extra), CFG['extra_weight'])])
    print(f'[{time.time()-T0:.0f}s] pool grouped, train rows', len(trn_text))

Xtr, Xte = build_features(trn_text, test_text, CFG['max_char'], CFG['max_word'])
print(f'[{time.time()-T0:.0f}s] features', Xtr.shape)

# raw (uncentred) model
lr_raw = LogisticRegression(C=10.0, max_iter=3000).fit(Xtr, y, sample_weight=w)
lp_raw = lr_raw.predict_log_proba(Xte); del lr_raw
# prompt-centred models
Xtr_c = centre_by_group(Xtr, grp); Xte_c = centre_by_group(Xte, rec); del Xtr, Xte; gc.collect()
lr_c = LogisticRegression(C=CFG['C_lr'], max_iter=3000).fit(Xtr_c, y, sample_weight=w)
lp_lr = lr_c.predict_log_proba(Xte_c); del lr_c
if CFG['w_svc'] > 0:
    svc = LinearSVC(C=CFG['C_svc'], max_iter=5000).fit(Xtr_c, y, sample_weight=w)
    lp_svc = np.log(softmax(svc.decision_function(Xte_c) * 3.0) + 1e-12); del svc
else:
    lp_svc = lp_lr
del Xtr_c, Xte_c; gc.collect()
print(f'[{time.time()-T0:.0f}s] models fitted')

# %%
# size-aware blend: a passage in a recovered cluster of size n gets centred-model weight (n-1)/(K-1);
# singletons (centred features are all zero) fall back to the uncentred model entirely.
sizes = pd.Series(rec).map(pd.Series(rec).value_counts()).values
w_cen = (1 - CFG['w_raw']) * np.clip((sizes - 1) / (K - 1), 0, 1)[:, None]
lp_cen = (1 - CFG['w_svc']) * lp_lr + CFG['w_svc'] * lp_svc
lp = (1 - w_cen) * lp_raw + w_cen * lp_cen
P = softmax(lp)
regular = n_test >= 2 * K and (sizes >= K // 2).mean() >= 0.8     # most rows sit in prompt-sized clusters
print('cluster structure regular:', regular, '| share of rows in clusters >= 9:', round(float((sizes >= K // 2).mean()), 3))
if regular:
    P = sinkhorn(P, CFG['sinkhorn_iters'])
pred = group_assign(np.log(P + 1e-12), rec)
sub = pd.DataFrame({'id': test['id'].astype(str), 'model': np.array(LABELS)[pred]})
sub.to_parquet('submission.parquet', index=False)
print(sub.model.value_counts().to_string())
print(f'done in {time.time()-T0:.0f}s')
