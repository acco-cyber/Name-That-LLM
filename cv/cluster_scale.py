"""How well does the pipeline's test-prompt recovery work at hidden-test scale?
Simulate with ALL 150 train prompts at once (2,699 rows; the hidden test is ~222 prompts x 18) and measure ARI / purity /
share of rows in dense (12-18) clusters for the exact capped average-linkage used by infer.py, over the cut threshold."""
import sys, time, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, '/home/claude/pub')
train = pd.read_csv('/home/claude/data/ntl/train.csv')
truth = train.prompt_id.values

def capped_agglo(S, cap=18, thr=0.02):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    merges = []
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        merges.append((M[i, j], lab.copy()))
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab, merges

def report(name, lab):
    s = pd.Series(lab); sizes = s.map(s.value_counts()).values
    pur = sum(pd.Series(truth[lab == g]).value_counts().iloc[0] for g in np.unique(lab)) / len(lab)
    dense = (sizes >= 12).mean()
    # hungarian-usable: rows in clusters of size 12-18 that are pure
    print(f'{name:40s} clusters={len(np.unique(lab)):4d} ARI={adjusted_rand_score(truth, lab):.3f} purity={pur:.3f} dense(>=12)={dense:.3f} size>18={int((sizes>18).sum())}', flush=True)

def run(vec_kw, label, thrs=(0.08, 0.12, 0.15, 0.2, 0.25, 0.3)):
    t = time.time()
    v = TfidfVectorizer(**vec_kw); X = v.fit_transform(train.text.values); S = (X @ X.T).toarray()
    lab, merges = capped_agglo(S, 18, thr=min(thrs))
    print(f'== {label}: {X.shape[1]} feats, agglo {time.time()-t:.0f}s')
    # replay: labels at each threshold = state right before the first merge below thr
    for thr in sorted(thrs):
        if thr == min(thrs): report(f'thr={thr}', lab); continue
        state = next((l for s_, l in merges if s_ < thr), lab)
        report(f'thr={thr}', state)

run(dict(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05), 'pipeline: word tfidf, stop en, min_df2, max_df.05')
run(dict(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05, ngram_range=(1, 2)), 'word 1-2gram')
run(dict(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.1), 'max_df 0.1')
run(dict(sublinear_tf=True, stop_words='english', min_df=1, max_df=0.02), 'max_df 0.02 min_df1')
