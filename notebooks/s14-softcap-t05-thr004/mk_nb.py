"""Generate ntl-s10 notebook: the public 0.651 recipe (nlm-v27 + Llama-3.1-8B v5 member via avazaku's hash-verified
asset cache) with a commit-mode short-circuit. Commit run: input discovery only (all 31 inputs must resolve) + placeholder
submission -> seconds of GPU time. Scoring rerun (KAGGLE_IS_COMPETITION_RERUN=1): full pipeline, exactly as scored 0.651."""
import json, sys, nbformat as nbf
SRC = '/home/claude/pub/avazaku/name-that-llm-0-651-verified-reproduction.ipynb'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'ntl-s10-nlmv27-llama-rerun.ipynb'
EXTRA_ARGS = sys.argv[2] if len(sys.argv) > 2 else ''      # e.g. "--adapt_thr 0.25" appended / overriding
TITLE = sys.argv[3] if len(sys.argv) > 3 else 's10'

# ---- s11+: optional soft-cap marginal decoding instead of the hard dense-cluster Hungarian -----------------------------
OLD_POST = """    _sz = np.bincount(Gte)[Gte]; _nh = 0
    for _g in np.unique(Gte[(_sz >= 12) & (_sz <= 18)]):
        _ii = np.where(Gte == _g)[0]; _r, _c = linear_sum_assignment(-tot[_ii]); _nh += int((tot[_ii].argmax(1)[_r] != _c).sum())
        tot[_ii[_r]] = -1e9; tot[_ii[_r], _c] = 0.0
    log(f'post: em-dash guard fixed {_nfix}, dense-cluster Hungarian changed {_nh}')"""
NEW_POST = """    _sz = np.bincount(Gte)[Gte]; _nh = 0
    if a.post_mode == 'softcap':
        # marginal decoding: per recovered cluster, iterative scaling of exp(tot/tau) with an INEQUALITY column cap
        # (each model at most ~once; cap = max(1, n/18) so impure over-size clusters may repeat a model), then argmax.
        # Maximises expected per-row accuracy instead of the joint MAP; degrades gracefully on impure clusters.
        for _g in np.unique(Gte[_sz >= a.post_min]):
            _ii = np.where(Gte == _g)[0]; _n = len(_ii)
            _P = np.exp((tot[_ii] - tot[_ii].max(1, keepdims=True)) / a.post_tau); _P /= _P.sum(1, keepdims=True)
            _cap = max(1.0, _n / 18)
            for _ in range(30):
                _cs = _P.sum(0); _P *= np.minimum(1.0, _cap / np.maximum(_cs, 1e-12)); _P /= _P.sum(1, keepdims=True)
            _nh += int((tot[_ii].argmax(1) != _P.argmax(1)).sum()); tot[_ii] = np.log(_P + 1e-12)
        log(f'post: em-dash guard fixed {_nfix}, soft-cap (tau {a.post_tau}, min {a.post_min}) changed {_nh} on clusters>={a.post_min}')
    else:
        for _g in np.unique(Gte[(_sz >= 12) & (_sz <= 18)]):
            _ii = np.where(Gte == _g)[0]; _r, _c = linear_sum_assignment(-tot[_ii]); _nh += int((tot[_ii].argmax(1)[_r] != _c).sum())
            tot[_ii[_r]] = -1e9; tot[_ii[_r], _c] = 0.0
        log(f'post: em-dash guard fixed {_nfix}, dense-cluster Hungarian changed {_nh}')"""
OLD_ARG = "p.add_argument('--post', type=int, default=0, help='1: spaced em-dash guard + Hungarian on dense test clusters (12-18)')"
NEW_ARG = OLD_ARG + "\np.add_argument('--post_mode', default='hung', help='hung: dense-cluster Hungarian (public) | softcap: soft-cap marginal decoding on all clusters >= post_min')\np.add_argument('--post_tau', type=float, default=0.5); p.add_argument('--post_min', type=int, default=2)"
def patch_infer(src):
    assert OLD_POST in src and OLD_ARG in src, 'infer.py post block not found'
    return src.replace(OLD_POST, NEW_POST).replace(OLD_ARG, NEW_ARG)

src = json.load(open(SRC)); c = [''.join(x['source']) for x in src['cells']]
nb = nbf.v4.new_notebook()
md = f"""# Name That LLM - {TITLE}: public 0.651 recipe (nlm-v27 + Llama member) with commit short-circuit

Credits: recipe and v5 adapters by Dhruv Tibarewal (nlm-v27, 0.651); Llama-3.1-8B base files via avazaku's hash-verified
asset cache (`name-that-llm-verified-llama-asset-cache` v1); complementary stack by prvsiyan / lucasboesen / vnton37nguyn /
kirilltym; ATLAS bundle by josephadamski91. All Apache-2.0 / public Kaggle assets, versions pinned in the notebook inputs.

**GPU policy.** The commit run only resolves every input path (fails loudly if any of the 31 inputs is missing) and writes a
placeholder `submission.parquet` - well under a minute of T4 time. The full ~8 h pipeline runs only inside Kaggle's hidden-test
scoring rerun, where `KAGGLE_IS_COMPETITION_RERUN` is set, so the weekly accelerator quota stays untouched.
{('**Variant:** `' + EXTRA_ARGS + '`') if EXTRA_ARGS else ''}"""
nb.cells.append(nbf.v4.new_markdown_cell(md))
cell1 = ("import os, pandas as pd\nRERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))\nprint('RERUN =', RERUN)\n"
         "# ---- input discovery: runs in both modes so a missing input fails at commit time, not in the scoring rerun ----\n"
         + c[1] +
         "\nif not RERUN:\n    _t = pd.read_csv(DATA + '/test.csv')\n"
         "    pd.DataFrame({'id': _t['id'].astype(str), 'model': 'gpt-5.5'}).to_parquet('submission.parquet', index=False)\n"
         "    print('commit mode: all inputs resolved, placeholder written for', len(_t), 'rows')\n")
nb.cells.append(nbf.v4.new_code_cell(cell1))
c[3] = patch_infer(c[3])
for k in (2, 3, 4): nb.cells.append(nbf.v4.new_code_cell(c[k]))
def indent(s): return 'if RERUN:\n' + '\n'.join('    ' + l if l.strip() else l for l in s.splitlines()) + '\n'
nb.cells.append(nbf.v4.new_code_cell(indent(c[5])))
run = c[6]
if EXTRA_ARGS:
    # override/append CLI args on the infer.py line
    import re
    line = [l for l in run.splitlines() if l.startswith('!python infer.py')][0]
    new = line
    for m in re.finditer(r'(--\w+)\s+(\S+)', EXTRA_ARGS):
        k, v = m.group(1), m.group(2)
        if re.search(rf'{k}\s+\S+', new): new = re.sub(rf'{k}\s+\S+', f'{k} {v}', new)
        else: new = new.replace(' --out ', f' {k} {v} --out ')
    run = run.replace(line, new)
nb.cells.append(nbf.v4.new_code_cell(indent(run)))
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata['language_info'] = {'name': 'python'}
nbf.write(nb, OUT)
print('wrote', OUT, len(nb.cells), 'cells')
