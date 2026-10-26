import json, re, sys
src, out, use_extra, mc, mw, kid, title, c_lr, w_raw, w_svc, cut = sys.argv[1:12]
code = open(src).read().replace('__USE_EXTRA__', use_extra).replace('__MAX_CHAR__', mc).replace('__MAX_WORD__', mw).replace('__C_LR__', c_lr).replace('__W_RAW__', w_raw).replace('__W_SVC__', w_svc).replace('__CUT__', cut)
cells = []
for chunk in re.split(r'^# %%', code, flags=re.M):
    if not chunk.strip(): continue
    if chunk.startswith(' [markdown]'):
        body = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in chunk.split('\n')[1:]).strip()
        cells.append({"cell_type": "markdown", "metadata": {}, "source": body})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": chunk.strip('\n')})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(out, 'w'), indent=1)
meta = {"id": f"gwey1025/{kid}", "title": title, "code_file": out.split('/')[-1], "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["dhruvtibarewal/nlm-matched-clean"] if use_extra == 'True' else [],
        "competition_sources": ["name-that-llm"], "kernel_sources": [], "model_sources": []}
json.dump(meta, open(out.rsplit('/', 1)[0] + '/kernel-metadata.json', 'w'), indent=1)
print('cells', len(cells))
