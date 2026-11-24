"""Validate benchmark preprocessing and materialize compact, ordered HVGs.

The downloaded H5AD is the public STATE/PerturbDiff benchmark, already
efficacy-filtered, library-normalized, log1p-transformed and HVG-selected.
Reapplying normalization or reselecting HVGs here would change the experiment.
"""
import json
import pickle
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from anndata.io import read_elem

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'data/processed'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(RAW / 'replogle.h5ad', 'r') as f:
        obs = read_elem(f['obs'])
        var = read_elem(f['var'])
        genes = pickle.load(open(RAW / 'replogle_real_selected_genes.pkl', 'rb'))
        assert genes == list(var.index[var.highly_variable]), 'HVG order mismatch'
        assert f['obsm/X_hvg'].shape == (643413, 2000)
        selected = np.flatnonzero(var.highly_variable)
        for i in [0, 200000, 600000]:
            np.testing.assert_allclose(f['obsm/X_hvg'][i:i+2], f['X'][i:i+2, selected], atol=1e-6)
        dest = OUT / 'expression.npy'
        if not dest.exists():
            arr = np.lib.format.open_memmap(str(dest) + '.tmp', mode='w+', dtype=np.float32, shape=(len(obs), 2000))
            sums = np.zeros(2000, dtype=np.float64); sqs = sums.copy(); maxval = 0.
            for a in range(0, len(obs), 8192):
                x = f['obsm/X_hvg'][a:a+8192].astype(np.float32)
                if not np.isfinite(x).all() or x.min() < 0: raise ValueError('Invalid log-expression')
                arr[a:a+len(x)] = x
                sums += x.sum(0, dtype=np.float64); sqs += np.square(x).sum(0, dtype=np.float64)
                maxval = max(maxval, float(x.max()))
            arr.flush(); del arr
            Path(str(dest) + '.tmp').replace(dest)
            (OUT/'expression_stats.json').write_text(json.dumps({'mean':float(sums.sum()/len(obs)/2000), 'std':float(np.sqrt((sqs/len(obs)-(sums/len(obs))**2).mean())), 'max':maxval},indent=2))
    protocol = yaml.safe_load((ROOT/'paper/perturbdiff_reference/configs/data/perturb_data/replogle.yaml').read_text())
    (OUT / 'split_protocol.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
    cell_line = obs.cell_line.astype(str).to_numpy()
    pert = obs.gene.astype(str).to_numpy()
    control = pert == 'non-targeting'
    split = np.zeros(len(obs), dtype=np.int8)  # 0 train,1 val,2 test,3 controls
    for i, key in [(1,'validation'), (2,'test')]:
        mask = np.isin(cell_line, protocol['holdout_celltype']) & np.isin(pert, protocol['holdout_pert'][key])
        split[mask] = i
    split[control] = 3
    report = {}
    for i,name in enumerate(['train','validation','test','control']):
        m = split == i
        report[name] = {'cells':int(m.sum()), 'perturbations':len(np.unique(pert[m])), 'cell_lines': sorted(np.unique(cell_line[m]).tolist())}
    assert [report[x]['cells'] for x in report] == [572545,4825,26878,39165], report
    assert report['validation']['perturbations'] == 60 and report['test']['perturbations'] == 380
    # Use the official pair holdout; some labels are additionally absent from
    # all training lines after the upstream efficacy filtering.
    train_pairs = set(zip(cell_line[split==0], pert[split==0]))
    for i in [1,2]: assert not train_pairs.intersection(zip(cell_line[split==i], pert[split==i]))
    all_genes = sorted(set(genes) | set(pert))
    gene_lookup = {x:i for i,x in enumerate(all_genes)}
    lines = sorted(np.unique(cell_line).tolist())
    # Match the paper's 56 batch labels. Cell line is embedded separately.
    batch_labels = obs.gem_group.astype(str).to_numpy()
    batches = sorted(np.unique(batch_labels).tolist())
    np.savez(OUT/'metadata.npz', pert=np.array([gene_lookup[x] for x in pert],dtype=np.int64),
             cell=pd.Categorical(cell_line,categories=lines).codes.astype(np.int64),
             batch=pd.Categorical(batch_labels,categories=batches).codes.astype(np.int64), split=split,
             gene_indices=np.array([gene_lookup[x] for x in genes],dtype=np.int64))
    obs.assign(split=np.array(['train','validation','test','control'])[split], batch_label=batch_labels).to_csv(OUT/'obs.csv.gz',index=True)
    table = torch.load(RAW/'esm2_human.pt', map_location='cpu', weights_only=True)
    table = {k.upper():v.reshape(-1).float() for k,v in table.items()}
    assert {len(v) for v in table.values()} == {5120}
    hgnc = pd.read_csv(RAW/'hgnc_complete_set.tsv', sep='\t', low_memory=False)
    aliases = {}
    for _,row in hgnc.iterrows():
        approved = row['symbol'].upper()
        for field in ['prev_symbol','alias_symbol']:
            for old in str(row.get(field,'')).split('|'):
                aliases.setdefault(old.upper(),set()).add(approved)
    mean_esm = torch.stack(list(table.values())).mean(0)
    emb = torch.stack([mean_esm.clone() for _ in all_genes])
    missing = torch.ones(len(all_genes),dtype=torch.bool)
    mapping = []
    for i,g in enumerate(all_genes):
        candidates = [g.upper()] + sorted(aliases.get(g.upper(),set())) if len(aliases.get(g.upper(),set())) <= 1 else [g.upper()]
        found = next((key for key in candidates if key in table),None)
        if found: emb[i] = table[found]; missing[i] = False
        mapping.append({'gene':g, 'esm_symbol':found, 'missing':bool(missing[i])})
    torch.save({'embeddings':emb,'missing':missing}, OUT/'embeddings.pt')
    pd.DataFrame(mapping).to_csv(OUT/'embedding_mapping.csv',index=False)
    vocab = {'genes':genes,'all_genes':all_genes,'cell_lines':lines,'batches':batches}
    (OUT/'vocab.json').write_text(json.dumps(vocab,indent=2))
    report['embedding_coverage'] = {'total':len(all_genes),'matched':int((~missing).sum()), 'hvg_missing':int(missing[[gene_lookup[x] for x in genes]].sum())}
    report['batch_combinations'] = len(batches)
    report['original_gem_group_labels'] = int(obs.gem_group.nunique())
    report['normalization'] = 'Published normalized log1p X_hvg retained without rescaling; validated against X and fixed HVGs.'
    (OUT/'data_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()

