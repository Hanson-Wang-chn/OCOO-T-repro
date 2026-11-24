"""Train-only mean/effect-transfer diagnostics. These are not OCOO-T models."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics


def main():
    d=BenchmarkData();root=Path('runs/baselines');root.mkdir(exist_ok=True,parents=True)
    pair=d.meta['cell']*len(d.vocab['all_genes'])+d.meta['pert']
    train=d.train
    unique,codes=np.unique(pair[train],return_inverse=True)
    indicator=sparse.csr_matrix((np.ones(len(train),dtype=np.float32),(codes,train)),shape=(len(unique),len(d.x)))
    counts=np.bincount(codes);means=(indicator@d.x)/counts[:,None]
    nperts=len(d.vocab['all_genes']);ncells=len(d.vocab['cell_lines'])
    line_mean=np.zeros((ncells,2000));ctrl_mean=line_mean.copy()
    for c in range(ncells):
        groups=unique//nperts==c
        line_mean[c]=np.average(means[groups],axis=0,weights=counts[groups])
        ctrl=d.control[d.meta['cell'][d.control]==c]
        ctrl_mean[c]=np.asarray(d.x[ctrl]).mean(0)
    # Average same-gene perturbation deltas from observed cell lines only.
    delta=np.zeros((nperts,2000));number=np.zeros(nperts,dtype=int)
    for pairid,mean in zip(unique,means):
        c,p=divmod(int(pairid),nperts)
        delta[p]+=mean-ctrl_mean[c];number[p]+=1
    delta/=np.maximum(1,number[:,None])
    idx=d.val;p=d.meta['pert'][idx];c=d.meta['cell'][idx]
    reports={}
    for alpha in [0.,.25,.5,1.]:
        # The intercept is the training perturbation mean, with centered
        # transferable effects. No held-out expression contributes to fitting.
        centered=delta-delta[d.meta['pert'][train]].mean(0)
        pred=np.maximum(line_mean[c]+alpha*centered[p],0).astype(np.float32)
        reports[f'effect_transfer_{alpha}']=official_metrics(d,idx,pred,root/f'effect_transfer_{alpha}')
    (root/'validation_summary.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':main()

