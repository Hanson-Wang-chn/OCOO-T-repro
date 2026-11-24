"""Validation-only diagnostics of cellular context; no classifier or test fitting.

Training cell-line centroids are reference summaries, not predictions used by
OCOO-T. Nearest-centroid counts help expose gross condition failures that a
global expression mean or denoising loss can conceal.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from ocoot.data import BenchmarkData


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('runs',nargs='*')
    parser.add_argument('--outdir',default='runs/pilot_diagnostics')
    args=parser.parse_args()
    paths=[Path(x) for x in args.runs] if args.runs else sorted(Path('runs').glob('pilot_*/validation_epoch_012'))
    paths=[p/'validation_epoch_012' if not (p/'prediction.npy').exists() else p for p in paths]
    paths=[p for p in paths if (p/'prediction.npy').exists() and
           json.loads((p/'generation_state.json').read_text())['completed']==len(np.load(p/'indices.npy'))]
    if not paths:raise ValueError('No completed validation pilot samples')
    data=BenchmarkData()
    line=data.meta['cell'][data.train]
    nlines=len(data.vocab['cell_lines'])
    membership=sparse.csr_matrix((np.ones(len(line),dtype=np.float32),(line,data.train)),shape=(nlines,len(data.x)))
    centroids=np.asarray(membership@data.x,dtype=np.float64)/np.bincount(line,minlength=nlines)[:,None]
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    records={}
    for path in paths:
        idx=np.load(path/'indices.npy');pred=np.load(path/'prediction.npy',mmap_mode='r')
        raw_path=path/'prediction_raw.npy'
        raw_pred=np.load(raw_path,mmap_mode='r') if raw_path.exists() else pred
        assert np.all(data.meta['split'][idx]==1),'Only validation samples allowed'
        target_lines=np.unique(data.meta['cell'][idx]);assert len(target_lines)==1
        target=int(target_lines[0])
        distances=np.stack([np.square(pred-center).mean(1) for center in centroids],axis=1)
        nearest=np.bincount(distances.argmin(1),minlength=nlines)
        bulk=pred.mean(0,dtype=np.float64)
        groups=np.unique(data.meta['pert'][idx])
        real_idx=data.val[np.isin(data.meta['pert'][data.val],groups)]
        def within_variance(values,pids):
            estimates=[np.var(values[pids==p],axis=0,ddof=1,dtype=np.float64).mean()
                       for p in groups if np.count_nonzero(pids==p)>1]
            return float(np.mean(estimates))
        raw_variance=within_variance(raw_pred,data.meta['pert'][idx])
        clipped_variance=within_variance(np.maximum(raw_pred,0),data.meta['pert'][idx])
        real_variance=within_variance(np.asarray(data.x[real_idx]),data.meta['pert'][real_idx])
        # Several diagnostic cases can live under one run directory. Preserve
        # their separate records instead of overwriting an identical parent key.
        label=path.parent.name if sum(p.parent==path.parent for p in paths)==1 else path.name
        if label in records:label=str(path.resolve())
        records[label]={
            'source':str(path),'generated_cells':len(idx),
            'evaluated_postprocess':'clamp_min_0' if raw_path.exists() else 'none',
            'target_cell_line':data.vocab['cell_lines'][target],
            'nearest_training_centroid_counts':dict(zip(data.vocab['cell_lines'],map(int,nearest))),
            'bulk_MSE_to_training_centroids':dict(zip(data.vocab['cell_lines'],map(float,np.square(centroids-bulk).mean(1)))),
            'within_perturbation_variance':{'real':real_variance,'raw':raw_variance,'clipped':clipped_variance,
                                           'raw_ratio_to_real':raw_variance/real_variance,
                                           'clipped_ratio_to_real':clipped_variance/real_variance},
            'metrics':json.loads((path/'metrics/summary.json').read_text()),
        }
    result={'note':'Validation only. Nearest training centroids are diagnostics, not an independent cell-type classifier.',
            'centroid_source':'perturbed training cells only, excluding validation, test and controls',
            'runs':records}
    (out/'context_diagnostics.json').write_text(json.dumps(result,indent=2))
    labels=list(records)
    values=np.array([list(records[x]['nearest_training_centroid_counts'].values()) for x in labels])
    values=values/values.sum(1,keepdims=True)
    fig,axes=plt.subplots(1,2,figsize=(12,max(3.5,.65*len(labels))),layout='constrained')
    image=axes[0].imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    axes[0].set_xticks(range(nlines),data.vocab['cell_lines'])
    axes[0].set_yticks(range(len(labels)),labels,fontsize=8)
    axes[0].set_title('Nearest training centroid (fraction)')
    for i in range(len(labels)):
        for j in range(nlines):axes[0].text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>.6 else 'black')
    fig.colorbar(image,ax=axes[0],shrink=.7)
    for i,x in enumerate(labels):
        m=records[x]['metrics']
        axes[1].scatter(m['PDCorr'],m['PDS_L2'],label=x.removeprefix('pilot_'),s=50)
    axes[1].axhline(.5,color='gray',linestyle=':',linewidth=1)
    axes[1].set(xlabel='Validation PDCorr',ylabel='Validation PDS L2',title='Validation generation, 50 Heun steps')
    axes[1].legend(fontsize=7)
    fig.savefig(out/'context_diagnostics.png',dpi=160)
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
