"""Estimate capped-generation sampling variability from validation only.

This empirical-distribution resampling diagnostic is optimistic about the true
distribution and is neither an independent test nor a model performance ceiling.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=4)
    p.add_argument('--outdir',default='runs/validation_sampling_reference')
    p.add_argument('--caps',default='8,16,32');p.add_argument('--de',action='store_true');a=p.parse_args()
    try:caps=[int(value) for value in a.caps.split(',')]
    except ValueError:p.error('caps must be positive integers')
    if a.seeds<1 or not caps or any(cap<1 for cap in caps):p.error('seeds and caps must be positive')
    torch.set_num_threads(4)
    data=BenchmarkData();root=Path(a.outdir);root.mkdir(parents=True,exist_ok=True)
    protocol={'split':'validation only','seeds':a.seeds,'caps':caps,'de':a.de,
        'procedure':'Sample with replacement within each empirical validation perturbation; compare with the fixed full validation reference.',
        'limitations':'The empirical reference is shared. This is not independent test performance or a rigorous ceiling.',
        'purpose':'Quantify sampling variability at pilot/periodic-validation cell caps, not select or fit a prediction model.'}
    saved_protocol=root/'protocol.json'
    if saved_protocol.exists():
        previous=json.loads(saved_protocol.read_text());previous.setdefault('de',False)
        if previous!=protocol:raise ValueError('Resampling directory contains a different protocol')
    temporary=root/'protocol.json.tmp';temporary.write_text(json.dumps(protocol,indent=2));temporary.replace(saved_protocol)
    rows=[]
    for cap in caps:
        idx=data.evaluation_indices('validation',cap)
        for seed in range(a.seeds):
            out=root/f'cap_{cap}_seed_{seed}'
            if (out/'summary.json').exists():
                metrics=json.loads((out/'summary.json').read_text())
                if a.de and 'DEPrec' not in metrics:raise ValueError('Cached result does not contain DE metrics')
            else:
                rng=np.random.default_rng(7719+seed)
                pred=np.empty((len(idx),data.x.shape[1]),dtype=np.float32)
                for pert in np.unique(data.meta['pert'][idx]):
                    positions=np.flatnonzero(data.meta['pert'][idx]==pert)
                    pool=data.val[data.meta['pert'][data.val]==pert]
                    drawn=rng.choice(pool,len(positions),replace=True)
                    pred[positions]=data.x[drawn]
                metrics=official_metrics(data,idx,pred,out,de=a.de)
            rows.append({'cap':cap,'seed':seed,**metrics})
    frame=pd.DataFrame(rows);frame.to_csv(root/'resamples.csv',index=False)
    stats=frame.drop(columns='seed').groupby('cap').agg(['mean','std'])
    stats.to_csv(root/'summary.csv')
    print(stats.to_string())


if __name__=='__main__':main()
