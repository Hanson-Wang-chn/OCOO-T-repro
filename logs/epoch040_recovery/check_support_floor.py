"""Validation-only diagnostic below the observed training positive-expression minimum."""
from pathlib import Path
import argparse
import json
import runpy

import numpy as np
import pandas as pd

from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics
from ocoot.postprocess import apply_zero_floor, quality_checks, primary_score
from ocoot.runtime import exclusive_run, file_sha256, timestamp


ROOT=Path('logs/epoch040_recovery/support_floor_diagnostic')
CALIBRATION=Path('runs/main_ddp_recovery/calibration')


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('refine_00','refine_01'))
    args=parser.parse_args()
    case=CALIBRATION/args.case
    data=BenchmarkData()
    idx=data.evaluation_indices('validation',None)
    marker=read(case/'generation_state.json')
    assert marker['completed']==marker['total']==len(idx)==4825
    np.testing.assert_array_equal(idx,np.load(case/'indices.npy'))
    positive_min=float('inf'); cell=data.meta['cell'][idx[0]]
    train=data.train[data.meta['cell'][data.train]==cell]
    for start in range(0,len(train),2048):
        values=np.asarray(data.x[train[start:start+2048]])
        positive_min=min(positive_min,float(np.min(values,initial=np.inf,where=values>0)))
    assert .07<positive_min
    raw=np.load(case/'prediction.npy',mmap_mode='r'); assert np.isfinite(raw).all()
    with exclusive_run(ROOT/args.case,'analysis'):
        out=ROOT/args.case
        plan={'time':timestamp(),'case':args.case,'candidate_floors':[.05,.07],
              'train_hepg2_min_positive':positive_min,'seed':9232,'draws':2000,
              'selection_or_test_authorized':False,'test_access':False,
              'plan_note':'Use immutable full validation raw predictions and compare an earlier independently '
                          'registered .07 cutoff below the minimum positive training value against .05. '
                          'Compare all nine metrics and retain the formal selector unchanged.',
              'raw_sha256':file_sha256(case/'prediction.npy'),'indices_sha256':file_sha256(case/'indices.npy'),
              'marker':marker}
        path=out/'plan.json'
        if not path.exists():path.write_text(json.dumps(plan,indent=2)+'\n')
        prior=read(path)
        assert {k:v for k,v in prior.items() if k!='time'}=={k:v for k,v in plan.items() if k!='time'}
        rec={}; arrays={}; helper=runpy.run_path('scripts/analyze_full_calibration.py')
        for floor in (.05,.07):
            label=f'{floor:.2f}'; path=out/label
            if not (path/'summary.json').exists():
                array=apply_zero_floor(raw,floor) if floor==.05 else np.where(raw<floor,0,raw)
                official_metrics(data,idx,array,path,de=True)
            metrics=read(path/'summary.json'); labels=sorted(pd.read_csv(path/'results.csv').perturbation.tolist())
            assert len(labels)==len(set(labels))==60
            arrays[label]=helper['load_metrics'](path,labels,metrics)
            rec[label]={'metrics':metrics,'checks':quality_checks(metrics),'score':primary_score(metrics),
                        'zero_fraction':float(np.mean(raw<floor))}
        draws=np.random.default_rng(9232).integers(0,len(labels),(2000,len(labels)))
        _, comparisons=helper['bootstrap_candidates'](arrays,draws)
        result={'time':timestamp(),'test_access':False,'candidates':rec,'paired':comparisons[0]}
        (out/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
