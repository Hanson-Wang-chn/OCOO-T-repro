"""One-factor validation study of the mature E45 guidance settings."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from ocoot.data import BenchmarkData
from ocoot.evaluate import generate, official_metrics
from ocoot.postprocess import apply_zero_floor, primary_score
from ocoot.runtime import exclusive_run, file_sha256, timestamp
from prefetch_validation import CALIBRATION, load, read


OUT = CALIBRATION.parent / 'postgate_guidance_study'
CASES = {
    'omit_early': (2., .1, 1.),
    'omit_late': (2., 0., .9),
    'weaker': (1.75, 0., 1.),
    'stronger': (2.25, 0., 1.),
}
REFERENCES = {'full_window': 'screen_full_interval', 'partial_window': 'screen_04'}


def write(path, value):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temp.replace(path)


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    original_path=CALIBRATION/'postprocessing/review.json'
    original=read(original_path)
    assert original['selected'] is None and len(original['candidates'])==4
    protocol_path=CALIBRATION/'calibration_config.json'
    protocol=read(protocol_path)
    payload={'cases':{key:list(value) for key,value in CASES.items()},'references':REFERENCES,
             'checkpoint_sha256':protocol['checkpoint_sha256'],'failed_review_sha256':file_sha256(original_path),
             'calibration_config_sha256':file_sha256(protocol_path), 'seed':4242,'cells_per_case':960,
             'cap':16,'batch_size':32,'steps':50,'floor':.05,'bootstrap':2000,'bootstrap_seed':9232,
             'scope':'E45 fixed checkpoint. Control early/late endpoints one at a time at w2, and bracket w2 at the '
                     'full interval. Match the original cap16 screen exactly. Primary diagnosis is precision with '
                     'paired DE and pseudobulk metrics. No fitting, training, or formal test access.',
             'full_followup_rule':'Prefer one higher-precision candidate, with observed DEPrec gain at least .008 '
                                  'and MSE/MAE increases under .002/.005, PDCorr drop under .01, DEOver drop '
                                  'under .01, and no PDS drop beyond .02, relative to the full-window baseline. '
                                  'Review paired intervals before spending the full follow-up budget. These '
                                  'screen limits do not replace the original full-validation checks.',
             'test_access':False}
    path=OUT/'plan.json'
    if not path.exists():
        write(path,{'time':timestamp(),**payload})
    assert {k:v for k,v in read(path).items() if k!='time'}==payload
    return payload


def evaluate(data, idx, raw, key):
    dest=OUT/key/'floor_0p05'
    if not (dest/'summary.json').exists():
        official_metrics(data,idx,apply_zero_floor(raw,.05),dest,de=True)
    return read(dest/'summary.json')


def generate_case(name, plan):
    p,cfg,data,model=load()
    assert p['checkpoint_sha256']==plan['checkpoint_sha256'] and p['screen_cap']==16 and p['batch_size']==32
    idx=data.evaluation_indices('validation',16); assert len(idx)==960
    w,lo,hi=plan['cases'][name]
    raw=generate(model,data,idx,OUT/name,noise_std=cfg['training']['noise_std'],steps=50,batch_size=32,
                 guidance=w,cfg_min=lo,cfg_max=hi,clamp=False,seed=4242)
    evaluate(data,idx,raw,name)


def analyze(plan):
    data=BenchmarkData(); indices=data.evaluation_indices('validation',16)
    cfg=read(CALIBRATION/'calibration_config.json')
    assert file_sha256(cfg['checkpoint'])==plan['checkpoint_sha256']
    helper=runpy.run_path('scripts/analyze_full_calibration.py')
    sources={**{k:CALIBRATION/v for k,v in plan['references'].items()},**{k:OUT/k for k in plan['cases']}}
    rows={}; arrays={}; hashes={}
    reference_options={'full_window':[2.,0.,1.],'partial_window':[2.,.1,.9]}
    for key,source in sources.items():
        w,lo,hi=(plan['cases']|reference_options)[key]
        if read(source/'generation_state.json')!={'completed':960,'total':960,'options':{
                'noise_std':1.,'batch_size':32,'steps':50,'guidance':w,'cfg_min':lo,'cfg_max':hi,'clamp':False,'seed':4242}}:
            raise ValueError(f'Incomplete or incompatible validation screen {key}')
        np.testing.assert_array_equal(np.load(source/'indices.npy'),indices)
        for name in ('prediction.npy','indices.npy','generation_state.json'):
            hashes[str(source/name)]=file_sha256(source/name)
        metrics=evaluate(data,indices,np.load(source/'prediction.npy',mmap_mode='r'),key)
        rows[key]={'metrics':metrics,'primary':primary_score(metrics)}
        dest=OUT/key/'floor_0p05'
        labels=sorted(pd.read_csv(dest/'results.csv').perturbation.tolist()); assert len(set(labels))==60
        arrays[key]=helper['load_metrics'](dest,labels,metrics)
    draws=np.random.default_rng(9232).integers(0,len(labels),(2000,len(labels)))
    _,pairs=helper['bootstrap_candidates'](arrays,draws)
    report={'time':timestamp(),'cases':rows,'all_pairwise_differences':pairs,'source_sha256':hashes,'test_access':False}
    write(OUT/'report.json',report)
    print(json.dumps(rows,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('case',choices=(*CASES,'analyze','register'))
    a=p.parse_args(); plan=register()
    if a.case=='analyze':
        with exclusive_run(OUT,'comparison'): analyze(plan)
    elif a.case!='register':
        with exclusive_run(OUT/a.case,'generation'): generate_case(a.case,plan)


if __name__=='__main__':
    main()
