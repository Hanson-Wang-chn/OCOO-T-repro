"""The validation workflow must preserve the paper's inference mechanism."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('conditional_score', [.99, .50])
@pytest.mark.parametrize('full_population_reverses_interval', [False, True])
@pytest.mark.parametrize('mse_weight', [0, 10])
def test_calibration_keeps_conditional_reference_out_of_formal_selection(
        tmp_path, monkeypatch, conditional_score, full_population_reverses_interval, mse_weight):
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/calibrate.py'))
    main=namespace['main']
    module=main.__globals__
    checkpoint=tmp_path/'frozen.pt'
    checkpoint.write_bytes(b'frozen EMA checkpoint placeholder')
    out=tmp_path/'calibration'
    calls=[]

    class Model:
        def load_state_dict(self, state):pass
        def cuda(self):return self
        def eval(self):return self

    class Data:
        def __init__(self, directory):pass
        def build_model(self, config):return Model()
        def evaluation_indices(self, split, cap):
            assert split=='validation'
            return np.arange(48 if cap is None else cap)

    def generate(forward, data, indices, case, **options):
        assert options['steps']==50
        calls.append({'tag':case.name,'cells':len(indices),**options})
        value=10 if options['guidance'] is None else options['guidance']
        result=np.tile([value,options['cfg_min'],options['cfg_max']],(len(indices),1))
        case.mkdir(parents=True,exist_ok=True)
        np.save(case/'prediction.npy',result)
        np.save(case/'indices.npy',indices)
        (case/'generation_state.json').write_text(json.dumps({
            'completed':len(indices),'total':len(indices),'options':{'seed':4242,**options}}))
        return result

    def metrics(data, indices, pred, directory, de=False):
        w,low,high=pred[0]
        value=conditional_score if w==10 else .80-.10*abs(w-1.5)+(.01 if low==0 and high==1 else 0)
        if full_population_reverses_interval and len(indices)==48 and low==0 and high==1:
            value-=.02
        mse = .005 if w == 1 else .02
        result={'PDCorr':value,'PDS_L1':value,'PDS_L2':value,'PDS_cos':value,'MSE':mse,'MAE':.06}
        if de:
            result.update(DEOver=.2,DEPrec=.21,DirAgr=.75)
        return result

    monkeypatch.setattr(module['torch'],'load',lambda *a,**kw:{'config':{'data_dir':'unused','model':{},'training':{'noise_std':1.}},'model':{},'step':123,'weights':'ema_0.9996'})
    monkeypatch.setattr(module['torch'],'compile',lambda model,**kw:model)
    monkeypatch.setattr(module['torch'],'set_num_threads',lambda *a:None)
    monkeypatch.setitem(module,'BenchmarkData',Data)
    monkeypatch.setitem(module,'generate',generate)
    monkeypatch.setitem(module,'official_metrics',metrics)
    monkeypatch.setitem(module,'readme_event',lambda *a,**kw:None)
    monkeypatch.setattr(sys,'argv',['calibrate.py','--checkpoint',str(checkpoint),'--outdir',str(out),
                                  '--scales','.5,1,1.5','--mse-weight',str(mse_weight)])
    main()

    selected=json.loads((out/'selected.json').read_text())
    records=json.loads((out/'all_results.json').read_text())
    assert selected['guidance']==(1 if mse_weight else 1.5)
    expected_interval=(.1,.9) if full_population_reverses_interval else (0,1)
    assert (selected['cfg_min'],selected['cfg_max'])==expected_interval
    assert selected['cap'] is None
    assert selected['formal_guidance_policy']=='literal_dual_cfg_positive_scale'
    assert selected['weights']=='ema_0.9996'
    assert set(selected['cache_provenance']['files'])=={'prediction.npy','indices.npy','generation_state.json'}
    assert any(r['guidance'] is None for r in records)
    assert all(c['guidance'] is not None for c in calls if c['tag'].startswith('refine_'))
    assert all(c['cells']==48 for c in calls if c['tag'].startswith('refine_'))
    assert all(c['cells']==16 for c in calls if c['tag'].startswith('screen_'))
    assert all({'DEOver','DEPrec','DirAgr'}.issubset(r['metrics']) for r in records)
    assert all('DEPrec' not in r['raw_metrics'] for r in records)
    if mse_weight:
        assert selected['mse_weight']==mse_weight
        for record in records:
            assert record['score']==pytest.approx(2 * record['metrics']['PDCorr'])
            assert record['selection_score']==pytest.approx(record['score'] - mse_weight * record['metrics']['MSE'])
    if conditional_score==.99:
        assert next(r['score'] for r in records if r['guidance'] is None)>selected['score']
