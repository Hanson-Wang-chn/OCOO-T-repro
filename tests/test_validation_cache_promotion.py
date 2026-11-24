"""A reused full population must reproduce fresh official DE evaluation exactly."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocoot.evaluate import official_metrics
from ocoot.runtime import file_sha256


@pytest.fixture
def calibration_cache(tmp_path, monkeypatch):
    x=np.array([[200,200,0]]+[[3,1,0]]*12+[[1,3,0]]*12+[[1,1,0]]*20+[[500,500,0]],dtype=np.float32)
    meta={'cell':np.zeros(len(x),dtype=np.int64),'pert':np.array([0]+[0]*12+[1]*12+[2]*20+[0]),
          'split':np.array([0]+[1]*24+[3]*20+[2])}
    path=tmp_path/'data';path.mkdir()
    vocab={'all_genes':['X','Y','non-targeting'],'genes':['G1','G2','G3']}
    np.save(path/'expression.npy',x);np.savez(path/'metadata.npz',**meta)
    (path/'vocab.json').write_text(json.dumps(vocab))
    data=SimpleNamespace(path=path,x=x,meta=meta,val=np.arange(1,25),control=np.arange(25,45),
                         test=np.array([45]),vocab=vocab)
    data.evaluation_indices=lambda split,cap: data.val if split=='validation' and cap is None else None
    calibration=tmp_path/'calibration';case=calibration/'refine_00';case.mkdir(parents=True)
    raw=x[data.val].copy();raw[:,2]=-.125
    np.save(case/'prediction.npy',raw);np.save(case/'indices.npy',data.val)
    options={'noise_std':1.,'batch_size':32,'steps':50,'guidance':1.5,'cfg_min':.1,'cfg_max':.9,
             'clamp':False,'seed':4242}
    (case/'generation_state.json').write_text(json.dumps({'completed':24,'total':24,'options':options}))
    checkpoint=tmp_path/'frozen.pt'
    torch.save({'config':{'data_dir':str(path),'model':{},'training':{'noise_std':1.}},
                'step':123,'weights':'ema_0.9996','model':{}},checkpoint)
    pinned_source=tmp_path/'pinned_source.py';pinned_source.write_text('pass\n')
    provenance={'checkpoint':str(checkpoint),'checkpoint_sha256':file_sha256(checkpoint),'step':123,
                'weights':'ema_0.9996','data_dir':str(path),
                'source_sha256':{str(pinned_source):file_sha256(pinned_source)},
                'formal_guidance_policy':'literal_dual_cfg_positive_scale'}
    selected={**provenance,'tag':'refine_00','cap':None,'clamp':True,'sampling_steps':50,
              'seed':4242,'batch_size':32,'guidance':1.5,'cfg_min':.1,'cfg_max':.9,
              'metrics':official_metrics(data,data.val,np.maximum(raw,0),case/'metrics_clamped',de=True),
              'cache_provenance':{'directory':str(case),'raw_unclamped':True,'split':'validation',
                                  'files':{name:file_sha256(case/name) for name in
                                           ('prediction.npy','indices.npy','generation_state.json')}}}
    (calibration/'calibration_config.json').write_text(json.dumps(provenance))
    (calibration/'selected.json').write_text(json.dumps(selected))
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/promote_validation_cache.py'))
    promote=namespace['promote'];monkeypatch.setitem(promote.__globals__,'BenchmarkData',lambda _:data)
    return SimpleNamespace(promote=promote,calibration=calibration,case=case,selected=selected,
                           out=tmp_path/'validation_full',raw=raw,data=data,checkpoint=checkpoint)


def test_full_cache_is_independent_raw_copy_and_exact_clamp_with_fresh_all9(calibration_cache):
    c=calibration_cache
    source_digest=file_sha256(c.case/'prediction.npy')
    result=c.promote(c.calibration,c.out)
    np.testing.assert_array_equal(np.load(c.out/'prediction_raw.npy'),c.raw)
    np.testing.assert_array_equal(np.load(c.out/'prediction.npy'),np.maximum(c.raw,0))
    np.testing.assert_array_equal(np.load(c.out/'indices.npy'),c.data.val)
    assert not (c.out/'prediction_raw.npy').is_symlink()
    assert result['cells']==24 and result['new_generated_samples']==0 and not result['gpu_used']
    assert result['metrics']==c.selected['metrics'] and len(result['metrics'])==9
    assert result['metrics']['MSE']==0
    counts=json.loads((c.out/'metrics/population_counts.json').read_text())
    assert counts=={'real_perturbed_cells':24,'generated_cells':24,'shared_real_controls':20}
    assert c.promote(c.calibration,c.out)==result
    assert file_sha256(c.case/'prediction.npy')==source_digest


def test_promotion_resumes_after_metric_process_failure(calibration_cache,monkeypatch):
    c=calibration_cache;module=c.promote.__globals__;original=module['official_metrics']
    def interrupted(*args,**kwargs):raise RuntimeError('simulated evaluator interruption')
    monkeypatch.setitem(module,'official_metrics',interrupted)
    with pytest.raises(RuntimeError,match='simulated evaluator interruption'):
        c.promote(c.calibration,c.out)
    assert (c.out/'generation_state.json').exists()
    assert not (c.out/'promotion_completed.json').exists()
    raw_digest=file_sha256(c.out/'prediction_raw.npy')
    monkeypatch.setitem(module,'official_metrics',original)
    result=c.promote(c.calibration,c.out)
    assert result['new_generated_samples']==0
    assert result['metrics']==c.selected['metrics']
    assert file_sha256(c.out/'prediction_raw.npy')==raw_digest


@pytest.mark.parametrize('change',['capped','sample_corruption','different_options','checkpoint_corruption','wrong_metrics'])
def test_reuse_rejects_incomplete_or_changed_inputs_before_completion(calibration_cache,change):
    c=calibration_cache
    if change=='capped':c.selected['cap']=1
    if change=='sample_corruption':np.save(c.case/'prediction.npy',c.raw+.01)
    if change=='different_options':
        marker=json.loads((c.case/'generation_state.json').read_text());marker['options']['guidance']=2.
        (c.case/'generation_state.json').write_text(json.dumps(marker))
        c.selected['cache_provenance']['files']['generation_state.json']=file_sha256(c.case/'generation_state.json')
    if change=='checkpoint_corruption':
        with c.checkpoint.open('ab') as stream:stream.write(b'changed')
    if change=='wrong_metrics':c.selected['metrics']['MSE']=1.
    (c.calibration/'selected.json').write_text(json.dumps(c.selected))
    with pytest.raises((ValueError,AssertionError)):
        c.promote(c.calibration,c.out)
    assert not (c.out/'promotion_completed.json').exists()
