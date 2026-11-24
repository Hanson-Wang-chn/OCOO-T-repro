"""Nested full-population diagnostics must retain guidance and saved samples."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocoot.runtime import file_sha256


def setup_extension(tmp_path,monkeypatch,guidance,extra_batch,approved=True):
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/extend_validation_cache.py'))
    main=namespace['main'];module=main.__globals__
    source=tmp_path/'source';source.mkdir();out=tmp_path/'full';out.mkdir();data_path=tmp_path/'data';data_path.mkdir()
    np.savez(data_path/'metadata.npz',split=np.ones(5,dtype=np.int64))
    options={'noise_std':1.,'batch_size':16,'steps':50,'guidance':guidance,'cfg_min':.1,'cfg_max':.9,
             'clamp':True,'seed':4242}
    base=np.array([0,3]);raw=np.array([[1.,-.1,2.],[4.,-.2,5.]],dtype=np.float32)
    np.save(source/'indices.npy',base);np.save(source/'prediction_raw.npy',raw);np.save(source/'prediction.npy',np.maximum(raw,0))
    (source/'generation_state.json').write_text(json.dumps({'completed':2,'total':2,'options':options}))
    torch.save({'config':{'data_dir':str(data_path),'model':{},'training':{}},'step':75,
                'weights':'ema_0.9996','model':{}},source/'model.pt')
    plan={'script_sha256':file_sha256(module['__file__']),'step':75,'extra_cells':3,'full_cells':5,
          'extra_seed':62525,'extra_batch_size':extra_batch,'allow_positive_cfg_extension':approved,
          'source_options':options,'source_sha256':{name:file_sha256(source/name) for name in
                              ('model.pt','indices.npy','prediction.npy','prediction_raw.npy','generation_state.json')}}
    (out/'plan.json').write_text(json.dumps(plan))
    calls=[]
    class Model:
        def cuda(self):return self
        def eval(self):return self
        def load_state_dict(self,state):pass
    data=SimpleNamespace(path=data_path,x=np.zeros((5,3),dtype=np.float32),val=np.arange(5),
                          build_model=lambda cfg:Model())
    def generate(forward,data,indices,directory,**kwargs):
        calls.append(kwargs)
        assert kwargs['guidance']==guidance and kwargs['steps']==50
        assert kwargs['batch_size']==extra_batch and kwargs['seed']==62525
        directory.mkdir();np.save(directory/'indices.npy',indices)
        extra=np.column_stack([indices+.25,-indices-1,np.ones(len(indices))]).astype(np.float32)
        np.save(directory/'prediction_raw.npy',extra);np.save(directory/'prediction.npy',np.maximum(extra,0))
        (directory/'generation_state.json').write_text(json.dumps({'completed':len(indices),'total':len(indices),
                                                                    'options':kwargs}))
    def metrics(data,indices,pred,directory,de=False):
        np.testing.assert_array_equal(indices,[0,3,1,2,4]);directory.mkdir(parents=True,exist_ok=True)
        values={'PDCorr':.4,'MSE':.01,'MAE':.06,'PDS_L1':.8,'PDS_L2':.81,'PDS_cos':.79}
        if de:values.update(DEOver=.2,DEPrec=.21,DirAgr=.75)
        (directory/'summary.json').write_text(json.dumps(values));return values
    monkeypatch.setitem(module,'BenchmarkData',lambda _:data)
    monkeypatch.setitem(module,'generate',generate);monkeypatch.setitem(module,'official_metrics',metrics)
    monkeypatch.setitem(module,'readme_event',lambda *a,**kw:None)
    monkeypatch.setattr(module['torch'],'compile',lambda model,**kw:model)
    monkeypatch.setattr(sys,'argv',['extend_validation_cache.py','--source-dir',str(source),'--outdir',str(out)])
    return SimpleNamespace(main=main,source=source,out=out,base=base,raw=raw,calls=calls,options=options)


@pytest.mark.parametrize(('guidance','batch'),[(None,16),(2.,16),(2.,32)])
def test_guidance_prefix_and_full_population_survive_extension_and_resume(tmp_path,monkeypatch,guidance,batch):
    c=setup_extension(tmp_path,monkeypatch,guidance,batch);c.main()
    complete=json.loads((c.out/'completed.json').read_text())
    assert complete['source_options']['guidance']==guidance
    assert complete['extra_batch_size']==batch and complete['full_cells']==5
    assert complete['prefix_predictions_exactly_preserved']
    np.testing.assert_array_equal(np.load(c.out/'stitched_raw.npy')[:2],c.raw)
    np.testing.assert_array_equal(np.load(c.out/'stitched_predictions.npy')[:2],np.maximum(c.raw,0))
    (c.out/'completed.json').unlink()  # Simulate evaluation interrupted after all samples were saved.
    c.main();assert len(c.calls)==1
    c.main();assert len(c.calls)==1
    if guidance is not None and batch==32:
        np.save(c.out/'stitched_predictions.npy',np.zeros((5,3),dtype=np.float32))
        with pytest.raises(ValueError,match='Stitched cache changed'):c.main()


def test_positive_cfg_requires_explicit_source_matching_registration(tmp_path,monkeypatch):
    c=setup_extension(tmp_path,monkeypatch,2.,32,approved=False)
    with pytest.raises(ValueError,match='explicit matching registration'):c.main()
    assert not c.calls
