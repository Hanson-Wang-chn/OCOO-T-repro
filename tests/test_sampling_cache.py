from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
import ocoot.evaluate as evaluation


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU sampling entrypoint')
def test_sampling_keeps_raw_values_and_resumes_without_recomputation(tmp_path,monkeypatch):
    calls=[]
    def fake_sample(model,p,c,b,n_genes,**kwargs):
        calls.append(kwargs['seed'])
        return torch.tensor([-1.,0.,2.],device=p.device).repeat(len(p),1)
    monkeypatch.setattr(evaluation,'sample',fake_sample)
    monkeypatch.setattr(evaluation,'update_readme_live',lambda message:None)
    data=SimpleNamespace(x=np.zeros((4,3)),meta={key:np.zeros(4,dtype=np.int64) for key in ['pert','cell','batch']})
    model=SimpleNamespace(eval=lambda:None);indices=np.arange(4)
    result=evaluation.generate(model,data,indices,tmp_path,batch_size=2,clamp=True)
    np.testing.assert_array_equal(result,np.tile([0.,0.,2.],(4,1)))
    np.testing.assert_array_equal(np.load(tmp_path/'prediction_raw.npy'),np.tile([-1.,0.,2.],(4,1)))
    assert len(calls)==2
    result2=evaluation.generate(model,data,indices,tmp_path,batch_size=2,clamp=True)
    np.testing.assert_array_equal(result2,result)
    assert len(calls)==2
    assert json.loads((tmp_path/'generation_state.json').read_text())['completed']==4
    statistics=json.loads((tmp_path/'sample_statistics.json').read_text())
    assert statistics['raw']['negative_fraction']==1/3
    assert statistics['evaluated']['negative_fraction']==0
    with pytest.raises(ValueError,match='different options'):
        evaluation.generate(model,data,indices,tmp_path,batch_size=2,clamp=True,guidance=2.)


def test_distributed_generation_preserves_batch_seeds_order_and_completed_prefix(tmp_path, monkeypatch):
    calls = []

    def fake_sample(model, p, c, b, n_genes, *, seed, **kwargs):
        calls.append(seed)
        return torch.stack((p, torch.full_like(p, -seed)), dim=1).float()

    class LocalRank:
        size = 4

        def __init__(self, rank):
            self.rank = rank
            self.primary = rank == 0

        def from_primary(self, value):
            return value if self.primary else 3

        def barrier(self):
            pass

    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(evaluation, 'sample', fake_sample)
    monkeypatch.setattr(evaluation, 'update_readme_live', lambda message: None)
    indices = np.arange(13)
    data = SimpleNamespace(x=np.empty((len(indices), 2)),
                           meta={key: indices for key in ('pert', 'cell', 'batch')})
    model = SimpleNamespace(eval=lambda: None)
    reference = tmp_path / 'reference'
    target = tmp_path / 'target'
    target.mkdir()
    expected = evaluation.generate(model, data, indices, reference, batch_size=3, clamp=True)
    marker = json.loads((reference / 'generation_state.json').read_text())
    options = marker['options']
    initialized, raw = evaluation._prediction_files(target, indices, 2, True, 0)
    initialized[:3] = expected[:3]
    raw[:3] = np.load(reference / 'prediction_raw.npy')[:3]
    initialized.flush()
    raw.flush()
    evaluation._mark_generated(target, options, 3, len(indices))

    calls.clear()
    for rank in (3, 2, 2, 1, 0):
        result = evaluation.generate(model, data, indices, target, batch_size=3, clamp=True, ranks=LocalRank(rank))
    assert calls == [4254, 4251, 4248, 4245]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.load(target / 'prediction_raw.npy'), np.load(reference / 'prediction_raw.npy'))
    assert json.loads((target / 'generation_state.json').read_text()) == marker

