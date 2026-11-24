"""An interrupted epoch-end generation must finish before advancing training."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from ocoot.model import ModelConfig,OCOOT
import ocoot.train as trainer
import ocoot.evaluate as evaluation


class MemoryData:
    def __init__(self,unused):
        self.x=np.random.default_rng(17).gamma(1.,.4,(16,16)).astype(np.float32)
        self.train=np.arange(8);self.val=np.arange(8,12);self.test=np.array([],dtype=np.int64)
        self.control=np.arange(12,16)
        self.meta={'pert':np.array([0,1]*4+[0,0,1,1]+[16]*4),
                   'cell':np.array([0]*4+[1]*4+[0]*8),'batch':np.zeros(16,dtype=np.int64),
                   'split':np.array([0]*8+[1]*4+[3]*4)}
        self.vocab={'genes':[f'G{i}' for i in range(16)],
                    'all_genes':[f'G{i}' for i in range(16)]+['non-targeting'],
                    'cell_lines':['line0','line1'],'batches':['b0']}

    def build_model(self,cfg):
        return OCOOT(ModelConfig(**cfg),torch.randn(17,32),torch.arange(16),2,1)

    def batch(self,indices,device='cuda'):
        return (torch.from_numpy(self.x[indices]).to(device),
                *[torch.from_numpy(self.meta[k][indices]).to(device) for k in ['pert','cell','batch']])

    def evaluation_indices(self,split='validation',cap=None):
        return self.val if cap is None else np.array([8,10])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Single-GPU trainer integration')
@pytest.mark.parametrize('epochs',[1,2])
def test_resume_pending_generation(tmp_path,monkeypatch,epochs):
    monkeypatch.setattr(trainer,'BenchmarkData',MemoryData)
    monkeypatch.setattr(trainer,'readme_event',lambda *a,**k:None)
    monkeypatch.setattr(trainer,'update_live_status',lambda *a,**k:None)
    monkeypatch.setattr(evaluation,'update_readme_live',lambda *a,**k:None)
    config={'seed':42,'data_dir':'memory','run_dir':str(tmp_path),
            'model':{'dim':16,'depth':2,'heads':2,'value_hidden':16,'time_dim':8,
                     'context_tokens':2,'context_start':1,'grad_checkpoint':False,
                     'null_condition_init':'training_mean'},
            'training':{'epochs':epochs,'warmup_epochs':1,'micro_batch':2,'effective_batch':4,
                        'lr':1e-4,'betas':[.9,.95],'weight_decay':0.,'ema_decays':[.9999,.9996],
                        'noise_std':1.,'p_mean':-.8,'p_std':.8,'compile':False,'train_subset':None,
                        'checkpoint_steps':1,'log_steps':1,'val_cells':2,'val_interval_steps':1,
                        'val_generation_epochs':1,'val_generation_cap':1,
                        'val_generation_steps':50,'val_generation_batch':1,'generation_mse_weight':10.},
            'sampling':{'steps':50,'guidance':None,'cfg_min':.1,'cfg_max':.9,'clamp':False}}
    original_sample=evaluation.sample
    calls=[]
    def interrupted_sample(*args,**kwargs):
        calls.append(kwargs['seed'])
        if len(calls)==2:raise RuntimeError('Simulated interruption during generation')
        return original_sample(*args,**kwargs)
    monkeypatch.setattr(evaluation,'sample',interrupted_sample)
    with pytest.raises(RuntimeError,match='Simulated interruption'):
        trainer.train(config)
    pending=torch.load(tmp_path/'last.pt',weights_only=False,map_location='cpu')
    assert pending['step']==2 and pending['pending_generation_epoch']==1
    marker=tmp_path/'validation_epoch_001/generation_state.json'
    assert json.loads(marker.read_text())['completed']==1
    first_prediction=np.load(tmp_path/'validation_epoch_001/prediction.npy')[0].copy()
    assert not (tmp_path/'finished.json').exists()
    resumed_calls=[]
    def counted_sample(*args,**kwargs):
        resumed_calls.append(kwargs['seed'])
        return original_sample(*args,**kwargs)
    monkeypatch.setattr(evaluation,'sample',counted_sample)
    trainer.train(config)
    final=torch.load(tmp_path/'last.pt',weights_only=False,map_location='cpu')
    assert final['step']==2*epochs and final['pending_generation_epoch'] is None
    assert len(resumed_calls)==1+2*(epochs-1)  # completed sample is never repeated
    np.testing.assert_array_equal(np.load(tmp_path/'validation_epoch_001/prediction.npy')[0],first_prediction)
    assert (tmp_path/'best_generation.pt').exists() and (tmp_path/'finished.json').exists()
    selected = torch.load(tmp_path/'best_generation.pt', map_location='cpu', weights_only=False)
    metrics = selected['metrics']
    score = metrics['PDCorr'] + np.mean([metrics[key] for key in ['PDS_L1', 'PDS_L2', 'PDS_cos']]) - 10 * metrics['MSE']
    assert selected['selection_score'] == pytest.approx(score)
    assert final['best_generation'] == selected['selection_score']
    events=[json.loads(x) for x in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [e['step'] for e in events if e['event']=='train']==list(range(1,2*epochs+1))
    if epochs==1:
        assert torch.equal(pending['rng']['torch'],final['rng']['torch'])
        assert all(torch.equal(a,b) for a,b in zip(pending['rng']['cuda'],final['rng']['cuda']))
    before=(tmp_path/'last.pt').stat().st_mtime_ns
    trainer.train(config)  # a completed run is idempotent
    assert (tmp_path/'last.pt').stat().st_mtime_ns==before
    with pytest.raises(FileExistsError,match='new run_dir'):
        trainer.train(config,resume=False)
