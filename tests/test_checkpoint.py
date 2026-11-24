import random
import numpy as np
import torch
from ocoot.runtime import atomic_save, rng_state, restore_rng, DualEMA


def test_checkpoint_recovers_optimizer_rng_and_both_emas(tmp_path):
    torch.manual_seed(12);random.seed(12);np.random.seed(12)
    m=torch.nn.Linear(3,2);o=torch.optim.AdamW(m.parameters(),lr=.01);ema=DualEMA(m)
    def step():
        o.zero_grad();(m(torch.randn(8,3))-torch.randn(8,2)).square().mean().backward();o.step();ema.update(m)
    step()
    state={'model':m.state_dict(),'opt':o.state_dict(),'ema':ema.state_dict(),'rng':rng_state()}
    atomic_save(state,tmp_path/'last.pt')
    expected_gpu=torch.randn(32,device='cuda') if torch.cuda.is_available() else None
    step();expected={k:v.clone() for k,v in m.state_dict().items()};expected_ema=ema.state_dict()
    restored=torch.load(tmp_path/'last.pt',weights_only=False)
    m.load_state_dict(restored['model']);o.load_state_dict(restored['opt']);ema.load_state_dict(restored['ema'],'cpu');restore_rng(restored['rng'])
    if expected_gpu is not None:torch.testing.assert_close(torch.randn(32,device='cuda'),expected_gpu,rtol=0,atol=0)
    step()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    for d,values in ema.state_dict().items():
        for k,v in values.items():torch.testing.assert_close(v,expected_ema[d][k],rtol=0,atol=0)

