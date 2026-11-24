"""Exercise real DDP EMA generation on the in-memory trainer fixture."""
import json
import runpy
from pathlib import Path

import torch
from ocoot import evaluate, train

memory_data = runpy.run_path('tests/test_pending_evaluation.py')['MemoryData']
train.BenchmarkData = memory_data
train.readme_event = lambda *args, **kwargs: None
train.update_live_status = lambda *args, **kwargs: None
evaluate.update_readme_live = lambda *args, **kwargs: None
run = 'runs/repro2_ddp_smoke'
config = {'seed':42, 'data_dir':'memory', 'run_dir':run,
          'model':{'dim':16,'depth':2,'heads':2,'value_hidden':16,'time_dim':8,
                   'context_tokens':2,'context_start':1,'grad_checkpoint':False},
          'training':{'epochs':2,'warmup_epochs':1,'micro_batch':2,'effective_batch':6,
                      'lr':1e-4,'betas':[.9,.95],'weight_decay':0.,'ema_decays':[.9999,.9996],
                      'noise_std':1.,'p_mean':-.8,'p_std':.8,'compile':False,'train_subset':None,
                      'checkpoint_steps':1,'log_steps':1,'val_cells':2,'val_interval_steps':1,
                      'val_generation_epochs':1,'val_generation_cap':1,
                      'val_generation_steps':50,'val_generation_batch':1,'distributed_generation':True},
          'sampling':{'steps':50,'guidance':1.5,'cfg_min':.1,'cfg_max':.9,'clamp':True}}
train.train(config)
