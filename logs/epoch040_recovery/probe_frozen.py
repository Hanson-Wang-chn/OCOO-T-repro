"""Verify the frozen raw and EMA fields on matched validation noise."""
import argparse
import json
import itertools
from pathlib import Path
import numpy as np
import torch
from ocoot.data import BenchmarkData
from ocoot.flow import sample
from ocoot.train import validate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('step', type=int)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = True
s=torch.load(f'runs/main_v1/snapshots/step_{a.step}/full.pt',map_location='cpu',weights_only=False,mmap=True)
cfg=s['config']; data=BenchmarkData(cfg['data_dir']); model=data.build_model(cfg['model']).cuda(); model.load_state_dict(s['model'])
idx=np.load('runs/main_v1/validation_probe_indices.npy'); ids=data.evaluation_indices(cap=32)
compiled=torch.compile(model, fullgraph=True)
rows={}
for key in ['raw','0.9996','0.9999']:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(s['model'][name] if key=='raw' else s['ema'][key][name])
    numeric={tag:validate(model,f,data,idx,cfg['training']) for tag,f in [('eager',model),('compiled',compiled)]}
    model.eval()
    q, c, b=[torch.from_numpy(data.meta[x][ids[:32]]).cuda() for x in ('pert','cell','batch')]
    x=sample(compiled,q,c,b,2000,steps=50,seed=4242).cpu().numpy()
    row={'denoise':numeric,'sample_batch0':{'mean':float(x.mean()),'std':float(x.std()),'negative':float((x<0).mean())}}
    rows[key]=row
    print(a.step,key,row,flush=True)
a.out.write_text(json.dumps(dict(step=a.step,results=rows),indent=2)+'\n')
