"""Localize field failure by replacing groups in the frozen EMA for diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from ocoot.data import BenchmarkData
from ocoot.model import sincos
from ocoot.train import validate

root = Path('logs/epoch040_recovery')
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = True
s=torch.load('runs/main_v1/snapshots/step_44720/full.pt',map_location='cpu',weights_only=False,mmap=True)
cfg=s['config']; data=BenchmarkData(cfg['data_dir']); model=data.build_model(cfg['model']).cuda(); model.load_state_dict(s['model'])
idx=np.load('runs/main_v1/validation_probe_indices.npy')[::4]; times=torch.linspace(0,1,64,device='cuda'); model.eval()
modules={
    'only_ema': (),
    'only_raw': ('',),
    'raw_time': ('time.',),
    'raw_input_condition': ('time.','cell.','batch.','null_','gene_proj.','missing_identity.'),
    'raw_output': ('output.','final_norm.','final_ada.'),
    'raw_context_and_value': ('context_pos','value.'),
    'raw_blocks': ('blocks.',),
}
rows={}
for key,prefixes in modules.items():
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(s['model'][name] if name.startswith(prefixes) else s['ema']['0.9996'][name])
        embedded=model.time(sincos(times,model.cfg.time_dim)).detach().float()
        time_stats={'rms':float(embedded.square().mean().sqrt()),'span_rms':float((embedded[0]-embedded[-1]).square().mean().sqrt())}
    rows[key]={'denoise':validate(model,model,data,idx,cfg['training']),'time':time_stats}
    print(key,rows[key],flush=True)
(root/'probe_components.json').write_text(json.dumps(rows,indent=2)+'\n')
