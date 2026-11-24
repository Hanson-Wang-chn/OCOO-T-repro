import json
from pathlib import Path
import torch
from ocoot.data import BenchmarkData
from ocoot.runtime import file_sha256,timestamp

torch.set_num_threads(4)
p=Path('runs/main_batch_retained_from025/last.pt');state=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
data=BenchmarkData(state['config']['data_dir']);model=data.build_model(state['config']['model']).eval()
model.load_state_dict(state['model'],strict=True)
params=dict(model.named_parameters());assert len(params)==189
for tag in ['0.9999','0.9996']:
 assert set(state['ema'][tag])==set(params)
 for name,param in params.items():
  assert state['ema'][tag][name].shape==param.shape and state['ema'][tag][name].dtype==param.dtype
optimizer=torch.optim.AdamW(model.parameters(),lr=state['config']['training']['lr'])
optimizer.load_state_dict(state['optimizer'])
restored=optimizer.state_dict();assert list(restored['state'])==list(state['optimizer']['state'])
for key,slots in state['optimizer']['state'].items():
 for name,value in slots.items():
  if isinstance(value,torch.Tensor):assert torch.equal(value,restored['state'][key][name])
  else:assert value==restored['state'][key][name]
x,pert,cell,batch=data.batch(data.val[:1],device='cpu');noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(1142));t=torch.full((1,),.3);z=.3*x+.7*noise;off=torch.zeros(1,dtype=torch.bool)
with torch.no_grad():
 model.cfg.drop_batch_with_cell=True;a=model(z,t,pert,cell,batch,off,off)
 model.cfg.drop_batch_with_cell=False;b=model(z,t,pert,cell,batch,off,off)
assert torch.equal(a,b) and a.abs().sum()>0
record={'time':timestamp(),'checkpoint_step':state['step'],'strict_raw_model_load':True,'parameters':len(params),'all_dual_ema_names_shapes_dtypes_match':True,'all_optimizer_slots_bitwise_restored':True,'actual_mature_nonzero_fully_conditional_output_bitwise_equal':True,'output_max_abs_difference':float((a-b).abs().max()),'cells':1,'genes':2000,'cuda_visible_devices':'','main_training_changed':False,'training_random_numbers_bitwise_audited':False,'note':'Initial complete RNG state is preserved; compiled training random streams across the two interpretations have not been separately instrumented. Inference comparisons use the same private per-batch seed.','source_sha256':file_sha256('src/ocoot/model.py'),'fork_checkpoint_sha256':file_sha256(p)}
Path('runs/batch_dropout_mature_trial/actual_checkpoint_verification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
