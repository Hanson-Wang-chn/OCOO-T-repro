import json,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from ocoot.data import BenchmarkData
from ocoot.runtime import exclusive_run,file_sha256,timestamp,readme_event
import torch
root=Path.cwd();ext=root/'runs/main_v1/interim_epoch025_full_guided';out=ext/'zero_floor';out.mkdir(exist_ok=True)
def save(value):
 p=out/'controller_state.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
with exclusive_run(out,'controller'):
 try:
  done=json.loads((ext/'completed.json').read_text());plan=json.loads((ext/'plan.json').read_text())
  assert done['plan_sha256']==file_sha256(ext/'plan.json')=='1cf7d0667337cdde1ecf8d777e73f2431b3e56813ea9e5bda5548e7024e65e7f'
  assert file_sha256('scripts/diagnose_zero_floor.py')=='37dc192f81806bede1cca4b36d309fbe2385967336170825d6324531d8247698'
  assert file_sha256('scripts/analyze_zero_floor.py')=='4cad8e652d6cdaaa891efa92e8cf2f91d3487ef6de10d03db869627c530f9172'
  for name,digest in done['stitched_sha256'].items():assert file_sha256(ext/name)==digest
  source=Path(done['source_dir'])
  for name,digest in plan['source_sha256'].items():assert file_sha256(source/name)==digest
  n=done['base_cells'];idx=np.load(ext/'indices.npy');pred=np.load(ext/'stitched_predictions.npy',mmap_mode='r');raw=np.load(ext/'stitched_raw.npy',mmap_mode='r')
  np.testing.assert_array_equal(pred[:n],np.load(source/'prediction.npy',mmap_mode='r'))
  np.testing.assert_array_equal(raw[:n],np.load(source/'prediction_raw.npy',mmap_mode='r'))
  np.testing.assert_array_equal(idx[:n],np.load(source/'indices.npy'))
  np.testing.assert_array_equal(pred,np.maximum(raw,0))
  assert np.isfinite(raw).all() and np.isfinite(pred).all()
  assert done['extra_seed']==62525 and done['extra_batch_size']==32 and done['full_cells']==4825
  assert done['source_options']['guidance']==2.0 and done['weights']=='ema_0.9996' and done['step']==27950
  checkpoint=torch.load(source/'model.pt',map_location='cpu',mmap=True,weights_only=False)
  data=BenchmarkData(checkpoint['config']['data_dir']);np.testing.assert_array_equal(np.sort(idx),data.val)
  del checkpoint,data,pred,raw
  save({'time':timestamp(),'pid':os.getpid(),'stage':'cpu_zero_floor','full_population_and_prefix_hashes_verified':True,'new_generation':False,'test_access':False})
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root/'src'))
  commands=[[sys.executable,'scripts/diagnose_zero_floor.py','--extension-dir',str(ext),'--outdir',str(out),'--floors','0,.05'],[sys.executable,'scripts/analyze_zero_floor.py','--study-dir',str(out)]]
  for i,command in enumerate(commands):
   with (out/f'phase_{i+1}.log').open('a') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((out/'paired_comparison.json').read_text())
  save({'time':timestamp(),'pid':os.getpid(),'stage':'completed_review_required','paired_analysis_sha256':file_sha256(out/'paired_comparison.json'),'full_cells':4825,'automatic_formal_selection':False,'training_changed':False})
  comparison=json.loads((out/'comparison.json').read_text())
  readme_event(f'第25轮固定CFG全量4825细胞的0/0.05近零处理诊断及九指标配对分析完成；DEPrec为{comparison["floor_0"]["metrics"]["DEPrec"]:.4f}/{comparison["floor_0p05"]["metrics"]["DEPrec"]:.4f}。等待综合复核，未自动更改正式设置；结果 `{out}/paired_comparison.json`。')
 except BaseException as exc:
  save({'time':timestamp(),'pid':os.getpid(),'stage':'failed','error':repr(exc)});traceback.print_exc();raise
