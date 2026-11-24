import json,os,subprocess,sys,time,hashlib
from pathlib import Path
from ocoot.runtime import exclusive_run,timestamp
out=Path('logs/preservation_epoch026');out.mkdir(exist_ok=True)
def state(value):
 p=out/'state.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
with exclusive_run(out,'watcher'):
 state({'time':timestamp(),'pid':os.getpid(),'stage':'waiting_step29068','purpose':'Preserve a matched full-data control checkpoint for possible later validation-only diagnosis; no experiment or training-policy change.'})
 for _ in range(450):
  try:cp=json.loads(Path('runs/main_v1/checkpoint_state.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(2);continue
  if cp['step']>29068:raise RuntimeError('Exact epoch26 checkpoint no longer current; no replacement snapshot or training change allowed')
  if cp['step']==29068:
   assert hashlib.sha256(Path('configs/main_selected.yaml').read_bytes()).hexdigest()=='e51d2cc9b8bb4ddc8e9dda24861f675b6a37a2486b7faffc69ed71e7f2342b52'
   command=[sys.executable,'scripts/preserve_checkpoint.py','--run-dir','runs/main_v1','--step','29068','--destination','/home/dataset-assist-0/whs/ocoo-t-artifacts/main-snapshots/epoch026/full.pt']
   with (out/'archive.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
   state({'time':timestamp(),'pid':os.getpid(),'stage':'completed','step':29068,'record':'runs/main_v1/snapshots/step_29068/preserved.json','training_changed':False});break
  time.sleep(2)
 else:raise TimeoutError('Checkpoint not reached within15minutes; no training changes')
