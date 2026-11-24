import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib
root=Path.cwd();out=root/'runs/batch_dropout_mature_trial'
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state(**extra):
 value={'time':now(),'pid':os.getpid(),'cpu_only':True,'test_access':False,**extra};p=out/'analysis_process.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def alive(pid):
 try:os.kill(pid,0);return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except (OSError,IndexError):return False
with (out/'analysis_followup.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  state(stage='waiting_for_endpoint_evaluations',controller_pid=979902)
  while not (out/'completed.json').exists():
   if not alive(979902):raise RuntimeError('Trial controller exited before complete endpoint evaluations; artifacts retained')
   time.sleep(55)
  registered=json.loads((out/'analysis_registration.json').read_text())
  for name,digest in registered['source_sha256'].items():
   if sha(root/name)!=digest:raise RuntimeError('Registered paired analysis source changed: '+name)
  command=[sys.executable,'scripts/analyze_mature_batch_dropout_trial.py','--trial-dir',str(out)]
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(root/'src'))
  with (out/'paired_analysis.log').open('a') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
   state(stage='paired_cpu_analysis',child_pid=child.pid,command=command)
   code=child.wait()
  if code:raise RuntimeError('Paired CPU analysis failed; inspect paired_analysis.log')
  result=json.loads((out/'paired_comparison.json').read_text())
  if result['script_sha256']!=registered['analysis_source_sha256']:raise RuntimeError('Paired analysis source pin mismatch')
  differences={tag:case['score_difference'] for tag,case in result['comparisons'].items()}
  state(stage='completed_review_required',differences=differences,paired_result_sha256=sha(out/'paired_comparison.json'),automatic_main_switch=False)
  from ocoot.runtime import readme_event
  text='；'.join(f'{tag}综合差{case["value"]:+.5f}[{case["ci95"][0]:+.5f},{case["ci95"][1]:+.5f}]' for tag,case in differences.items())
  readme_event('成熟batch dropout两组固定CFG九指标配对分析完成：'+text+'。仍需结合九指标取舍复核，不自动切换主模型或改正式推理；见 `runs/batch_dropout_mature_trial/paired_comparison.json`。')
 except BaseException as exc:
  state(stage='failed',error=repr(exc));traceback.print_exc();raise
