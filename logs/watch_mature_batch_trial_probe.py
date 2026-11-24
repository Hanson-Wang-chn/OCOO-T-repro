import hashlib,json,math,os,shutil,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
root=Path.cwd();now=datetime.now(timezone(timedelta(hours=8)));out=root/'runs/batch_dropout_mature_trial';run=root/'runs/main_batch_retained_from025'
def read(path):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def alive(pid):
 try:os.kill(pid,0);return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except (OSError,IndexError):return False
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def events(path):
 result=[]
 try:lines=Path(path).read_text().splitlines()[-120:]
 except FileNotFoundError:return []
 for line in lines:
  try:result.append(json.loads(line))
  except json.JSONDecodeError:pass
 return result
def latest(records,kind):return next((e for e in records[::-1] if e.get('event')==kind),{})
main_events=events(root/'runs/main_v1/events.jsonl');trial_events=events(run/'events.jsonl')
main=latest(main_events,'train');trial=latest(trial_events,'train');process=read(out/'process.json');completed=read(out/'completed.json');cp=read(run/'checkpoint_state.json');main_cp=read(root/'runs/main_v1/checkpoint_state.json');execution=read(out/'execution_config.json')
state={'time':now.isoformat(timespec='seconds'),'main':main,'main_checkpoint':main_cp,'main_alive':alive(656512),'pipeline_alive':alive(961459),'trial':trial,'trial_validation':latest(trial_events,'validation'),'trial_checkpoint':cp,'trial_process':process,'trial_controller_alive':alive(979902),'trial_child_alive':alive(process.get('child_pid',0)) if process.get('child_pid') else False,'trial_completed':bool(completed),'main_config_unchanged':sha(root/'configs/main_selected.yaml')=='e51d2cc9b8bb4ddc8e9dda24861f675b6a37a2486b7faffc69ed71e7f2342b52','trial_plan_unchanged':sha(out/'plan.json')=='97e36ed156277baac3415e38ff75e886c344c14d9f2ef488e8b24d5f2671bc02','trial_sources_unchanged':all(sha(root/p)==h for p,h in execution.get('source_sha256',{}).items()),'README_age_seconds':time.time()-(root/'README.md').stat().st_mtime,'local_free_GiB':shutil.disk_usage(root).free/2**30}
mem={}
for line in Path('/sys/fs/cgroup/memory/memory.stat').read_text().splitlines():
 key,value=line.split();mem[key]=int(value)
state['memory_anon_GiB']=mem['rss']/2**30;state['memory_reclaimable_file_GiB']=(mem['inactive_file']+mem['active_file'])/2**30
state['oom_control']=Path('/sys/fs/cgroup/memory/memory.oom_control').read_text();state['generation']={tag:read(out/tag/'generation_state.json') for tag in ['reference','candidate']}
state['endpoint_verification']=read(out/'endpoint_independent_verification.json')
state['analysis_process']=read(out/'analysis_process.json')
reference=read(out/'control_denoise_reference.json').get(str(state['trial_validation'].get('step')),{});current=state['trial_validation'].get('ema_0.9996',{}).get('endpoint_mse');ref=reference.get('ema_0.9996',{}).get('endpoint_mse');state['trial_ema_endpoint_delta_vs_control']=current-ref if current is not None and ref is not None else None
reasons=[]
for key in ['main_alive','pipeline_alive','main_config_unchanged','trial_plan_unchanged','trial_sources_unchanged']:
 if not state[key]:reasons.append(key)
if completed:reasons.append('trial_completed_review_required')
elif not state['trial_controller_alive']:reasons.append('trial_controller_exited')
if process.get('stage')=='training' and not state['trial_child_alive'] and cp.get('step',0)<29068:reasons.append('trial_child_exited_before_budget')
if trial.get('step',27950)>29068:reasons.append('trial_exceeded_registered_budget')
if state['trial_ema_endpoint_delta_vs_control'] is not None and state['trial_ema_endpoint_delta_vs_control']>.001:reasons.append('denoise_instability_review')
if state['local_free_GiB']<10:reasons.append('low_disk')
if state['README_age_seconds']>1600:reasons.append('readme_stale')
if state['memory_anon_GiB']>90:reasons.append('high_anonymous_memory')
if '\noom_kill 0\n' not in '\n'+state['oom_control']:reasons.append('oom_kill_count')
if main and (now-datetime.fromisoformat(main['time'])).total_seconds()>1800:reasons.append('main_event_stale')
if process.get('stage')=='training' and trial and (now-datetime.fromisoformat(trial['time'])).total_seconds()>1800:reasons.append('trial_event_stale')
state['break_reason']=reasons
p=root/'logs/session_watch_latest.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(state,indent=2));t.replace(p)
p=root/'logs/active_work_snapshot.json';s=read(p);s['time']=state['time'];s['main']['latest_train']=main;s['main']['latest_checkpoint']=main_cp
validation=latest(main_events,'validation')
if validation:s['main']['latest_denoise']=validation
s['latest_mature_batch_dropout_trial_state']={'controller_alive':state['trial_controller_alive'],'process':process,'latest_train':trial,'latest_validation':state['trial_validation'],'checkpoint':cp,'completed':bool(completed),'sources_unchanged':state['trial_sources_unchanged']}
publications=list((root/'logs').glob('readme_*.published.json'))
if publications:
 publication=max(publications,key=lambda p:p.stat().st_mtime);s['readme_last_structured_publication']=str(publication.relative_to(root))
 pending=root/'logs/readme_mature_batch_dropout_endpoint_update.json'
 s['readme_pending_edit']=str(pending.relative_to(root)) if pending.exists() and not pending.with_suffix('.published.json').exists() else None
observer=read(root/'runs/main_v1/observation/state.json')
if observer:s['main']['latest_observer_state']=observer;s['main']['completed_epoch']=observer.get('completed_epoch',s['main']['completed_epoch'])
if s.get('supplemental_active',{}).get('controller_pid')==979902:
 s['supplemental_active']['status']='completed_review_required' if completed else process.get('stage','unknown')
 s['supplemental_active']['child_pid']=process.get('child_pid')
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(s,indent=2,ensure_ascii=False));t.replace(p)
print(json.dumps({'time':state['time'],'main_epoch':main.get('epoch'),'main_step':main.get('step'),'trial_step':trial.get('step'),'trial_checkpoint_step':cp.get('step'),'trial_validation_step':state['trial_validation'].get('step'),'trial_ema_endpoint':state['trial_validation'].get('ema_0.9996',{}).get('endpoint_mse'),'trial_endpoint_delta_vs_control':state['trial_ema_endpoint_delta_vs_control'],'phase':process.get('stage'),'generation':{k:v.get('completed') for k,v in state['generation'].items()},'free_GiB':round(state['local_free_GiB'],2),'anon_GiB':round(state['memory_anon_GiB'],2),'readme_age':round(state['README_age_seconds']),'break_reason':reasons}))
