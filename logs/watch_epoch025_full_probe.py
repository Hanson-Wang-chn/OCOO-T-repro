import hashlib,json,os,shutil,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
root=Path.cwd();out=root/'runs/main_v1/interim_epoch025_full_guided';now=datetime.now(timezone(timedelta(hours=8)))
def read(path):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def alive(pid):
 try:os.kill(pid,0);return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except (OSError,IndexError):return False
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
events=[]
for line in (root/'runs/main_v1/events.jsonl').read_text().splitlines()[-100:]:
 try:events.append(json.loads(line))
 except json.JSONDecodeError:pass
train=next((e for e in events[::-1] if e.get('event')=='train'),{})
denoise=next((e for e in events[::-1] if e.get('event')=='validation'),{})
cp=read(root/'runs/main_v1/checkpoint_state.json');gen=read(out/'extra/generation_state.json');done=read(out/'completed.json')
status={'time':now.isoformat(timespec='seconds'),'main':train,'checkpoint':cp,'main_alive':alive(656512),'pipeline_alive':alive(961459),'extension_alive':alive(965660),'extra_generation':gen,'extension_completed':bool(done),'full_metrics':done.get('metrics',{}),'observer':read(root/'runs/main_v1/observation/state.json'),'main_config_unchanged':sha(root/'configs/main_selected.yaml')=='e51d2cc9b8bb4ddc8e9dda24861f675b6a37a2486b7faffc69ed71e7f2342b52','extension_source_unchanged':sha(root/'scripts/extend_validation_cache.py')=='f7619abaa9f713c1d32b50d975fc440e036a55d8c910e2e9cf2b4e8bb8f4952f','extension_plan_unchanged':sha(out/'plan.json')=='1cf7d0667337cdde1ecf8d777e73f2431b3e56813ea9e5bda5548e7024e65e7f','readme_age_seconds':time.time()-(root/'README.md').stat().st_mtime,'local_free_GiB':shutil.disk_usage(root).free/2**30,'main_event_age_seconds':(now-datetime.fromisoformat(train['time'])).total_seconds() if train else None}
status['extra_options_match']=gen.get('options')=={'noise_std':1.0,'batch_size':32,'steps':50,'guidance':2.0,'cfg_min':.1,'cfg_max':.9,'clamp':True,'seed':62525}
status['main_finished']=(root/'runs/main_v1/finished.json').exists()
p=root/'logs/session_watch_latest.json';temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(p)
p=root/'logs/active_work_snapshot.json';state=read(p);state['time']=status['time'];state['main']['latest_train']=train;state['main']['latest_checkpoint']=cp
if denoise:state['main']['latest_denoise']=denoise
state['main']['latest_observer_state']=status['observer'];state['latest_full_guided_diagnostic_state']={'pid':965660,'alive':status['extension_alive'],'extra_generation':gen,'metrics_ready':bool(status['full_metrics']),'completed':bool(done)}
state['last_server_recovery_check']={'time':'2026-11-20T09:32:58+08:00','audit':'logs/recovery_audit_20261120_0933.json','training_pid_unchanged':656512,'aux_pid_unchanged':965660,'action':'Existing jobs retained; no relaunch.'}
temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2,ensure_ascii=False));temp.replace(p)
reasons=[]
for key in ['main_alive','pipeline_alive','main_config_unchanged','extension_source_unchanged','extension_plan_unchanged','extra_options_match']:
 if not status[key]:reasons.append(key)
if done:reasons.append('extension_completed')
elif not status['extension_alive']:reasons.append('extension_exited')
if status['local_free_GiB']<10:reasons.append('low_disk')
if status['readme_age_seconds']>1600:reasons.append('readme_stale')
if status['main_event_age_seconds'] is None or status['main_event_age_seconds']>1800:reasons.append('training_stale')
print(json.dumps({'time':status['time'],'epoch':train.get('epoch'),'step':train.get('step'),'extra':gen.get('completed'),'extra_total':gen.get('total'),'readme_age':round(status['readme_age_seconds']),'free_GiB':round(status['local_free_GiB'],2),'full_metrics':status['full_metrics'],'break_reason':reasons}))
