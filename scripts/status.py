"""Read-only local progress; no heavy ML imports."""
import json
import argparse
from pathlib import Path
import subprocess
import os


def process_state(pidfile,script):
    if not pidfile.exists():return {}
    try:
        pid=int(pidfile.read_text())
        alive=script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes() and Path(os.readlink(f'/proc/{pid}/cwd'))==Path.cwd()
        return {'pid':pid,'alive':alive}
    except (OSError,ValueError):return {'alive':False}


def status(run_dir=None):
    candidates=sorted((p for p in Path('runs').glob('*/events.jsonl') if not p.parent.name.startswith('engine_smoke')),
                 key=lambda p:p.stat().st_mtime,reverse=True)
    files=[Path(run_dir)/'events.jsonl'] if run_dir else candidates
    files=[p for p in files if p.is_file()]
    if not files:return {'status':'no training events'}
    run=files[0].parent
    events=[]
    for line in files[0].read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    train=next((e for e in reversed(events) if e['event']=='train'),{})
    val=next((e for e in reversed(events) if e['event']=='validation'),None)
    generation_val=next((e for e in reversed(events) if e['event']=='validation_generation'),None)
    diagnostic=next((e for e in reversed(events) if e['event']=='diagnostic_generation'),None)
    result={'run':str(run),'latest':events[-1] if events else None,'train':train,'validation':val,
            'validation_generation':generation_val,'diagnostic_generation':diagnostic,
            'finished':(run/'finished.json').exists()}
    ck=run/'checkpoint_state.json'
    if ck.exists():
        try:result['checkpoint']=json.loads(ck.read_text())
        except json.JSONDecodeError:pass  # metadata may be in the middle of a write
    pidfile=run/'training.pid'
    training_process=process_state(pidfile,'scripts/train.py')
    if 'pid' in training_process:result['pid']=training_process['pid']
    if training_process:result['pid_alive']=training_process['alive']
    pipeline=run/'pipeline_state.json'
    if pipeline.exists():
        result['pipeline']=json.loads(pipeline.read_text())
        result['pipeline_process']=process_state(run/'pipeline.pid','scripts/run_experiment.py')
    generation=sorted(run.rglob('generation_state.json'),key=lambda p:p.stat().st_mtime)
    if generation:
        result['generation']=json.loads(generation[-1].read_text())
        result['generation_path']=str(generation[-1].parent)
    result['active']=bool(result.get('pid_alive') or result.get('pipeline_process',{}).get('alive'))
    # Pilots may run beside the main model. Make that visible instead of
    # allowing the most recently written log to hide the other training job.
    result['other_active_training']=[]
    for path in candidates:
        if path.parent==run:continue
        process=process_state(path.parent/'training.pid','scripts/train.py')
        if not process.get('alive'):continue
        latest=None
        for line in reversed(path.read_text().splitlines()):
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if event.get('event')=='train':latest=event;break
        result['other_active_training'].append({'run':str(path.parent),**process,'latest_train':latest})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir')
    args=parser.parse_args()
    print(json.dumps(status(args.run_dir),ensure_ascii=False,indent=2))

