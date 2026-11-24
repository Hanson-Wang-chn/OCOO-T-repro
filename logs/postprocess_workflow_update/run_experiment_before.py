"""Resume the full-data train -> calibrate -> validate -> test -> report workflow.

Validation quality gates catch obvious failures before an expensive test run.
The final report records measured performance without declaring paper parity.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from ocoot.runtime import file_sha256,log_event,readme_event,timestamp,exclusive_run


def training_lock_held(run):
    path=Path(run)/'training.lock'
    if not path.exists():return False
    with path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        finally:
            # Unlocking a descriptor that did not acquire the lock is harmless.
            fcntl.flock(lock,fcntl.LOCK_UN)
    return False


def wait_for_existing_training(run,poll_seconds=10):
    attached=False
    while training_lock_held(run):
        attached=True
        time.sleep(poll_seconds)
    return attached


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--min-val-pdcorr',type=float,default=.35);p.add_argument('--min-val-pds',type=float,default=.75)
    p.add_argument('--min-val-pds-each',type=float,default=.70)
    p.add_argument('--max-val-mse',type=float,default=.02);p.add_argument('--max-val-mae',type=float,default=.09)
    p.add_argument('--min-val-deover',type=float,default=.18);p.add_argument('--min-val-deprec',type=float,default=.18)
    p.add_argument('--min-val-diragr',type=float,default=.70);a=p.parse_args()
    cfg=yaml.safe_load(open(a.config));run=Path(cfg['run_dir']);run.mkdir(exist_ok=True,parents=True)
    if cfg['training'].get('train_subset'):raise ValueError('This workflow is for full-data experiments, not pilots')
    with exclusive_run(run,'pipeline'):
        (run/'pipeline.pid').write_text(str(os.getpid()))
        def state(stage,**extra):
            value={'time':timestamp(),'pid':os.getpid(),'stage':stage,'config':str(Path(a.config).resolve()),**extra}
            tmp=run/'pipeline_state.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(run/'pipeline_state.json')
            log_event(run/'events.jsonl',{'event':'pipeline','stage':stage,**extra})
        def execute(args,cpu_only=False):
            with open(run/'pipeline.log','a') as log:
                environment=dict(os.environ,CUDA_VISIBLE_DEVICES='') if cpu_only else None
                subprocess.run([sys.executable,*args],stdout=log,stderr=log,check=True,env=environment)
        def evaluate(selected,split,outdir):
            args=['scripts/evaluate.py','--checkpoint',selected['checkpoint'],'--split',split,'--outdir',str(outdir),
                  '--batch-size',str(selected['batch_size']),'--cfg-min',str(selected['cfg_min']),'--cfg-max',str(selected['cfg_max']),'--de']
            if selected['guidance'] is not None:args+=['--guidance',str(selected['guidance'])]
            if selected['clamp']:args+=['--clamp']
            execute(args)
        try:
            if not (run/'finished.json').exists():
                state('training',attached_existing_training=training_lock_held(run))
                wait_for_existing_training(run)
                if not (run/'finished.json').exists():execute(['scripts/train.py','--config',a.config])
                if not (run/'finished.json').exists():raise RuntimeError('Training stopped before completion; checkpoint retained')
            checkpoint=run/'best_generation.pt'
            if not checkpoint.exists():raise RuntimeError('No EMA generation checkpoint has been selected')
            calibration=run/'calibration'
            if not (calibration/'selected.json').exists():
                state('calibrating_validation');execute(['scripts/calibrate.py','--checkpoint',str(checkpoint),'--outdir',str(calibration)])
            selected=json.loads((calibration/'selected.json').read_text())
            if selected['checkpoint_sha256']!=file_sha256(selected['checkpoint']):raise ValueError('Selected checkpoint changed')
            w=selected.get('guidance')
            if (selected.get('formal_guidance_policy')!='literal_dual_cfg_positive_scale' or
                    w is None or not math.isfinite(w) or w<=0):
                raise ValueError('Formal reproduction requires literal dual-CFG with a positive scale; rerun validation calibration')
            validation=run/'validation_full'
            reuse=(selected.get('cap') is None and selected.get('cache_provenance') is not None
                   and ((validation/'promotion_config.json').exists()
                        or not (validation/'generation_state.json').exists()))
            if reuse:
                state('full_validation_de_reusing_calibration')
                execute(['scripts/promote_validation_cache.py','--calibration',str(calibration),'--outdir',str(validation)],cpu_only=True)
            elif not (validation/'metrics/summary.json').exists():
                state('full_validation_de');evaluate(selected,'validation',validation)
            metrics=json.loads((validation/'metrics/summary.json').read_text())
            pds=sum(metrics[k] for k in ['PDS_L1','PDS_L2','PDS_cos'])/3
            checks={'PDCorr':metrics['PDCorr']>=a.min_val_pdcorr,'mean_PDS':pds>=a.min_val_pds,
                    'MSE':metrics['MSE']<=a.max_val_mse,'MAE':metrics['MAE']<=a.max_val_mae,
                    'DEOver':metrics['DEOver']>=a.min_val_deover,'DEPrec':metrics['DEPrec']>=a.min_val_deprec,
                    'DirAgr':metrics['DirAgr']>=a.min_val_diragr,
                    **{k:metrics[k]>=a.min_val_pds_each for k in ['PDS_L1','PDS_L2','PDS_cos']}}
            validation_review={'metrics':metrics,'checks':checks,
                               'thresholds':{k:v for k,v in vars(a).items() if k!='config'},
                               'note':'Validation-only engineering checks, not a declaration of agreement with paper test performance.'}
            (run/'validation_review.json').write_text(json.dumps(validation_review,indent=2))
            if not all(checks.values()):
                state('validation_needs_review',**validation_review)
                readme_event(f'`{run}` 的完整验证未通过预设质量检查，保留全部结果并停止在测试集之前；需排查/优化。指标见 `{validation}/metrics/summary.json`。')
                raise SystemExit(2)
            test=run/'test_full'
            if not (test/'metrics/summary.json').exists():
                state('full_test_de');evaluate(selected,'test',test)
            state('reporting');execute(['scripts/report.py','--evaluation-dir',str(test),'--outdir',str(run/'results')])
            execute(['scripts/plot_progress.py',str(run)])
            summary=json.loads((test/'metrics/summary.json').read_text())
            state('evaluated',test_metrics=summary,report=str(run/'results/REPORT.md'))
            readme_event(f'`{run}` 的完整测试与报告已产出：`{run}/results/REPORT.md`。以下是实测结果，是否达到论文水平需结合全部指标审查；当前不会自动宣称复现成功。')
        except Exception as error:
            state('failed',error=f'{type(error).__name__}: {error}')
            readme_event(f'`{run}` 工作流发生错误：{type(error).__name__}；检查点和已完成采样保留，继续排查时查看 `{run}/pipeline.log`。')
            raise

if __name__=='__main__':main()

