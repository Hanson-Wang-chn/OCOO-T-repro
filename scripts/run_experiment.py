"""Resume the full-data train -> calibrate -> validate -> test -> report workflow.

Full validation freezes the evaluation configuration and records quality
shortfalls. The final report compares the measured test scores with the paper.
"""
# 第三轮全量复现的总控入口：在同一个 run_dir 中按顺序完成全量训练、
# 仅用验证集校准采样参数、完整验证与质量门控、正式测试和报告。
# 各阶段均可根据已落盘的完成标记/产物续跑；验证不达门槛时默认停止在测试前，
# 避免把未通过检查的配置直接用于正式测试，也不把测试集用于调参。
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
from ocoot.postprocess import DEFAULT_THRESHOLDS,allow_validation_shortfall,quality_checks
from ocoot.runtime import file_sha256,gpu_command,log_event,readme_event,timestamp,exclusive_run


def training_lock_held(run):
    # 通过训练进程持有的操作系统文件锁判断是否已有训练在运行；
    # 单看 lock 文件是否存在不可靠，因为训练结束后文件本身可能仍保留。
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
    # 若已有独立训练进程占用同一 run，则等待它释放锁，再检查训练完成标记；
    # 避免总控入口与 watch/resume 进程同时启动两份训练。
    attached=False
    while training_lock_held(run):
        attached=True
        time.sleep(poll_seconds)
    return attached


def main():
    # 命令行开关覆盖校准、并行度和验证质量线；未提供 postprocess plan 时，
    # 沿用原始 clamp0 验证/测试流程，不启用额外的全验证后处理选择。
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--postprocess-plan',help='Opt-in registered full-validation zero-floor selection; default original clamp0 workflow')
    p.add_argument('--gpu-workers',type=int,default=1,help='Local GPU processes for training and generation')
    p.add_argument('--calibration-scales',help='Comma-separated positive guidance scales')
    p.add_argument('--calibration-mse-weight',type=float,default=0.0,help='MSE penalty for calibration selection')
    p.add_argument('--compare-windows',action='store_true',help='Screen positive scales in both guidance windows')
    p.add_argument('--min-val-pdcorr',type=float,default=.35);p.add_argument('--min-val-pds',type=float,default=.75)
    p.add_argument('--min-val-pds-each',type=float,default=.70)
    p.add_argument('--max-val-mse',type=float,default=.02);p.add_argument('--max-val-mae',type=float,default=.09)
    p.add_argument('--min-val-deover',type=float,default=.18);p.add_argument('--min-val-deprec',type=float,default=.18)
    p.add_argument('--min-val-diragr',type=float,default=.70);a=p.parse_args()
    if a.gpu_workers < 1:p.error('gpu-workers must be positive')
    if not math.isfinite(a.calibration_mse_weight) or a.calibration_mse_weight < 0:
        p.error('calibration-mse-weight must be nonnegative and finite')
    cfg=yaml.safe_load(open(a.config));run=Path(cfg['run_dir']);run.mkdir(exist_ok=True,parents=True)
    # 此入口只接受全量实验；子集 pilot 的模型和指标不能替代正式复现。
    if cfg['training'].get('train_subset'):raise ValueError('This workflow is for full-data experiments, not pilots')
    thresholds={key:getattr(a,key) for key in DEFAULT_THRESHOLDS}
    report_shortfall=False
    if a.postprocess_plan:
        # 计划文件可登记“未过门槛仍报告短缺”的处理规则，但不能借此改变原质量线。
        postprocess_plan=json.loads(Path(a.postprocess_plan).read_text())
        if postprocess_plan['quality_thresholds']!=thresholds or thresholds!=DEFAULT_THRESHOLDS:
            raise ValueError('Optional postprocessing must preserve every original validation threshold')
        report_shortfall=allow_validation_shortfall(postprocess_plan)
    with exclusive_run(run,'pipeline'):
        (run/'pipeline.pid').write_text(str(os.getpid()))
        def state(stage,**extra):
            # 原子替换状态快照，并追加事件日志，供断点检查与运行记录使用。
            value={'time':timestamp(),'pid':os.getpid(),'stage':stage,'config':str(Path(a.config).resolve()),**extra}
            if a.postprocess_plan:value['postprocess_plan_sha256']=file_sha256(a.postprocess_plan)
            tmp=run/'pipeline_state.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(run/'pipeline_state.json')
            log_event(run/'events.jsonl',{'event':'pipeline','stage':stage,**extra})
        def execute(args,cpu_only=False,parallel=False):
            # 所有子脚本统一写入 pipeline.log；训练/生成按指定 GPU worker 数启动，
            # 只做文件/缓存处理的阶段则可明确隐藏 CUDA。
            with open(run/'pipeline.log','a') as log:
                environment=dict(os.environ,CUDA_VISIBLE_DEVICES='') if cpu_only else None
                subprocess.run(gpu_command(args,a.gpu_workers if parallel else 1),
                               stdout=log,stderr=log,check=True,env=environment)
        def evaluate(selected,split,outdir):
            # 完整评估严格使用已选 checkpoint、CFG 区间及截断设置；
            # 校准产物将这些设置冻结，测试阶段不再重新搜索。
            args=['scripts/evaluate.py','--checkpoint',selected['checkpoint'],'--split',split,'--outdir',str(outdir),
                  '--batch-size',str(selected['batch_size']),'--cfg-min',str(selected['cfg_min']),'--cfg-max',str(selected['cfg_max']),'--de']
            if selected['guidance'] is not None:args+=['--guidance',str(selected['guidance'])]
            if selected['clamp']:args+=['--clamp']
            execute(args,parallel=True)
        try:
            # 先等待/执行训练。finished 标记用于幂等续跑；训练意外中断时保留断点，
            # 不继续做推理，也不将“有 checkpoint”误当成训练已完整结束。
            if not (run/'finished.json').exists():
                state('training',attached_existing_training=training_lock_held(run))
                wait_for_existing_training(run)
                if not (run/'finished.json').exists():execute(['scripts/train.py','--config',a.config],parallel=True)
                if not (run/'finished.json').exists():raise RuntimeError('Training stopped before completion; checkpoint retained')
            checkpoint=run/'best_generation.pt'
            if not checkpoint.exists():raise RuntimeError('No EMA generation checkpoint has been selected')
            calibration=run/'calibration'
            if not (calibration/'selected.json').exists():
                # 在验证集上筛选正尺度双 CFG；可按登记尺度和两个时间窗筛选，
                # MSE 权重仅影响验证候选排序，不改变训练损失。
                state('calibrating_validation')
                args=['scripts/calibrate.py','--checkpoint',str(checkpoint),'--outdir',str(calibration)]
                if a.calibration_scales:args+=['--scales',a.calibration_scales]
                if a.calibration_mse_weight:args+=['--mse-weight',str(a.calibration_mse_weight)]
                if a.compare_windows:args+=['--compare-windows']
                execute(args,parallel=True)
            selection_file=calibration/'selected.json'
            # 若 calibration 已存在，仍核对其 MSE 选择权重与本次请求一致，
            # 防止将旧设置下的选择误用于本次正式复现。
            if json.loads(selection_file.read_text()).get('mse_weight', 0.0) != a.calibration_mse_weight:
                raise ValueError('The requested MSE weight differs from the completed calibration')
            if a.postprocess_plan:
                # 登记的可选流程在完整验证原始缓存上比较后处理候选；短缺规则及
                # 选择结果均写入独立目录，后续测试只采用已冻结的 selection_file。
                postprocessing=calibration/'postprocessing'
                state('selecting_full_validation_postprocess')
                try:
                    execute(['scripts/select_validation_postprocess.py','--calibration',str(calibration),
                             '--outdir',str(postprocessing),'--plan',a.postprocess_plan],cpu_only=True)
                except subprocess.CalledProcessError as error:
                    if error.returncode!=2:raise
                    review_file=postprocessing/'review.json'
                    if not review_file.exists() or json.loads(review_file.read_text()).get('selected') is not None:raise
                    state('validation_needs_review',postprocess_review=str(postprocessing/'review.json'))
                    readme_event(f'`{run}` 的完整0/.05验证候选均未通过原质量检查，保留结果并停在测试前；见 `{postprocessing}/review.json`。')
                    raise SystemExit(2)
                selection_file=postprocessing/'selected.json'
            selected=json.loads(selection_file.read_text())
            # 校验选中 checkpoint 未被替换，并强制论文规定的双 CFG 正尺度推理策略；
            # 普通条件分支只用于诊断，不能作为正式复现配置。
            if selected['checkpoint_sha256']!=file_sha256(selected['checkpoint']):raise ValueError('Selected checkpoint changed')
            w=selected.get('guidance')
            if (selected.get('formal_guidance_policy')!='literal_dual_cfg_positive_scale' or
                    w is None or not math.isfinite(w) or w<=0):
                raise ValueError('Formal reproduction requires literal dual-CFG with a positive scale; rerun validation calibration')
            validation=run/('validation_full_postprocess' if a.postprocess_plan else 'validation_full')
            reuse=(selected.get('cap') is None and selected.get('cache_provenance') is not None
                   and ((validation/'promotion_config.json').exists()
                        or not (validation/'generation_state.json').exists()))
            if a.postprocess_plan:
                # 若候选来自完整原始精炼缓存，只需按冻结规则提升/整理缓存，
                # 不重复生成；否则原流程可复用校准缓存或进行完整验证采样。
                if selected.get('cap') is not None or selected.get('cache_provenance') is None:
                    raise ValueError('Optional postprocessing requires a complete raw refinement cache')
                state('full_validation_de_reusing_calibration_postprocess')
                execute(['scripts/promote_validation_cache.py','--calibration',str(calibration),
                         '--selection',str(selection_file),'--outdir',str(validation)],cpu_only=True)
            elif reuse:
                state('full_validation_de_reusing_calibration')
                execute(['scripts/promote_validation_cache.py','--calibration',str(calibration),'--outdir',str(validation)],cpu_only=True)
            elif not (validation/'metrics/summary.json').exists():
                state('full_validation_de');evaluate(selected,'validation',validation)
            # 重新对完整验证指标执行质量检查并保存审查记录。默认任一项失败即停在测试前；
            # 只有显式登记的短缺报告规则允许继续，并会在记录中保留失败项目。
            metrics=json.loads((validation/'metrics/summary.json').read_text())
            checks=quality_checks(metrics,thresholds)
            validation_review={'metrics':metrics,'checks':checks,
                               'thresholds':thresholds,
                               'note':'Validation-only engineering checks, not a declaration of agreement with paper test performance.'}
            review_path=run/('validation_postprocess_review.json' if a.postprocess_plan else 'validation_review.json')
            review_path.write_text(json.dumps(validation_review,indent=2))
            if not all(checks.values()):
                if not report_shortfall:
                    state('validation_needs_review',**validation_review)
                    readme_event(f'`{run}` 的完整验证未通过预设质量检查，保留结果以便审查。指标见 `{validation}/metrics/summary.json`。')
                    raise SystemExit(2)
                state('validation_shortfall_reported',**validation_review)
                missing=', '.join(key for key,passed in checks.items() if not passed)
                readme_event(f'`{run}` 的完整验证未达到 {missing} 的检查线。配置已按登记规则冻结，正式测试量化表 3 差距。')
            test=run/('test_full_postprocess' if a.postprocess_plan else 'test_full')
            # 测试只运行一次冻结配置；可选后处理路径同时保留同批原始生成的 clamp0 对照。
            if a.postprocess_plan:
                state('full_test_de_with_frozen_postprocess')
                args=['scripts/evaluate_postprocess.py','--selection',str(selection_file),
                      '--validation-dir',str(validation),'--baseline-outdir',str(run/'test_full_clamp0'),
                      '--outdir',str(test)]
                if a.gpu_workers>1:args+=['--gpu-workers',str(a.gpu_workers)]
                execute(args)
            elif not (test/'metrics/summary.json').exists():
                state('full_test_de');evaluate(selected,'test',test)
            # 报告和进度图由已完成的正式测试产物生成，不据测试集反向选择超参数。
            state('reporting');execute(['scripts/report.py','--evaluation-dir',str(test),'--outdir',str(run/'results')])
            if a.postprocess_plan:
                execute(['scripts/report.py','--evaluation-dir',str(run/'test_full_clamp0'),'--outdir',str(run/'results_clamp0')])
            execute(['scripts/plot_progress.py',str(run)])
            summary=json.loads((test/'metrics/summary.json').read_text())
            state('evaluated',test_metrics=summary,report=str(run/'results/REPORT.md'))
            readme_event(f'`{run}` 的完整测试与报告已产出：`{run}/results/REPORT.md`。以下是实测结果，是否达到论文水平需结合全部指标审查；当前不会自动宣称复现成功。')
        except Exception as error:
            # 记录失败阶段和错误类型后原样抛出；保留 checkpoint、采样缓存和日志供排查/续跑。
            state('failed',error=f'{type(error).__name__}: {error}')
            readme_event(f'`{run}` 工作流发生错误：{type(error).__name__}；检查点和已完成采样保留，继续排查时查看 `{run}/pipeline.log`。')
            raise

if __name__=='__main__':main()

