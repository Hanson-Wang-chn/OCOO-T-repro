"""Run and supervise the registered round-four reproduction from the repository root."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
from statistics import median
import subprocess
import time

import torch
import yaml

from calibrate import write_json_atomic
from ocoot.distributed import live_worker_pids
from ocoot.data import BenchmarkData
from ocoot.postprocess import DEFAULT_THRESHOLDS, choose_validation_candidate, selection_score
from ocoot.runtime import edit_readme, exclusive_run, file_sha256, flush_readme, gpu_command, readme_event, timestamp
from repro4_analysis import (age_cutoff, batch_rms, case_metrics, overfit_distance, paired_bootstrap,
                             perturbation_effective_rank, read_json, verify_and_compare_test, verify_training, vu_error)
from report import PAPER
from run_experiment import training_lock_held


PHASE0 = Path('runs/repro4_phase0')
PILOTS = Path('runs/repro4_pilots')
MAIN = Path('runs/repro4_main')
PRIOR = Path('runs/repro3_main')
ALL_GPUS = (0, 1, 2, 3)
SCALES = (.65, .75, .85, .95, 1.05, 1.15, 1.25)
WINDOWS = ((.05, .75), (.05, .85), (.1, .8), (.1, .9), (0., 1.))
MSE_WEIGHT = 10.


def update_config(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if yaml.safe_load(path.read_text()) != value:
            raise ValueError(f'Registered configuration differs at {path}')
    else:
        temporary = path.with_suffix('.yaml.tmp')
        temporary.write_text(yaml.safe_dump(value, sort_keys=False))
        temporary.replace(path)


def postprocess_plan():
    return {'floors': [0., .05], 'quality_thresholds': DEFAULT_THRESHOLDS,
            'allow_validation_shortfall': False,
            'scope': 'Round four full validation and one frozen full test on Replogle–Nadig.',
            'calibration': {'scales': list(SCALES), 'windows': [list(window) for window in WINDOWS],
                            'screen_cap': 16, 'refine_top': 3, 'seed': 4242, 'batch_size': 32,
                            'mse_weight': MSE_WEIGHT}}


@dataclass(frozen=True)
class Job:
    name: str
    arguments: tuple
    log: Path
    gpus: tuple = ALL_GPUS
    training_dir: Path | None = None
    distributed: bool = True


def run_events(run):
    path = run / 'events.jsonl'
    if not path.exists():
        return []
    events = []
    with path.open() as stream:
        for line in stream:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a writer may be appending the final line
    return events


def status_line(job):
    if job.training_dir is None:
        folder = job.log.parent
        results = folder / 'all_results.json'
        if results.exists():
            rows = read_json(results)
            if rows:
                return f'{job.name}，已记录 {len(rows)} 组，最近一组 {rows[-1]["tag"]}'
        stage = folder / 'pipeline_state.json'
        if stage.exists():
            return f'{job.name}，阶段 {read_json(stage)["stage"]}'
        return job.name
    run = job.training_dir
    events = run_events(run)
    if not events:
        return f'{job.name} 正在初始化'
    latest = {event['event']: event for event in events}
    checkpoint = run / 'checkpoint_state.json'
    step = read_json(checkpoint)['step'] if checkpoint.exists() else 0
    parts = [f'{job.name} 已保存 step {step:,}']
    if 'train' in latest:
        row = latest['train']
        parts.append(f'E{row["epoch"]:.3f}，step {row["step"]:,}，loss {row["loss"]:.5f}，'
                     f'lr {row["lr"]:.3g}，grad_norm {row["grad_norm"]:.5f}，'
                     f'{row["cells_per_sec"]:.1f} cells/s，峰值 {row["peak_gpu_GB"]:.2f} GB')
    if 'validation' in latest:
        row = latest['validation']
        gap = row.get('ema_to_raw_gap')
        gap_text = f'，相对差 {gap:+.2%}' if gap is not None else ''
        parts.append(f'去噪验证 step {row["step"]:,} raw {row["raw"]["endpoint_mse"]:.6f}，'
                     f'EMA {row["ema_0.9996"]["endpoint_mse"]:.6f}{gap_text}')
    if 'validation_generation' in latest:
        row = latest['validation_generation']
        parts.append(f'E{row["epoch"]} 生成 PDCorr {row["PDCorr"]:.4f}，PDS_L2 {row["PDS_L2"]:.4f}，'
                     f'MSE {row["MSE"]:.5f}，S {row["selection_score"]:.4f}')
    return '。'.join(parts)


def main_health(epoch, emse, raw, generation):
    if 'PDCorr' not in generation:
        if epoch == 1 and raw is not None:
            return '异常' if raw > .15 else '警告' if raw > .125 else '正常'
        return '观察'
    pdcorr, pds, mse = [generation[k] for k in ('PDCorr', 'PDS_L2', 'MSE')]
    if epoch == 5:
        return '警告' if raw is not None and raw > .106 else '正常'
    if epoch == 10:
        return '警告' if pdcorr < .26 or (raw and emse > raw * 1.03) else '正常'
    if epoch == 15:
        if pds < .65:
            return '异常'
        return '警告' if pdcorr < .29 or (raw and emse > raw) else '正常'
    if 30 <= epoch < 45:
        if pdcorr < .32 or mse > .014:
            return '异常'
        return '达预期' if pdcorr >= .35 and mse <= .0105 else '观察'
    if epoch == 45:
        return '达预期' if pdcorr >= .365 else '观察'
    if epoch >= 60:
        return '达预期' if pdcorr >= .37 else '观察'
    return '观察'


class Supervisor:
    def __init__(self):
        self.next_update = time.monotonic() + 15 * 60
        self.training_checks = set()
        self.generation_checks = set()
        self.probe_processes = {}
        self.started = {}

    def launch(self, job):
        job.log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(map(str, job.gpus)),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   TORCHINDUCTOR_COMPILE_THREADS='6', PYTHONUNBUFFERED='1')
        workers = max(1, len(job.gpus)) if job.distributed else 1
        command = gpu_command(list(map(str, job.arguments)), workers)
        with job.log.open('a') as log:
            log.write(json.dumps({'time': timestamp(), 'job': job.name, 'gpus': job.gpus,
                                  'command': command}, ensure_ascii=False) + '\n')
            log.flush()
            started = time.time()
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        self.started[process.pid] = started
        return process

    def inspect_training(self, job, process, gpus, free):
        run = job.training_dir
        if run is None or process.poll() is not None or (run / 'finished.json').exists():
            return
        execution_path = run / 'training_execution.json'
        if not execution_path.exists() or execution_path.stat().st_mtime < self.started[process.pid] - 2:
            return
        execution = read_json(execution_path)
        events = run_events(run)
        start = execution['start_step']
        run_id = (run, execution['time'])
        if run_id not in self.training_checks:
            if (execution['world_size'] != len(job.gpus) or execution['global_batch'] != 512
                    or execution['dynamo_ddp_optimizer'] is not False or not training_lock_held(run)):
                raise RuntimeError(f'Unexpected training execution at {execution_path}')
            live_worker_pids(execution['workers'])
            losses = [event for event in events if event['event'] == 'train' and event['step'] > start][:10]
            if len(losses) == 10:
                rate = median(event['cells_per_sec'] for event in losses[1:])
                peak = max(event['peak_gpu_GB'] for event in losses)
                readme_event(f'`{run}` 启动核验完成，{execution["world_size"]} 卡，累积 '
                             f'{execution["accumulation_by_rank"]}，全局 batch 512，关闭 Dynamo DDP 分图。'
                             f'首批训练记录的稳定吞吐 {rate:.1f} cells/s，rank 0 峰值 {peak:.2f} GB。')
                self.training_checks.add(run_id)

        validations = {event['step']: event for event in events if event['event'] == 'validation'}
        generated = {event['epoch']: event for event in events if event['event'] == 'validation_generation'}
        if run == MAIN:
            for epoch in sorted({1, *generated}):
                step = epoch * 1118
                if step not in validations or (run, epoch) in self.generation_checks:
                    continue
                event = validations[step]
                ema, raw = event['ema_0.9996']['endpoint_mse'], event['raw']['endpoint_mse']
                generation = generated.get(epoch, {})
                severity = main_health(epoch, ema, raw, generation)
                review = {'time': timestamp(), 'epoch': epoch, 'severity': severity, 'validation': event,
                          'validation_generation': generation, 'test_access': False}
                with (run / 'health_milestones.jsonl').open('a') as stream:
                    stream.write(json.dumps(review, allow_nan=False) + '\n')
                readme_event(f'主训练 E{epoch} 体检为{severity}，EMA endpoint MSE {ema:.6f}，raw {raw:.6f}。'
                             f'生成记录与门槛检查见 `{run}/health_milestones.jsonl`。')
                self.generation_checks.add((run, epoch))
            self.launch_main_probe(generated, gpus)

        checkpoint = run / 'checkpoint_state.json'
        event_log = run / 'events.jsonl'
        now = time.time()
        if checkpoint.exists() and event_log.exists():
            step = read_json(checkpoint)['step']
            budget_config = yaml.safe_load((run / 'config.yaml').read_text())['training']
            epoch, offset = divmod(step, 1118)
            at_generation = (offset == 0 and epoch > 0 and epoch not in generated
                             and epoch % budget_config['val_generation_epochs'] == 0)
            idle = all(gpus[gpu][1] == 0 for gpu in job.gpus)
            stale = now - max(event_log.stat().st_mtime, checkpoint.stat().st_mtime) > 20 * 60
            if stale and idle and not at_generation:
                readme_event(f'`{run}` 20 分钟无事件且训练 GPU 空闲，已向本次训练 rank 发送 SIGTERM，'
                             '等待保存后核对断点。')
                for pid in live_worker_pids(execution['workers']):
                    os.kill(pid, signal.SIGTERM)
                raise RuntimeError(f'Suspected DDP hang at {run}; see {job.log}')
        if free < 40:
            readme_event(f'工作分区只剩 {free:.1f} GiB，向 `{run}` 的训练 rank 发送 SIGTERM 保存断点。')
            for pid in live_worker_pids(execution['workers']):
                os.kill(pid, signal.SIGTERM)
            raise RuntimeError('Less than 40 GiB remains in the training workspace')

    def launch_main_probe(self, generated, gpus):
        for epoch, process in list(self.probe_processes.items()):
            if process is not None and process.poll() is not None:
                if process.returncode != 0:
                    readme_event(f'主训练 E{epoch} 在线探针退出码 {process.returncode}，保存日志并安排补跑。')
                self.probe_processes[epoch] = None
        if any(process is not None for process in self.probe_processes.values()):
            return
        eligible = [gpu for gpu in reversed(ALL_GPUS) if gpus[gpu][0] < 27500]
        if not eligible:
            return
        for epoch in sorted(generated):
            out = MAIN / 'velocity_probe' / f'epoch_{epoch:03d}'
            if epoch % 10 or epoch in self.probe_processes or (out / 'summary.json').exists():
                continue
            checkpoint = MAIN / f'validation_epoch_{epoch:03d}' / 'model.pt'
            self.probe_processes[epoch] = self.launch(velocity_job(checkpoint, out, eligible[0]))
            break

    def monitor(self, jobs, processes):
        now = time.monotonic()
        raw_gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
        gpus = {index: (memory, utilization) for index, memory, utilization in
                (tuple(map(int, row.split(','))) for row in raw_gpu.splitlines())}
        free = shutil.disk_usage('.').free / 2**30
        for job, process in zip(jobs, processes):
            self.inspect_training(job, process, gpus, free)
        if now >= self.next_update:
            gpu_status = [f'GPU {index} {memory} MiB / {utilization}%'
                          for index, (memory, utilization) in gpus.items()]
            lines = [status_line(job) for job, process in zip(jobs, processes) if process.poll() is None]
            readme_event('巡检。' + '。'.join(lines + ['、'.join(gpu_status), f'工作分区可用 {free:.0f} GiB']) + '。')
            flush_readme()
            self.next_update = now + 15 * 60

    def run(self, *jobs, allowed=(0,)):
        processes = [self.launch(job) for job in jobs]
        while any(process.poll() is None for process in processes):
            self.monitor(jobs, processes)
            first = next((process for process in processes if process.poll() is None), None)
            if first is not None:
                try:
                    first.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
        for job, process in zip(jobs, processes):
            if process.returncode not in allowed:
                raise subprocess.CalledProcessError(process.returncode, process.args, stderr=str(job.log))
        return [process.returncode for process in processes]

    def wait_probes(self):
        for process in self.probe_processes.values():
            if process is not None:
                while process.poll() is None:
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pass


def calibration_args(scales, windows=WINDOWS):
    return ('--scales', ','.join(map(str, scales)), '--windows',
            ','.join(f'{low}:{high}' for low, high in windows), '--mse-weight', str(MSE_WEIGHT), '--refine-top', '3')


def calibrate_and_review(supervisor, checkpoint, out, plan, scales=SCALES, windows=WINDOWS):
    kwargs = calibration_args(scales, windows)
    if not (out / 'selected.json').exists():
        supervisor.run(Job(f'{out} 验证 CFG 校准', ('scripts/calibrate.py', '--checkpoint', checkpoint,
                                                  '--outdir', out, *kwargs), out / 'calibration.log'))
    select = out / 'postprocessing'
    args = ('scripts/select_validation_postprocess.py', '--calibration', out, '--outdir', select, '--plan', plan)
    supervisor.run(Job(f'{out} 全量验证阈值比较', args, out / 'postprocessing.log', ()), allowed=(0, 2))
    return read_json(select / 'review.json')


def ordered_scope_candidates(reviews):
    return [{**candidate, 'calibration_dir': str(out)} for out, review in reviews.items()
            for candidate in review['candidates']]


def validation_job(checkpoint, out, guidance, low, high, gpus=ALL_GPUS):
    args = ('scripts/evaluate.py', '--checkpoint', checkpoint, '--outdir', out, '--split', 'validation',
            '--cap', '16', '--clamp', '--de', '--cfg-min', low, '--cfg-max', high)
    if guidance is not None:
        args += ('--guidance', guidance)
    return Job(f'{out} cap=16 验证生成', args, out / 'console.log', gpus)


def velocity_job(checkpoint, out, gpu):
    return Job(f'{out} 速度探针', ('scripts/probe_velocity.py', '--checkpoint', checkpoint, '--outdir', out),
               out / 'console.log', (gpu,))


def run_phase0(supervisor):
    summary_path = PHASE0 / 'summary.json'
    if summary_path.exists():
        result = read_json(summary_path)
        if result['choice'] is None:
            raise RuntimeError('Phase 0 found no complete validation candidate satisfying the registered checks')
        return result
    ddp = read_json(PHASE0 / 'gradient_ddp4.json')
    smoke = read_json(PHASE0 / 'ddp_smoke/checkpoint_state.json')
    if not ddp['passed'] or smoke['step'] != 10:
        raise RuntimeError('Phase 0 requires the ten-step DDP smoke test and the matched-gradient check')
    readme_event(f'Phase 0 完成回归与 DDP 核验。113 项测试通过。4 卡冒烟完成 10 步，'
                 f'同批梯度 L2 相对偏差 {ddp["relative_grad_l2_error"]:.3%}。'
                 f'首次回归发现后处理模块导入时修改 CUDA 环境，已移除该副作用并全量复测。')
    probes = PHASE0 / 'velocity_probe'
    e45 = PRIOR / 'best_generation.pt'
    e30 = PRIOR / 'validation_epoch_030/model.pt'
    e15 = PRIOR / 'validation_epoch_015/model.pt'
    supervisor.run(velocity_job(e30, probes / 'e30', 0), velocity_job(e45, probes / 'e45', 1),
                   velocity_job(e15, probes / 'e15', 2))
    plan = PHASE0 / 'postprocess_plan.json'
    if not plan.exists():
        write_json_atomic(plan, postprocess_plan())
    initial = PHASE0 / 'repro3_e45_calibration'
    reviews = {initial: calibrate_and_review(supervisor, e45, initial, plan)}
    first = reviews[initial]['selected']
    if first is not None and first['guidance'] == min(SCALES):
        extra = PHASE0 / 'repro3_e45_lower_guidance'
        reviews[extra] = calibrate_and_review(supervisor, e45, extra, plan, (.50, .55))
    if first is None:
        extra = PHASE0 / 'repro3_e45_gate_followup'
        reviews[extra] = calibrate_and_review(supervisor, e45, extra, plan, (1.125, 1.375))
    chosen = choose_validation_candidate(ordered_scope_candidates(reviews), DEFAULT_THRESHOLDS, mse_weight=MSE_WEIGHT)
    comparisons = []
    records = read_json(initial / 'all_results.json')
    for record in (r for r in records if r['tag'].startswith('refine_')):
        w, low, high = (record[key] for key in ('guidance', 'cfg_min', 'cfg_max'))
        out = PHASE0 / 'repro3_e30_comparison' / record['tag']
        if not (out / 'metrics/summary.json').exists():
            supervisor.run(validation_job(e30, out, w, low, high))
        metrics = read_json(out / 'metrics/summary.json')
        match = next(r for r in records if r['tag'].startswith('screen_')
                     and (r['guidance'], r['cfg_min'], r['cfg_max']) == (w, low, high))
        screen = initial / match['tag']
        if file_sha256(out / 'indices.npy') != file_sha256(screen / 'indices.npy'):
            raise ValueError('E30 and E45 validation indices differ')
        comparisons.append({'guidance': w, 'cfg_min': low, 'cfg_max': high, 'cap': 16,
                            'e30_dir': str(out), 'e45_dir': str(screen),
                            'e30_metrics': metrics, 'e45_metrics': match['metrics'],
                            'e30_selection_score': selection_score(metrics, MSE_WEIGHT),
                            'e45_selection_score': selection_score(match['metrics'], MSE_WEIGHT)})
    result = {'test_access': False, 'choice': chosen, 'e30_comparison': comparisons,
              'reviews': {str(out): review for out, review in reviews.items()},
              'velocity_probes': {epoch: str(probes / epoch / 'summary.json') for epoch in ('e15', 'e30', 'e45')},
              'gradient_verification': ddp}
    write_json_atomic(summary_path, result)
    if chosen is None:
        raise RuntimeError('Phase 0 found no complete validation candidate satisfying the registered checks')
    readme_event(f'Phase 0 选定周期验证 CFG，w={chosen["guidance"]}，时间窗 '
                 f'[{chosen["cfg_min"]}, {chosen["cfg_max"]}]。全量验证 S={chosen["selection_score"]:.5f}，'
                 f'PDCorr={chosen["metrics"]["PDCorr"]:.5f}，MSE={chosen["metrics"]["MSE"]:.5f}，'
                 f'近零阈值 {chosen["zero_floor"]}。E30、E45 及探针结果见 `{summary_path}`。')
    return result


def review_phase0_upper_boundary(supervisor, result):
    original = result['choice']
    if original['guidance'] != max(SCALES) or result.get('upper_boundary_review'):
        return result
    out = PHASE0 / 'repro3_e45_upper_guidance'
    plan = PHASE0 / 'postprocess_plan.json'
    best_window = (original['cfg_min'], original['cfg_max'])
    alternatives = [case for review in result['reviews'].values() for case in review['candidates']
                    if (case['cfg_min'], case['cfg_max']) != best_window]
    alternative = max(alternatives, key=lambda case: case['selection_score'])
    windows = (best_window, (alternative['cfg_min'], alternative['cfg_max']))
    readme_event(f'Phase 0 初始全量验证优选 w={original["guidance"]} 触及上界。依据异常预案，'
                 f'在所选窗口和得分最高的另一窗口 {windows} 补测 w=1.375、1.50。')
    review = calibrate_and_review(supervisor, PRIOR / 'best_generation.pt', out, plan, (1.375, 1.50), windows)
    result['reviews'][str(out)] = review
    candidates = ordered_scope_candidates(result['reviews'])
    winner = choose_validation_candidate(candidates, DEFAULT_THRESHOLDS, mse_weight=MSE_WEIGHT)
    if winner['calibration_dir'] == str(out):
        w, low, high = (winner[k] for k in ('guidance', 'cfg_min', 'cfg_max'))
        records = read_json(out / 'all_results.json')
        screen = next(row for row in records if row['tag'].startswith('screen_')
                      and (row['guidance'], row['cfg_min'], row['cfg_max']) == (w, low, high))
        e30 = PHASE0 / 'repro3_e30_comparison/upper_boundary_winner'
        if not (e30 / 'metrics/summary.json').exists():
            supervisor.run(validation_job(PRIOR / 'validation_epoch_030/model.pt', e30, w, low, high))
        e45 = out / screen['tag']
        if file_sha256(e30 / 'indices.npy') != file_sha256(e45 / 'indices.npy'):
            raise ValueError('Extended E30 and E45 validation indices differ')
        metrics = read_json(e30 / 'metrics/summary.json')
        result['e30_comparison'].append({'guidance': w, 'cfg_min': low, 'cfg_max': high, 'cap': 16,
            'e30_dir': str(e30), 'e45_dir': str(e45), 'e30_metrics': metrics, 'e45_metrics': screen['metrics'],
            'e30_selection_score': selection_score(metrics, MSE_WEIGHT),
            'e45_selection_score': selection_score(screen['metrics'], MSE_WEIGHT)})
    result['choice'] = winner
    result['upper_boundary_review'] = {'initial_choice': original, 'scales': [1.375, 1.50],
                                       'windows': [list(w) for w in windows], 'extended_dir': str(out)}
    write_json_atomic(PHASE0 / 'summary.json', result)
    readme_event(f'Phase 0 上界补测结束，周期 CFG 锁定为 w={winner["guidance"]}，'
                 f'时间窗 [{winner["cfg_min"]}, {winner["cfg_max"]}]，完整验证 S {winner["selection_score"]:.5f}。')
    return result


def register_pilot(tag, model, sampling, *, epochs=10, warmup=2, noise=.75):
    config = yaml.safe_load(Path('configs/repro3_main.yaml').read_text())
    config['run_dir'] = str(PILOTS / tag)
    config['model'] = dict(model)
    config['sampling'] = dict(sampling)
    config['training'].update(epochs=epochs, warmup_epochs=warmup, min_epochs=epochs,
                              early_stop_patience_epochs=None, val_generation_cap=16,
                              val_unguided_epochs=epochs, pilot_eval_raw_final=True, noise_std=noise,
                              max_ema_endpoint_relative_gap=.04)
    path = Path('configs/repro4_pilots') / f'{tag}.yaml'
    update_config(path, config)
    return path


def training_job(config_path, gpus, *options):
    config = yaml.safe_load(config_path.read_text())
    run = Path(config['run_dir'])
    return Job(f'{run} 全量数据训练', ('scripts/train.py', '--config', config_path, *options), run / 'training.log', gpus, run)


def model_checkpoint(run, epoch):
    return run / f'validation_epoch_{epoch:03d}' / 'model.pt'


def pilot_report(path, data):
    config = yaml.safe_load(path.read_text())
    run = Path(config['run_dir'])
    finished = read_json(run / 'finished.json')
    epoch = config['training']['epochs']
    if finished['step'] != epoch * 1118 or finished['epoch'] != epoch or finished['stopped_early']:
        raise ValueError(f'Pilot {run} did not finish its registered epoch budget')
    events = run_events(run)
    initial = next(event for event in events if event['event'] == 'start')
    if initial['training_cells'] != 572545 or initial['steps_per_epoch'] != 1118:
        raise ValueError(f'Pilot {run} did not use the full training split')
    endpoints = next(event for event in events if event['event'] == 'validation' and event['step'] == finished['step'])
    guided = model_checkpoint(run, epoch).parent
    raw = run / f'diagnostic_raw_epoch_{epoch:03d}'
    unguided = run / f'validation_unguided_epoch_{epoch:03d}'
    probe = read_json(run / 'velocity_probe/summary.json')
    report = {'run': str(run), 'epochs': epoch, 'steps': finished['step'], 'training_cells': 572545,
              'config': config, 'config_sha256': file_sha256(path), 'training_execution': read_json(run / 'training_execution.json'),
              'checkpoint': str(guided / 'model.pt'), 'checkpoint_sha256': file_sha256(guided / 'model.pt'),
              'guided': case_metrics(data, guided, run / 'review/ema_guided', MSE_WEIGHT),
              'unguided': case_metrics(data, unguided, run / 'review/ema_unguided', MSE_WEIGHT),
              'raw_diagnostic_only': case_metrics(data, raw, run / 'review/raw_diagnostic', MSE_WEIGHT),
              'batch': batch_rms(data, guided), 'endpoints': endpoints, 'velocity_probe': probe,
              'periodic_validation': [event for event in events if event['event'] == 'validation_generation'],
              'ema_health_stops': [event for event in events if event['event'] == 'ema_health_stop'],
              'test_access': False}
    write_json_atomic(run / 'summary.json', report)
    return report


def run_pilot_batch(supervisor, data, pairs):
    jobs = [training_job(path, gpus) for path, gpus in pairs]
    pending = [job for job in jobs if not (job.training_dir / 'finished.json').exists()]
    if pending:
        supervisor.run(*pending)
    probes = []
    for (path, gpus), job in zip(pairs, jobs):
        run = job.training_dir
        if not (run / 'finished.json').exists():
            raise RuntimeError(f'{run} stopped with its checkpoint preserved; inspect {job.log}')
        epoch = yaml.safe_load(path.read_text())['training']['epochs']
        if not (run / 'velocity_probe/summary.json').exists():
            probes.append(velocity_job(model_checkpoint(run, epoch), run / 'velocity_probe', gpus[0]))
    if probes:
        supervisor.run(*probes)
    results = {}
    for (path, _), job in zip(pairs, jobs):
        summary_path = job.training_dir / 'summary.json'
        first_review = not summary_path.exists()
        report = pilot_report(path, data)
        results[path.stem] = report
        if first_review:
            guided = report['guided']['floors']['clamp0']
            unguided = report['unguided']['floors']['clamp0']['metrics']
            probe = report['velocity_probe']
            pdcorr, pds, mse, deprec = [guided['metrics'][name] for name in ('PDCorr', 'PDS_L2', 'MSE', 'DEPrec')]
            readme_event(f'`{job.training_dir}` 完成 E{report["epochs"]}、{report["steps"]:,} 步。'
                         f'EMA 有引导 S {guided["score"]:.5f}，PDCorr {pdcorr:.5f}，PDS_L2 {pds:.5f}，'
                         f'MSE {mse:.5f}，DEPrec {deprec:.5f}。无引导 PDCorr {unguided["PDCorr"]:.5f}，'
                         f'PDS_L2 {unguided["PDS_L2"]:.5f}。EMA endpoint MSE '
                         f'{report["endpoints"]["ema_0.9996"]["endpoint_mse"]:.6f}，批次 RMS 比值 '
                         f'{report["batch"]["inflation_ratio"]:.3f}，速度探针首次无收益 t={probe["val_t_cutoff"]}。'
                         f'两种阈值、原始权重与九项指标见 `{summary_path}`。')
    return results


def guided(report):
    return report['guided']['floors']['clamp0']


def baseline_rank(report):
    unguided = report['unguided']['floors']['clamp0']['metrics']
    return (guided(report)['score'], unguided['PDCorr'], unguided['PDS_L2'],
            -report['endpoints']['ema_0.9996']['endpoint_mse'], -abs(report['batch']['inflation_ratio'] - 1.))


def advances_baseline(candidate, reference):
    proposed = baseline_rank(candidate)
    previous = baseline_rank(reference)
    return proposed[0] > previous[0] and all(a >= b for a, b in zip(proposed[1:4], previous[1:4]))


def run_round1(supervisor, data, sampling):
    out = PILOTS / 'round1_summary.json'
    if out.exists():
        return read_json(out)
    model = yaml.safe_load(Path('configs/repro3_main.yaml').read_text())['model']
    configs = {'p1_ctrl': model,
               'p1_cell_nobatch': dict(model, cell_init_std=.02, batch_condition=False),
               'p1_cell_batch_noguide': dict(model, cell_init_std=.02, batch_condition=True, drop_batch_with_cell=False),
               'p1_cell_nobatch_nopos': dict(model, cell_init_std=.02, batch_condition=False, pos_encoding=False)}
    paths = {tag: register_pilot(tag, cfg, sampling) for tag, cfg in configs.items()}
    readme_event('Pilot Round 1 按登记的四组启动。全部使用 572,545 个训练细胞、10 轮、11,180 步，'
                 '两卡训练各累积 16 次。周期验证与末轮对照使用同一批 cap=16 验证细胞。')
    results = run_pilot_batch(supervisor, data, [(paths['p1_ctrl'], (0, 1)), (paths['p1_cell_nobatch'], (2, 3))])
    results.update(run_pilot_batch(supervisor, data, [(paths['p1_cell_batch_noguide'], (0, 1)),
                                                     (paths['p1_cell_nobatch_nopos'], (2, 3))]))
    batch = results['p1_cell_batch_noguide']
    no_batch = results['p1_cell_nobatch']
    no_pos = results['p1_cell_nobatch_nopos']
    batch_effect = paired_bootstrap(guided(no_batch)['per_perturbation_csv'], guided(batch)['per_perturbation_csv'])
    position_effect = paired_bootstrap(guided(no_batch)['per_perturbation_csv'], guided(no_pos)['per_perturbation_csv'])
    if batch_effect['score_delta']['ci95'][0] > 0 and advances_baseline(no_pos, no_batch):
        extra = 'p1_cell_batch_noguide_nopos'
        config = register_pilot(extra, dict(configs['p1_cell_batch_noguide'], pos_encoding=False), sampling)
        readme_event('保留不参与 CFG 的批次条件与移除位置编码分别出现收益，增加 10 轮交互核验。')
        results.update(run_pilot_batch(supervisor, data, [(config, ALL_GPUS)]))

    control = results['p1_ctrl']
    eligible = [tag for tag, report in results.items() if advances_baseline(report, control)]
    winner = max(eligible, key=lambda tag: baseline_rank(results[tag])) if eligible else 'p1_ctrl'
    result = {'base1': winner, 'eligible': eligible, 'cases': {tag: str(PILOTS / tag / 'summary.json') for tag in results},
              'configuration': results[winner]['config']['model'],
              'batch_comparison': batch_effect, 'position_comparison': position_effect,
              'rule': 'Prefer higher guided clamp0 score with nondecreasing unguided PDCorr and PDS_L2 and no endpoint MSE regression. '
                      'Order by score and unguided metrics, with batch RMS proximity as the final tie-break. Retain control if none qualifies.',
              'test_access': False}
    write_json_atomic(out, result)
    readme_event(f'Round 1 选定 Base1 `{winner}`，有引导、无引导、去噪和批次 RMS 的选择依据见 `{out}`。')
    return result


def guidance_sweep(supervisor, data, report, scales, low, high, *, gpus=ALL_GPUS):
    run = Path(report['run'])
    records = []
    for w in scales:
        if w == report['config']['sampling']['guidance']:
            records.append({'guidance': w, **guided(report)})
            continue
        tag = f'{w:g}'.replace('.', 'p')
        out = run / 'guidance_sweep' / f'w_{tag}'
        if not (out / 'metrics/summary.json').exists():
            supervisor.run(validation_job(report['checkpoint'], out, w, low, high, gpus))
        case = case_metrics(data, out, out / 'floor_review', MSE_WEIGHT)
        records.append({'guidance': w, **case['floors']['clamp0']})
    return records


def unseen_pdcorr(case):
    values = [row['PDCorr'] for row in case['unseen'].values() if row['PDCorr'] is not None]
    if not values:
        raise ValueError('Expected previously unseen perturbations in validation')
    return sum(values) / len(values)


def improved_jit(candidate, reference):
    probe = candidate['velocity_probe']
    baseline = reference['velocity_probe']
    return (guided(candidate)['score'] >= guided(reference)['score'] - .008
            and (overfit_distance(probe) < .9 * overfit_distance(baseline) or age_cutoff(probe) > age_cutoff(baseline)))


def run_round2(supervisor, data, sampling, round1):
    out = PILOTS / 'round2_summary.json'
    if out.exists():
        return read_json(out)
    base = read_json(round1['cases'][round1['base1']])
    base_model = base['config']['model']
    changes = {'p2_jit_drop': {'attn_drop': .1, 'proj_drop': .1, 'dropout_layer_policy': 'jit_middle'},
               'p2_sep_pert': {'pert_projection': 'separate'},
               'p2_cfg_drop': {'condition_dropout_mode': 'balanced_3mode', 'drop_cell_only_prob': .08,
                               'drop_pert_only_prob': .08, 'drop_both_prob': .06}}
    paths = {tag: register_pilot(tag, dict(base_model, **change), sampling) for tag, change in changes.items()}
    results = run_pilot_batch(supervisor, data, [(paths['p2_jit_drop'], (0, 1)), (paths['p2_sep_pert'], (2, 3))])
    results.update(run_pilot_batch(supervisor, data, [(paths['p2_cfg_drop'], ALL_GPUS)]))
    baseline = guided(base)
    jit, sep, cfg = [results[tag] for tag in ('p2_jit_drop', 'p2_sep_pert', 'p2_cfg_drop')]
    base_probe = base['velocity_probe']
    jit_adopted = improved_jit(jit, base)
    jit_details = {'adopted': jit_adopted, 'score_delta': guided(jit)['score'] - baseline['score'],
                   'overfit_distance': overfit_distance(jit['velocity_probe']),
                   'cutoff': age_cutoff(jit['velocity_probe'])}
    if abs(jit_details['score_delta']) <= .002:
        readme_event('JiT Dropout 的 E10 综合分在 0.002 内持平，登记两组同预算 E20 核验。'
                     '两组各从头使用 20 轮 cosine，以保持可比的学习率轨迹。')
        extended = run_pilot_batch(supervisor, data, [
            (register_pilot('p2_jit_e20_control', base_model, sampling, epochs=20), (0, 1)),
            (register_pilot('p2_jit_e20', dict(base_model, **changes['p2_jit_drop']), sampling, epochs=20), (2, 3)),
        ])
        results.update(extended)
        jit_adopted = improved_jit(extended['p2_jit_e20'], extended['p2_jit_e20_control'])
        jit_details['e20_adopted'] = jit_adopted
    sep_guided = guided(sep)
    sep_details = {'score_delta': sep_guided['score'] - baseline['score'],
                   'pdcorr_delta': sep_guided['metrics']['PDCorr'] - baseline['metrics']['PDCorr'],
                   'unseen_pdcorr_delta': unseen_pdcorr(sep_guided) - unseen_pdcorr(baseline),
                   'overfit_distance': overfit_distance(sep['velocity_probe'])}
    sep_adopted = (sep_details['score_delta'] >= 0 and sep_details['pdcorr_delta'] >= 0
                   and sep_details['unseen_pdcorr_delta'] > 0 and sep_details['overfit_distance'] <= overfit_distance(base_probe))
    sep_details.update(adopted=sep_adopted, projection_effective_rank=perturbation_effective_rank(Path(sep['checkpoint'])))

    w0, low, high = (sampling[k] for k in ('guidance', 'cfg_min', 'cfg_max'))
    scales = (round(w0 - .15, 8), w0, round(w0 + .15, 8))
    base_sweep = guidance_sweep(supervisor, data, base, scales, low, high)
    cfg_sweep = guidance_sweep(supervisor, data, cfg, scales, low, high)
    base_best = max(base_sweep, key=lambda row: row['score'])
    cfg_best = max(cfg_sweep, key=lambda row: row['score'])
    comparable = ('PDCorr', 'PDS_L1', 'PDS_L2', 'PDS_cos')
    lower_error = (cfg_best['metrics']['MSE'] <= .9 * base_best['metrics']['MSE']
                   and all(cfg_best['metrics'][key] >= base_best['metrics'][key] for key in comparable))
    cfg_adopted = (vu_error(cfg['velocity_probe']) < vu_error(base_probe)
                   and (cfg_best['score'] >= base_best['score'] + .005 or lower_error))
    cfg_details = {'adopted': cfg_adopted, 'base_sweep': base_sweep, 'candidate_sweep': cfg_sweep,
                   'vu_mse_val_mean': vu_error(cfg['velocity_probe']), 'base_vu_mse_val_mean': vu_error(base_probe),
                   'score_delta': cfg_best['score'] - base_best['score']}
    adopted = [tag for tag, adopt in [('p2_jit_drop', jit_adopted), ('p2_sep_pert', sep_adopted),
                                     ('p2_cfg_drop', cfg_adopted)] if adopt]
    model = dict(base_model)
    for tag in adopted:
        model.update(changes[tag])
    result = {'base1': round1['base1'], 'adopted': adopted, 'configuration': model, 'changes': changes,
              'cases': {tag: str(PILOTS / tag / 'summary.json') for tag in results},
              'jit': jit_details, 'separate_projection': sep_details, 'balanced_cfg': cfg_details,
              'test_access': False}
    write_json_atomic(out, result)
    readme_event(f'Round 2 完成。Base2 采纳项为 {"、".join(adopted) or "沿用 Base1"}。'
                 f'速度探针、未见扰动、投影有效秩和 3 档同尺度比较见 `{out}`。')
    return result


def run_round3(supervisor, data, sampling, round1, round2, phase0):
    out = PILOTS / 'round3_summary.json'
    if out.exists():
        return read_json(out)
    e15 = read_json(PRIOR / 'validation_epoch_015/metrics/summary.json')
    historical = {'metrics': e15, 'score': selection_score(e15, MSE_WEIGHT),
                  'probe': read_json(phase0['velocity_probes']['e15']), 'ema_endpoint_mse': .09823}
    model = round2['configuration']
    trials = [('p3_combo_s075', model, .75), ('p3_combo_s055', model, .55)]

    def review(trials):
        devices = [ALL_GPUS] if len(trials) == 1 else [(0, 1), (2, 3)]
        paths = [register_pilot(tag, cfg, sampling, epochs=15, warmup=3, noise=noise) for tag, cfg, noise in trials]
        reports = run_pilot_batch(supervisor, data, list(zip(paths, devices, strict=True)))
        rows = {}
        for tag, report in reports.items():
            low, high = (sampling[k] for k in ('cfg_min', 'cfg_max'))
            sweep = guidance_sweep(supervisor, data, report, (.65, .85, 1.05, 1.25), low, high)
            best = max(sweep, key=lambda row: row['score'])
            metrics = best['metrics']
            probe = report['velocity_probe']
            pdcorr_delta = metrics['PDCorr'] - e15['PDCorr']
            ratio = overfit_distance(probe)
            checks = {'score_improved': best['score'] > historical['score'],
                      'pdcorr_gain_or_mse_reduction': pdcorr_delta >= .010 or
                      (metrics['MSE'] <= .9 * e15['MSE'] and pdcorr_delta >= 0),
                      'overfit_reduced': ratio < .9 * overfit_distance(historical['probe']),
                      'cutoff': age_cutoff(probe) >= .75,
                      'training_health': not report['ema_health_stops']}
            rows[tag] = {'summary': str(PILOTS / tag / 'summary.json'), 'sweep': sweep, 'best': best,
                         'pdcorr_delta': pdcorr_delta, 'overfit_distance': ratio, 'cutoff': age_cutoff(probe),
                         'ema_endpoint_mse': report['endpoints']['ema_0.9996']['endpoint_mse'],
                         'checks': checks, 'pass': all(checks.values()), 'model': report['config']['model'],
                         'noise_std': report['config']['training']['noise_std']}
            failed = ', '.join(key for key, passed in checks.items() if not passed)
            readme_event(f'Round 3 `{tag}` 的 E15 最优 S {best["score"]:.5f}，w={best["guidance"]}，'
                         f'PDCorr 相对第三轮 E15 变化 {pdcorr_delta:+.5f}，MSE {metrics["MSE"]:.5f}，'
                         f'端点 MSE {rows[tag]["ema_endpoint_mse"]:.6f}，速度截止 {age_cutoff(probe):.2f}。'
                         f'门控结果为 {failed or "全部通过"}。')
        return rows

    cases = review(trials)
    if not any(row['pass'] for row in cases.values()):
        winner = max(cases, key=lambda tag: cases[tag]['best']['score'])
        noise = cases[winner]['noise_std']
        adopted = round2['adopted']
        base1 = round1['configuration']
        repairs = []
        if 'p2_sep_pert' in adopted:
            repairs.append(('p3_shared_projection', dict(model, pert_projection='shared'), noise))
        if 'p2_cfg_drop' in adopted:
            repairs.append(('p3_lower_joint_dropout', dict(model, drop_both_prob=.04), noise))
        if not repairs and 'p2_jit_drop' in adopted:
            repairs.append(('p3_jit_drop_005', dict(model, attn_drop=.05, proj_drop=.05), noise))
        if not repairs and base1.get('pos_encoding', True):
            repairs.append(('p3_position_0125', dict(model, position_scale=.125), noise))
        if repairs:
            readme_event('Round 3 两个组合均未过门控，按 E15 最优噪声尺度登记定向消融。'
                         '消融组沿用相同训练预算和验证搜索。')
            cases.update(review(repairs))
    eligible = [tag for tag, row in cases.items() if row['pass']]
    winner = max(eligible, key=lambda tag: cases[tag]['best']['score']) if eligible else None
    result = {'selected': winner, 'historical_e15': historical, 'cases': cases, 'test_access': False}
    write_json_atomic(out, result)
    if winner:
        readme_event(f'Round 3 选定 `{winner}` 进入正式全量重训，九项验证指标和准入检查见 `{out}`。')
    return result


def check_early_stop(supervisor, data, sampling):
    result_path = MAIN / 'early_stop_review.json'
    if result_path.exists():
        return read_json(result_path)['continue_schedule']
    events = run_events(MAIN)
    unguided = [event for event in events if event['event'] == 'validation_unguided'][-3:]
    endpoint = [event['ema_0.9996']['endpoint_mse'] for event in events
                if event['event'] == 'validation' and event['step'] % (5 * 1118) == 0][-3:]
    endpoint_advances = len(endpoint) == 3 and all(b < a for a, b in zip(endpoint, endpoint[1:]))
    unguided_advances = (len(unguided) >= 2 and all(
        b['PDCorr'] > a['PDCorr'] and b['PDS_L2'] >= a['PDS_L2'] for a, b in zip(unguided, unguided[1:])))
    rows = []
    if endpoint_advances and unguided_advances:
        generation = [event['epoch'] for event in events if event['event'] == 'validation_generation'][-3:]
        w = round(sampling['guidance'] - .15, 8)
        for epoch in generation:
            out = MAIN / 'early_stop_review' / f'epoch_{epoch:03d}'
            if not (out / 'metrics/summary.json').exists():
                supervisor.run(validation_job(model_checkpoint(MAIN, epoch), out, w, sampling['cfg_min'], sampling['cfg_max']))
            case = case_metrics(data, out, out / 'floor_review', MSE_WEIGHT)['floors']['clamp0']
            rows.append({'epoch': epoch, 'guidance': w, **case})
    continue_schedule = len(rows) == 3 and all(b['score'] > a['score'] for a, b in zip(rows, rows[1:]))
    write_json_atomic(result_path, {'recent_ema_endpoint_mse': endpoint, 'recent_unguided': unguided,
                                    'lower_guidance': rows, 'continue_schedule': continue_schedule,
                                    'test_access': False})
    return continue_schedule


def choose_checkpoint(supervisor, data, sampling):
    path = MAIN / 'checkpoint_review.json'
    if path.exists():
        return Path(read_json(path)['checkpoint'])
    generation = MAIN / 'best_generation.pt'
    denoise = MAIN / 'best_denoise.pt'
    options = {'generation': generation, 'denoising': denoise}
    steps = {}
    for name, checkpoint in options.items():
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        if state.get('weights') != 'ema_0.9996':
            raise ValueError(f'Expected default EMA weights for {checkpoint}')
        steps[name] = state['step']
        del state
    comparisons = {}
    selected = 'generation'
    if abs(steps['generation'] - steps['denoising']) >= 10 * 1118:
        w, low, high = (sampling[key] for key in ('guidance', 'cfg_min', 'cfg_max'))
        for name, checkpoint in options.items():
            out = MAIN / 'checkpoint_review' / name
            if not (out / 'metrics/summary.json').exists():
                supervisor.run(validation_job(checkpoint, out, w, low, high))
            probe_dir = out / 'velocity_probe'
            if not (probe_dir / 'summary.json').exists():
                supervisor.run(velocity_job(checkpoint, probe_dir, 0))
            probe = read_json(probe_dir / 'summary.json')
            case = case_metrics(data, out, out / 'floor_review', MSE_WEIGHT)['floors']['clamp0']
            comparisons[name] = {'metrics': case['metrics'], 'score': case['score'],
                                 'per_perturbation_csv': case['per_perturbation_csv'],
                                 'velocity_probe': probe,
                                 'overfit_flag': overfit_distance(probe) > 4. or age_cutoff(probe) < .70}
        selected = max(comparisons, key=lambda name: (not comparisons[name]['overfit_flag'], comparisons[name]['score']))
    paired = (paired_bootstrap(comparisons['generation']['per_perturbation_csv'],
                               comparisons['denoising']['per_perturbation_csv']) if comparisons else None)
    proof = {'checkpoint': str(options[selected]), 'checkpoint_sha256': file_sha256(options[selected]),
             'steps': steps, 'selected': selected, 'validation_comparison': comparisons,
             'paired_generation_to_denoising': paired, 'test_access': False}
    write_json_atomic(path, proof)
    readme_event(f'主训练校准选用 `{options[selected]}`，生成最优 step {steps["generation"]:,}，'
                 f'去噪最优 step {steps["denoising"]:,}。轮次复核见 `{path}`。')
    return options[selected]


def final_calibration(supervisor, checkpoint, plan):
    root = MAIN / 'calibration'
    reviews = {root: calibrate_and_review(supervisor, checkpoint, root, plan)}
    candidates = ordered_scope_candidates(reviews)
    first = choose_validation_candidate(candidates, DEFAULT_THRESHOLDS, mse_weight=MSE_WEIGHT)
    boundary = first or max(candidates, key=lambda case: case['selection_score'])
    if boundary['guidance'] in (min(SCALES), max(SCALES)):
        extra_scales = (.50, .575) if boundary['guidance'] == min(SCALES) else (1.375, 1.50)
        out = root / 'boundary_extension'
        readme_event(f'主实验完整验证优选 w={boundary["guidance"]} 触及网格边界，补测 {extra_scales}。')
        reviews[out] = calibrate_and_review(supervisor, checkpoint, out, plan, extra_scales)
        candidates = ordered_scope_candidates(reviews)
    selected = choose_validation_candidate(candidates, DEFAULT_THRESHOLDS, mse_weight=MSE_WEIGHT)
    if selected is None:
        closest = choose_validation_candidate(candidates, DEFAULT_THRESHOLDS, report_shortfall=True, mse_weight=MSE_WEIGHT)
        repro3_val = read_json(PRIOR / 'validation_full_postprocess/metrics/summary.json')
        result = {'stage': 'validation_needs_review', 'time': timestamp(), 'test_access': False,
                  'selected': None, 'closest_validation_candidate': closest,
                  'repro3_full_validation': repro3_val, 'paper_table3_test_reference': PAPER,
                  'candidate_minus_repro3_validation': {k: closest['metrics'][k] - v for k, v in repro3_val.items()},
                  'review_paths': [str(out / 'postprocessing/review.json') for out in reviews]}
        write_json_atomic(MAIN / 'pipeline_state.json', result)
        failures = '、'.join(key for key, passed in closest['checks'].items() if not passed)
        readme_event(f'主实验完整验证停在 `validation_needs_review`。最接近的候选尚未通过 {failures}。'
                     f'完整九指标和失败项目见 `{MAIN}/pipeline_state.json`。')
        return None
    folder = Path(selected['calibration_dir'])
    choice_path = folder / 'postprocessing/selected.json'
    freeze = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
              'selection': str(choice_path), 'selection_sha256': file_sha256(choice_path),
              'config_sha256': file_sha256(plan), 'mse_weight': MSE_WEIGHT, 'test_access': False,
              'candidate': selected}
    write_json_atomic(root / 'freeze.json', freeze)
    readme_event(f'正式全量验证通过全部质量门槛，冻结 EMA step {read_json(choice_path)["step"]:,}，'
                 f'w={selected["guidance"]}，时间窗 [{selected["cfg_min"]}, {selected["cfg_max"]}]，'
                 f'阈值 {selected["zero_floor"]}，验证 S {selected["selection_score"]:.5f}。')
    return folder


def run_main_experiment(supervisor, data, sampling, round3, plan):
    training_config = Path('configs/repro4_main.yaml')
    config = yaml.safe_load(Path('configs/repro3_main.yaml').read_text())
    candidate = round3['cases'][round3['selected']]
    config.update(run_dir=str(MAIN), model=candidate['model'], sampling=sampling)
    config['training'].update(epochs=80, warmup_epochs=5, min_epochs=35, early_stop_patience_epochs=15,
                              early_stop_min_delta=.002, val_generation_cap=32, val_unguided_epochs=10,
                              noise_std=candidate['noise_std'], max_ema_endpoint_relative_gap=.04)
    update_config(training_config, config)
    if not (MAIN / 'finished.json').exists():
        resumed = ('--finish-schedule',) if (MAIN / 'finished_early_stop.json').exists() else ()
        supervisor.run(training_job(training_config, ALL_GPUS, *resumed))
        supervisor.wait_probes()
    if not (MAIN / 'finished.json').exists():
        raise RuntimeError('Main training stopped before completion; its checkpoint and health events were preserved')
    if read_json(MAIN / 'finished.json')['stopped_early'] and check_early_stop(supervisor, data, sampling):
        readme_event('主训练早停复核显示去噪、无引导指标与更低指导尺度仍在改善。'
                     '从原断点继续完成已登记的 80 轮 cosine 预算。')
        supervisor.run(training_job(training_config, ALL_GPUS, '--finish-schedule'))
        supervisor.wait_probes()
    epochs = sorted({event['epoch'] for event in run_events(MAIN) if event['event'] == 'validation_generation'})
    for epoch in epochs:
        out = MAIN / 'velocity_probe' / f'epoch_{epoch:03d}'
        if epoch % 10 == 0 and not (out / 'summary.json').exists():
            supervisor.run(velocity_job(model_checkpoint(MAIN, epoch), out, 0))
    trajectory = {epoch: read_json(MAIN / 'velocity_probe' / f'epoch_{epoch:03d}' / 'summary.json')
                  for epoch in epochs if epoch % 10 == 0}
    write_json_atomic(MAIN / 'velocity_probe/trajectory.json', trajectory)
    proof_path = MAIN / 'training_completion_verification.json'
    proof = read_json(proof_path) if proof_path.exists() else verify_training(MAIN, config, epochs)
    checkpoint = choose_checkpoint(supervisor, data, sampling)
    freeze_path = MAIN / 'calibration/freeze.json'
    if freeze_path.exists():
        freeze = read_json(freeze_path)
        selection = Path(freeze['selection'])
        if (freeze['checkpoint_sha256'] != file_sha256(checkpoint) or freeze['config_sha256'] != file_sha256(plan)
                or freeze['selection_sha256'] != file_sha256(selection)):
            raise ValueError('Frozen main validation selection changed')
        calibration = selection.parent.parent
    else:
        calibration = final_calibration(supervisor, checkpoint, plan)
    if calibration is None:
        return None
    with (MAIN / 'events.jsonl').open('rb') as stream:
        event_log = stream.read(proof['training_event_log']['bytes'])
    if hashlib.sha256(event_log).hexdigest() != proof['training_event_log']['sha256']:
        raise ValueError('Training event history changed after verification')

    # The pipeline launches its own GPU workers, so it must run as a single process.
    # Its child CUDA_VISIBLE_DEVICES still contains every registered device.
    pipeline = ('scripts/run_experiment.py', '--config', training_config, '--gpu-workers', '4',
                '--postprocess-plan', plan, '--calibration-checkpoint', checkpoint, '--calibration-dir', calibration,
                '--calibration-mse-weight', str(MSE_WEIGHT), '--calibration-refine-top', '3')
    supervisor.run(Job('冻结验证与唯一一次正式测试', pipeline, MAIN / 'workflow_pipeline.log', distributed=False))
    return verify_and_compare_test(MAIN)


def archive_reproduction(phase0, round1, round2, round3, test):
    finished = read_json(MAIN / 'finished.json')
    freeze = read_json(MAIN / 'calibration/freeze.json')['candidate']
    selected = round3['cases'][round3['selected']]
    lines = ['# OCOO-T Replogle–Nadig 第四轮复现归档', '',
             f'全量训练完成 {finished["epoch"]:.0f} 轮、{finished["step"]:,} 步，'
             f'用 EMA 0.9996 完成 26,878 个 HepG2 细胞、380 个扰动的冻结测试。'
             '两份后处理报告共享同一批原始生成。', '',
             '## 改动与实验选择', '',
             f'- Phase 0 完成回归、四卡冒烟和同批梯度验证，单卡与四卡梯度 L2 相对偏差为 '
             f'{phase0["gradient_verification"]["relative_grad_l2_error"]:.3%}。'
             '校准与验证后处理均采用 S = PDCorr + mean(PDS) − 10 × MSE。 '
             '[Phase 0 记录](runs/repro4_phase0/summary.json)',
             f'- Round 1 选定 `{round1["base1"]}`，Round 2 采纳 '
             f'{"、".join(round2["adopted"]) or "Base1 配置"}。 '
             '[Round 1 记录](runs/repro4_pilots/round1_summary.json) · '
             '[Round 2 记录](runs/repro4_pilots/round2_summary.json)',
             f'- Round 3 选定 `{round3["selected"]}`，噪声标准差 {selected["noise_std"]}。 '
             '[Round 3 门控](runs/repro4_pilots/round3_summary.json)',
             f'- 完整验证选定 w={freeze["guidance"]}，时间窗 '
             f'[{freeze["cfg_min"]}, {freeze["cfg_max"]}]，零阈值 {freeze["zero_floor"]}。 '
             '[冻结记录](runs/repro4_main/calibration/freeze.json) · '
             '[训练产物核验](runs/repro4_main/training_completion_verification.json) · '
             '[速度体检轨迹](runs/repro4_main/velocity_probe/trajectory.json)', '',
             '## 冻结测试结果', '',
             '| 指标 | 论文表 3 | 第一轮验证 | 第二轮测试 | 第三轮测试 | 第四轮测试 | 第四轮 clamp0 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    first_review = read_json('runs/main_ddp_recovery/calibration/postprocessing/review.json')
    first = next(row['metrics'] for row in first_review['candidates']
                 if (row['guidance'], row['cfg_min'], row['cfg_max'], row['zero_floor']) == (2., 0., 1., .05))
    prior = [read_json(Path(f'runs/repro{i}_main') / 'test_full_postprocess/metrics/summary.json') for i in (2, 3)]
    for metric, value in PAPER.items():
        lines.append(f'| {metric} | {value:.5f} | {first[metric]:.5f} | {prior[0][metric]:.5f} | {prior[1][metric]:.5f} | '
                     f'{test["metrics"][metric]:.5f} | {test["baseline_clamp0_metrics"][metric]:.5f} |')
    lines.extend(['', '配对区间按相同测试扰动重采样 2,000 次。论文表 3 的区间对比将发表的聚合点估计固定。 '
                  '[选定阈值报告](runs/repro4_main/results/REPORT.md) · '
                  '[裁负对照报告](runs/repro4_main/results_clamp0/REPORT.md) · '
                  '[与第二、三轮配对比较](runs/repro4_main/results/paired_previous_rounds.csv) · '
                  '[冻结测试核验](runs/repro4_main/test_completion_verification.json)', '',
                  '完整的代码调整、实验过程与逐次训练观察见本轮日志和 [README.md](README.md)。', ''])
    temporary = Path('repro4.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace('repro4.md')
    readme_event(f'第四轮全量训练与冻结评测均已完成，实际训练 {finished["epoch"]:.0f} 轮。'
                 f'测试 PDCorr {test["metrics"]["PDCorr"]:.5f}，MSE {test["metrics"]["MSE"]:.5f}，'
                 f'DEPrec {test["metrics"]["DEPrec"]:.5f}。九项指标和配对结果见 `repro4.md`。')


def mark_readme_stage(stage, status):
    def replace(content):
        lines = content.splitlines(keepends=True)
        prefix = f'| {stage} |'
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                left = line.rstrip().removesuffix('|').rsplit('|', 1)[0]
                lines[i] = f'{left}| {status} |\n'
                break
        return ''.join(lines)

    edit_readme(replace)


def add_readme_section(title, rows):
    heading = f'## {title}\n'

    def add(content):
        if heading in content:
            return content
        section = heading + '\n' + '\n'.join(rows) + '\n\n'
        return content.replace('## 在线记录\n', section + '## 在线记录\n', 1)

    edit_readme(add)


def readme_phase0(result):
    choice = result['choice']
    rows = [f'周期验证锁定 `w={choice["guidance"]}` 和时间窗 [{choice["cfg_min"]}, {choice["cfg_max"]}]。'
            f'E45 全量验证选定阈值 {choice["zero_floor"]}，S 为 {choice["selection_score"]:.5f}。 '
            '[完整候选与速度探针](runs/repro4_phase0/summary.json)', '',
            '| w | 时间窗 | E30 S | E45 S | E30 PDCorr | E45 PDCorr | E30 MSE | E45 MSE |',
            '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in result['e30_comparison']:
        e30, e45 = case['e30_metrics'], case['e45_metrics']
        rows.append(f'| {case["guidance"]} | [{case["cfg_min"]}, {case["cfg_max"]}] | '
                    f'{case["e30_selection_score"]:.5f} | {case["e45_selection_score"]:.5f} | '
                    f'{e30["PDCorr"]:.5f} | {e45["PDCorr"]:.5f} | {e30["MSE"]:.5f} | {e45["MSE"]:.5f} |')
    add_readme_section('Phase 0 验证结果', rows)


def readme_pilots(round_number, cases):
    metrics = ('PDCorr', 'PDS_L1', 'PDS_L2', 'PDS_cos', 'MSE', 'MAE', 'DEOver', 'DEPrec', 'DirAgr')
    rows = ['各行均来自同一组 960 个验证细胞。末轮原始权重用于诊断。', '',
            '| 实验 | 模式 | 阈值 | S | ' + ' | '.join(metrics) + ' |',
            '| --- | --- | ---: | ---: | ' + ' | '.join(['---:'] * len(metrics)) + ' |']
    for tag, path in cases.items():
        case = read_json(path)
        for key, label in (('guided', 'EMA CFG'), ('unguided', 'EMA 无引导'), ('raw_diagnostic_only', 'raw CFG')):
            for record in case[key]['floors'].values():
                scores = ' | '.join(f'{record["metrics"][metric]:.5f}' for metric in metrics)
                rows.append(f'| [{tag}]({path}) | {label} | {record["floor"]:.2f} | {record["score"]:.5f} | {scores} |')
    rows.extend(['', f'[本轮选择依据](runs/repro4_pilots/round{round_number}_summary.json)'])
    add_readme_section(f'Pilot Round {round_number} 实测记录', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase0-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    with exclusive_run(PHASE0, 'repro4_workflow'):
        supervisor = Supervisor()
        state_path = PHASE0 / 'workflow_state.json'

        def state(stage, **extra):
            write_json_atomic(state_path, {'stage': stage, 'time': timestamp(), 'pid': os.getpid(), **extra})

        stage = 'phase0'
        try:
            state(stage)
            phase0 = run_phase0(supervisor)
            phase0 = review_phase0_upper_boundary(supervisor, phase0)
            mark_readme_stage('Phase 0', '[已完成](runs/repro4_phase0/summary.json)')
            readme_phase0(phase0)
            if args.phase0_only:
                state('phase0_complete')
                return
            plan = Path('configs/repro4_postprocess.json')
            if plan.exists():
                if read_json(plan) != postprocess_plan():
                    raise ValueError('Registered round-four postprocess plan differs')
            else:
                write_json_atomic(plan, postprocess_plan())
            choice = phase0['choice']
            sampling = {key: choice[key] for key in ('guidance', 'cfg_min', 'cfg_max')}
            sampling.update(steps=50, clamp=True)
            data = BenchmarkData('data/processed')
            stage = 'pilot_round1'
            state(stage)
            mark_readme_stage('Pilot Round 1', '训练与验证进行中')
            round1 = run_round1(supervisor, data, sampling)
            mark_readme_stage('Pilot Round 1', f'Base1 `{round1["base1"]}` · [结果](runs/repro4_pilots/round1_summary.json)')
            readme_pilots(1, round1['cases'])
            stage = 'pilot_round2'
            state(stage)
            mark_readme_stage('Pilot Round 2', '单因素比较进行中')
            round2 = run_round2(supervisor, data, sampling, round1)
            mark_readme_stage('Pilot Round 2', '[已锁定 Base2](runs/repro4_pilots/round2_summary.json)')
            readme_pilots(2, round2['cases'])
            stage = 'pilot_round3'
            state(stage)
            mark_readme_stage('Pilot Round 3', '组合与噪声尺度比较进行中')
            round3 = run_round3(supervisor, data, sampling, round1, round2, phase0)
            readme_pilots(3, {tag: row['summary'] for tag, row in round3['cases'].items()})
            if round3['selected'] is None:
                mark_readme_stage('Pilot Round 3', '[待门控审查](runs/repro4_pilots/round3_summary.json)')
                state('pilot_needs_review', summary=str(PILOTS / 'round3_summary.json'), test_access=False)
                readme_event('Round 3 定向消融后仍无方案达到全量重训准入条件。'
                             '各项实测、差距与准入检查见 `runs/repro4_pilots/round3_summary.json`。')
                raise SystemExit(2)
            mark_readme_stage('Pilot Round 3', f'选定 `{round3["selected"]}` · [门控](runs/repro4_pilots/round3_summary.json)')
            stage = 'main_experiment'
            state(stage)
            mark_readme_stage('主实验', '全量重训、校准与评测流程进行中')
            test = run_main_experiment(supervisor, data, sampling, round3, plan)
            if test is None:
                mark_readme_stage('主实验', '[完整验证待审查](runs/repro4_main/pipeline_state.json)')
                state('validation_needs_review', summary=str(MAIN / 'pipeline_state.json'), test_access=False)
                raise SystemExit(2)
            archive_reproduction(phase0, round1, round2, round3, test)
            mark_readme_stage('主实验', '[冻结评测完成](repro4.md)')
            state('evaluated', report='repro4.md', test_access=True)
        except Exception as error:
            state('interrupted', interrupted_stage=stage, error=str(error))
            readme_event(f'第四轮流程在 {stage} 阶段暂停，保留断点和已有评测。'
                         f'异常为 `{type(error).__name__}`，详见 `{state_path}` 与本阶段日志。')
            raise
        finally:
            flush_readme(force=True)


if __name__ == '__main__':
    main()
