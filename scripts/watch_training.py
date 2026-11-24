"""Observe a running experiment and score completed validation caches on CPU.

This process never changes training, selects a model, restarts a job, or reads
test expression. Its extra DE and context diagnostics use saved predictions.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from statistics import median
import sys
import time

from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def events(path):
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # The trainer can be appending the final row.
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--from-epoch', type=int, default=10)
    parser.add_argument('--poll-seconds', type=float, default=55)
    parser.add_argument('--publish-progress', action='store_true', help='Record training snapshots in README at 15-minute intervals')
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error('poll-seconds must be at least 10')
    run = Path(args.run_dir)
    output = run / 'observation'
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    with exclusive_run(output, 'observer'):
        (output / 'observer.pid').write_text(str(os.getpid()))
        history = output / 'generation_history.json'
        observed = ({str(r['epoch']): r for r in json.loads(history.read_text())['epochs']}
                    if history.exists() else {})
        failures = {}
        last_plot_epoch = -1
        last_status_time = time.monotonic()
        def command(arguments, label):
            with (output / f'{label}.log').open('a') as log:
                subprocess.run([sys.executable, *arguments], env=environment,
                               stdout=log, stderr=log, check=True)

        while True:
            rows = events(run / 'events.jsonl')
            validations = [r for r in rows if r.get('event') == 'validation']
            completed_epoch = max((int(r['epoch']) for r in validations if float(r['epoch']).is_integer()), default=0)
            if args.publish_progress and time.monotonic() - last_status_time >= 15 * 60:
                recent = [r for r in rows if r.get('event') == 'train'][-20:]
                checkpoint_path = run / 'checkpoint_state.json'
                if recent and checkpoint_path.exists():
                    checkpoint = json.loads(checkpoint_path.read_text())
                    current = recent[-1]
                    rate = median(r['cells_per_sec'] for r in recent)
                    probe = ''
                    if validations:
                        latest = validations[-1]
                        raw = latest['raw']['endpoint_mse']
                        ema = latest['ema_0.9996']['endpoint_mse']
                        probe = f'。固定探针 step {latest["step"]:,}，raw/EMA endpoint MSE {raw:.5f}/{ema:.5f}'
                    readme_event(f'`{run}` 训练至 E{current["epoch"]:.3f}、step {current["step"]:,}。'
                                 f'最近窗口中位吞吐 {rate:.1f} cells/s、loss {current["loss"]:.5f}、'
                                 f'lr {current["lr"]:.3g}。已落盘完整断点 step {checkpoint["step"]:,}{probe}。')
                    last_status_time = time.monotonic()
            if rows and completed_epoch > last_plot_epoch:
                command(['scripts/plot_progress.py', str(run)], 'plot_progress')
                last_plot_epoch = completed_epoch
            for generation in sorted(run.glob('validation_epoch_*')):
                try:
                    epoch = int(generation.name.rsplit('_', 1)[1])
                except ValueError:
                    continue
                if epoch < args.from_epoch or str(epoch) in observed or str(epoch) in failures:
                    continue
                required = ['model.pt', 'generation_state.json', 'metrics/summary.json']
                if not all((generation / name).exists() for name in required):
                    continue
                marker = json.loads((generation / 'generation_state.json').read_text())
                if marker['completed'] != marker['total']:
                    continue
                # The model file is atomically published only after generation
                # and the trainer's six-metric evaluation have finished.
                try:
                    de = generation / 'metrics_de_diagnostic'
                    if not (de / 'completed.json').exists():
                        command(['scripts/diagnose_cached_de.py', '--generation-dir', str(generation)],
                                f'epoch_{epoch:03d}_de')
                    context = generation / 'context_diagnostics'
                    if not (context / 'context_diagnostics.json').exists():
                        command(['scripts/diagnose_pilots.py', str(generation), '--outdir', str(context)],
                                f'epoch_{epoch:03d}_context')
                    summary = json.loads((de / 'summary.json').read_text())
                    record = {'time': timestamp(), 'epoch': epoch, 'generation_dir': str(generation),
                              'generated_cells': marker['total'], 'sampling': marker['options'],
                              'checkpoint_sha256': file_sha256(generation / 'model.pt'),
                              'metrics': summary,
                              'context': str(context / 'context_diagnostics.json'),
                              'note': 'Capped validation diagnostic, not calibrated full validation or test.'}
                    observed[str(epoch)] = record
                    write_json(output / f'epoch_{epoch:03d}.json', record)
                    readme_event(f'第 {epoch} 轮缓存的 CPU 九指标检查完成：DEOver={summary["DEOver"]:.4f}、'
                                 f'DEPrec={summary["DEPrec"]:.4f}、DirAgr={summary["DirAgr"]:.4f}；'
                                 f'复用既有预测，无新增生成。完整结果见 `{de}/summary.json`（周期验证诊断，非正式测试）。')
                except Exception as error:
                    failures[str(epoch)] = {'time': timestamp(), 'error': f'{type(error).__name__}: {error}'}
                    readme_event(f'第 {epoch} 轮附加 CPU 检查发生错误，主训练继续；详见 `{output}` 中日志，需要排查。')
            finished = (run / 'finished.json').exists()
            state = {'time': timestamp(), 'pid': os.getpid(), 'stage': 'finished' if finished else 'observing',
                     'run_dir': str(run.resolve()), 'completed_epoch': completed_epoch,
                     'observed_epochs': sorted(map(int, observed)), 'failures': failures,
                     'gpu_disabled_for_observer_and_children': True,
                     'training_mutations': False, 'script_sha256': file_sha256(__file__)}
            write_json(output / 'state.json', state)
            if observed:
                write_json(output / 'generation_history.json', {
                    'note': 'All entries are intermediate capped validation diagnostics; fixed sampling options are stored per entry.',
                    'epochs': [observed[key] for key in sorted(observed, key=int)]})
            if finished:
                break
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
