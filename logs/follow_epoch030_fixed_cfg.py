"""Wait for the existing E30 observation; run one registered fixed-CFG case."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp


def read(path):
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_registered_sources(plan):
    for path, digest in plan['source_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError('Registered source changed before E30 review: ' + path)
    if file_sha256('configs/main_selected.yaml') != plan['main_config_sha256']:
        raise ValueError('Main configuration changed')
    for path, digest in plan['prerequisite_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError('Registered prerequisite changed: ' + path)


def readiness(plan):
    fields = Path(f'/proc/{plan["trainer_pid"]}/stat').read_text().split(') ')[1].split()
    if fields[0] == 'Z' or fields[19] != plan['trainer_start_tick']:
        raise RuntimeError('Original trainer changed; review this follower before restarting')
    observer = read('runs/main_v1/observation/state.json')
    if observer.get('failures'):
        raise RuntimeError('CPU epoch observer has an unresolved failure')
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    gpu_pids = sorted({int(line.strip()) for line in output.splitlines() if line.strip()})
    checks = {'observer_epoch30_complete': 30 in observer.get('observed_epochs', []),
              'epoch30_inference_checkpoint_exists': Path(plan['candidate_checkpoint']).exists(),
              'epoch30_full_snapshot_preserved': Path(plan['full_snapshot_record']).exists(),
              'gpu_only_original_trainer': set(gpu_pids).issubset({plan['trainer_pid']})}
    return {'ready': all(checks.values()), 'checks': checks, 'gpu_pids': gpu_pids,
            'observer_epochs': observer.get('observed_epochs', [])}


def verify_actual_checkpoint(plan, out):
    import torch
    import yaml
    torch.set_num_threads(1)
    full_record = read(plan['full_snapshot_record'])
    assert full_record['step'] == plan['candidate_step'] == 33540
    full = torch.load(full_record['archive_path'], map_location='cpu', mmap=True, weights_only=False)
    checkpoint = torch.load(plan['candidate_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    assert full['step'] == checkpoint['step'] == plan['candidate_step']
    assert checkpoint['weights'] == 'ema_0.9996'
    assert full['config'] == checkpoint['config'] == yaml.safe_load(Path('configs/main_selected.yaml').read_text())
    assert len(checkpoint['model']) == 193 and len(full['ema']['0.9996']) == 189
    for name, value in full['model'].items():
        assert torch.equal(full['ema']['0.9996'].get(name, value), checkpoint['model'][name]), name
    assert file_sha256(full_record['archive_path']) == full_record['sha256']
    proof = {'time': timestamp(), 'step': checkpoint['step'], 'weights': checkpoint['weights'],
             'checkpoint_sha256': file_sha256(plan['candidate_checkpoint']),
             'full_checkpoint_sha256': full_record['sha256'], 'all193_tensors_equal_default_ema_and_buffers': True,
             'main_configuration_equal': True, 'cpu_only': True}
    atomic_json(out / 'checkpoint_verification.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = read(plan_path)
    verify_registered_sources(plan)
    if args.check_only:
        print(json.dumps(readiness(plan)), flush=True)
        return
    out = plan_path.parent
    with exclusive_run(out, 'controller'):
        if (out / 'completed.json').exists():
            return

        def state(stage, **extra):
            atomic_json(out / 'process.json', {'time': timestamp(), 'pid': os.getpid(), 'stage': stage,
                        'test_access': False, 'training_changed': False, **extra})

        def execute(command, stage, cpu):
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            if cpu:
                environment['CUDA_VISIBLE_DEVICES'] = ''
            with (out / f'{stage}.log').open('a') as stream:
                child = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                state(stage, child_pid=child.pid, command=command, cpu_only=cpu)
                code = child.wait()
            if code:
                raise RuntimeError(f'{stage} exited with{code}; all caches retained')

        try:
            deadline = time.monotonic() + 48 * 3600
            while True:
                ready = readiness(plan)
                if ready['ready']:
                    break
                state('waiting_for_registered_epoch30', readiness=ready, cpu_only=True)
                if time.monotonic() >= deadline:
                    raise TimeoutError('E30 readiness not reached within48h')
                time.sleep(30)
            verify_registered_sources(plan)
            state('verifying_mature_checkpoint', cpu_only=True)
            proof = verify_actual_checkpoint(plan, out)
            fingerprint = {'plan_sha256': file_sha256(plan_path), 'checkpoint': proof,
                           'source_sha256': plan['source_sha256']}
            execution = out / 'execution_config.json'
            if execution.exists():
                previous = read(execution)
                # Verification time is observational; the scientific input must match.
                for record in (previous, fingerprint):
                    record['checkpoint'].pop('time', None)
                if previous != fingerprint:
                    raise ValueError('E30 output directory belongs to another protocol')
            else:
                atomic_json(execution, fingerprint)
            if not (out / 'candidate/metrics/summary.json').exists():
                execute(plan['generation_command'], 'fixed_cfg_generation', cpu=False)
            execute(plan['fixed_comparison_command'], 'fixed_cfg_comparison', cpu=True)
            execute(plan['periodic_comparison_command'], 'periodic_comparison', cpu=True)
            fixed = read(out / 'paired/comparison.json')
            periodic = read('runs/main_v1/observation/epoch025_to030/comparison.json')
            completed = {'time': timestamp(), 'stage': 'completed_review_required',
                         'plan_sha256': file_sha256(plan_path), 'step': plan['candidate_step'],
                         'fixed_score_difference': fixed['score_difference'],
                         'periodic_score_difference': periodic['score_difference'],
                         'fixed_comparison_sha256': file_sha256(out / 'paired/comparison.json'),
                         'periodic_comparison_sha256': file_sha256('runs/main_v1/observation/epoch025_to030/comparison.json'),
                         'automatic_checkpoint_or_convergence_change': False, 'test_access': False}
            atomic_json(out / 'completed.json', completed)
            state('completed_review_required', result=completed, cpu_only=True)
            readme_event('第30轮ordinary与固定w2的E25→30九指标配对分析完成，等待联合复核；'
                         '未自动更改断点选择或收敛策略。见 `runs/main_v1/interim_epoch030_fixed_cfg/completed.json`。')
        except BaseException as exc:
            state('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
