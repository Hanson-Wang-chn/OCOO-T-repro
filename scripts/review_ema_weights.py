"""Matched validation diagnostics for two already-maintained EMA weights.

This does not train, recalibrate CFG, change checkpoint selection, or access
test cells. Immutable coarse references avoid repeating default-EMA generation.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.distributed import review_trainer_pids
from ocoot.runtime import exclusive_run, file_sha256, timestamp


BUFFERS = {'esm', 'gene_indices', 'missing', 'position'}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def verify_pins(plan, plan_path=None, plan_digest=None, prerequisites=True):
    if plan_path is not None and file_sha256(plan_path) != plan_digest:
        raise ValueError('Registered EMA review plan changed')
    groups = ('source_sha256', 'prerequisite_sha256') if prerequisites else ('source_sha256',)
    for group in groups:
        for path, digest in plan[group].items():
            if file_sha256(path) != digest:
                raise ValueError('Registered EMA review source changed: ' + path)


def validate_protocol(expected, reference_marker, candidate_config=None,
                      candidate_marker=None, full_checkpoint=None, checkpoint_sha=None):
    """Permit EMA decay alone; clamp differs only as a stored representation."""
    if (reference_marker['completed'] != 960 or reference_marker['total'] != 960
            or reference_marker['options'] != {**expected, 'clamp': False}):
        raise ValueError('Reference must be the exact completed raw coarse protocol')
    if candidate_config is None:
        return
    required = {
        'checkpoint': str(Path(full_checkpoint).resolve()), 'checkpoint_sha256': checkpoint_sha,
        'step': 39130, 'weights': 'ema_0.9999', 'split': 'validation', 'cap': 16,
        'guidance': expected['guidance'], 'cfg_min': expected['cfg_min'],
        'cfg_max': expected['cfg_max'], 'clamp': True, 'de': True,
        'seed': 4242, 'steps': 50, 'batch_size': 32, 'noise_std': 1.,
    }
    if any(candidate_config.get(k) != v for k, v in required.items()):
        raise ValueError('Candidate differs beyond the registered EMA decay')
    if (candidate_marker['completed'] != 960 or candidate_marker['total'] != 960
            or candidate_marker['options'] != {**expected, 'clamp': True}):
        raise ValueError('Candidate generation is incomplete or has different noise/options')


def validate_indices(expected, *indices):
    expected = np.asarray(expected)
    if len(expected) != 960 or len(np.unique(expected)) != 960:
        raise ValueError('The registered validation population must contain 960 distinct cells')
    for array in indices:
        np.testing.assert_array_equal(array, expected)


def checkpoint_proof(plan, out):
    full_path, inference_path = plan['full_checkpoint'], plan['reference_checkpoint']
    if file_sha256(full_path) != plan['full_checkpoint_sha256']:
        raise ValueError('Frozen full snapshot changed')
    if file_sha256(inference_path) != plan['reference_checkpoint_sha256']:
        raise ValueError('Default-EMA reference checkpoint changed')
    full = torch.load(full_path, map_location='cpu', mmap=True, weights_only=False)
    reference = torch.load(inference_path, map_location='cpu', mmap=True, weights_only=False)
    if (full['step'] != 39130 or reference['step'] != 39130
            or reference['weights'] != 'ema_0.9996' or full['config'] != reference['config']
            or full['config']['training'].get('train_subset')):
        raise ValueError('Expected the same full-data mature E35 state')
    model = full['model']
    first, second = full['ema']['0.9996'], full['ema']['0.9999']
    if (len(model) != 193 or len(first) != 189 or set(first) != set(second)
            or set(model) - set(first) != BUFFERS or set(reference['model']) != set(model)):
        raise ValueError('EMA/model tensor schema differs')
    differences = 0
    for name, value in model.items():
        original = first[name] if name in first else value
        if not torch.equal(reference['model'][name], original):
            raise ValueError('Default-EMA reference tensor differs: ' + name)
        if name in first:
            other = second[name]
            if (other.shape != value.shape or other.dtype != value.dtype
                    or not torch.isfinite(other).all() or not torch.isfinite(first[name]).all()):
                raise ValueError('Invalid EMA tensor: ' + name)
            differences += int(not torch.equal(first[name], other))
    if not differences:
        raise ValueError('The proposed EMA factor is unchanged')
    # Exercise the same CPU loading/copying path as the unmodified evaluate CLI.
    data = BenchmarkData(full['config']['data_dir'])
    network = data.build_model(full['config']['model'])
    network.load_state_dict(model, strict=True)
    with torch.no_grad():
        for name, param in network.named_parameters():
            param.copy_(second[name])
    for name, value in network.state_dict().items():
        if not torch.equal(value, second[name] if name in second else model[name]):
            raise ValueError('Actual EMA inference loading differs: ' + name)
    proof = {
        'time': timestamp(), 'cpu_only': True, 'step': 39130,
        'full_checkpoint_sha256': plan['full_checkpoint_sha256'],
        'reference_checkpoint_sha256': plan['reference_checkpoint_sha256'],
        'default193_tensors_equal_full_EMA_and_buffers': True,
        'candidate193_tensors_equal_full_EMA0p9999_and_buffers': True,
        'different_parameter_tensors': differences, 'training_configs_identical': True,
        'new_training': False, 'test_access': False,
    }
    proof_path = out / 'checkpoint_verification.json'
    if proof_path.exists():
        prior = read(proof_path)
        if {k: v for k, v in prior.items() if k != 'time'} != {
                k: v for k, v in proof.items() if k != 'time'}:
            raise ValueError('Keep prior CPU proof; the actual checkpoint checks differ')
    else:
        atomic(proof_path, proof)
    return full['config']


def main_guard(plan):
    run = Path('runs/main_v1')
    if 'trainer_workers' in plan:
        review_trainer_pids(plan['trainer_workers'], run)
    else:
        pid = plan['main_pid']
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            start_tick = int(fields[19])
        except (OSError, ValueError):
            raise ValueError('Original main trainer is no longer alive') from None
        if fields[0] == 'Z' or start_tick != plan['main_start_tick'] or (run / 'finished.json').exists():
            raise ValueError('Original main training phase changed')
    if file_sha256('configs/main_selected.yaml') != plan['main_config_sha256']:
        raise ValueError('Main configuration changed')
    if list(run.glob('test_full*')):
        raise ValueError('Unexpected test artifact before this validation-only review')


def readiness(plan):
    full_root = Path(plan['preceding_validation_root'])
    required = [full_root / 'completed.json', full_root / 'decision.json',
                full_root / 'full_candidate_audit/completed.json']
    reviewed = all(p.is_file() for p in required)
    if reviewed:
        complete, decision, audit = [read(p) for p in required]
        if (complete['status'] != 'completed_validation_review_required'
                or complete['test_access'] or complete['final_test_protocol_adopted']
                or complete['plan_sha256'] != file_sha256(full_root / 'plan.json')
                or audit['plan_sha256'] != plan['preceding_analysis_plan_sha256']
                or audit['test_access'] or decision.get('test_access') is not False):
            raise ValueError('Preceding full validation review provenance differs')
        for proof in (complete, audit):
            for path, digest in proof['output_sha256'].items():
                if file_sha256(path) != digest:
                    raise ValueError('Preceding completed validation review changed: ' + path)
    query = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    pids = sorted(int(line.strip()) for line in query.stdout.splitlines() if line.strip())
    local_review = 'trainer_workers' in plan
    trainer_pids = {worker['pid'] for worker in plan['trainer_workers']} if local_review else {plan['main_pid']}
    gpu_ready = (set(pids).issubset(trainer_pids) if local_review and Path('runs/main_v1/finished.json').exists()
                 else set(pids) == trainer_pids)
    return {'preceding_full_review_complete': reviewed,
            'gpu_only_original_trainer': gpu_ready, 'gpu_pids': pids}


def references(plan, data):
    records = read(plan['coarse_snapshot'])
    population = data.evaluation_indices('validation', 16)
    configs = read(plan['reference_calibration_config'])
    if (configs['weights'] != 'ema_0.9996' or configs['step'] != 39130
            or configs['checkpoint_sha256'] != plan['reference_checkpoint_sha256']
            or configs['batch_size'] != 32 or configs['screen_cap'] != 16):
        raise ValueError('Coarse reference provenance changed')
    result = {}
    for case in plan['cases']:
        matching = [r for r in records if r['tag'] == case['reference_tag']]
        if len(matching) != 1:
            raise ValueError('Missing unique immutable reference case')
        record = matching[0]
        if any(record[k] != case['options'][k] for k in ('guidance', 'cfg_min', 'cfg_max')):
            raise ValueError('Reference record differs from fixed diagnostic setting')
        source = Path(record['cache_provenance']['directory'])
        for name, digest in record['cache_provenance']['files'].items():
            if file_sha256(source / name) != digest:
                raise ValueError('Coarse reference cache changed: ' + name)
        validate_protocol(case['options'], read(source / 'generation_state.json'))
        validate_indices(population, np.load(source / 'indices.npy'))
        result[case['name']] = (source, record)
    return population, result


def analyze(plan, out, data, population, refs, model_config):
    helpers = runpy.run_path('scripts/analyze_full_calibration.py')
    labels = sorted(data.vocab['all_genes'][int(p)]
                    for p in np.unique(data.meta['pert'][population]))
    if len(labels) != 60:
        raise ValueError('Expected all sixty validation perturbations')
    draws = np.random.default_rng(plan['seed']).integers(0, 60, (plan['bootstrap'], 60))
    results, files = {}, []
    expected_population = {'real_perturbed_cells': 4825, 'generated_cells': 960,
                           'shared_real_controls': 4976}
    real_indices = data.evaluation_indices('validation', None)
    for case in plan['cases']:
        source, record = refs[case['name']]
        candidate = Path(case['candidate'])
        config = read(candidate / 'evaluation_config.json')
        validate_protocol(case['options'], read(source / 'generation_state.json'), config,
                          read(candidate / 'generation_state.json'), plan['full_checkpoint'],
                          plan['full_checkpoint_sha256'])
        if config['model'] != model_config:
            raise ValueError('Candidate model config changed')
        validate_indices(population, np.load(candidate / 'indices.npy'))
        raw = np.load(candidate / 'prediction_raw.npy', mmap_mode='r')
        clipped = np.load(candidate / 'prediction.npy', mmap_mode='r')
        if raw.shape != (960, 2000) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError('Invalid candidate prediction population')
        np.testing.assert_array_equal(clipped, np.maximum(raw, 0))
        values = {}
        for key, directory, expected in [
            ('EMA0.9996', source / 'metrics_clamped', record['metrics']),
            ('EMA0.9999', candidate / 'metrics', read(candidate / 'metrics/summary.json')),
        ]:
            if read(directory / 'population_counts.json') != expected_population:
                raise ValueError('Real/generated/control populations differ')
            np.testing.assert_array_equal(np.load(directory / 'real_indices.npy'), real_indices)
            values[key] = helpers['load_metrics'](directory, labels, expected)
            files.extend(directory / p for p in
                         ('summary.json', 'results.csv', 'population_counts.json', 'real_indices.npy'))
        cases, pairs = helpers['bootstrap_candidates'](values, draws)
        results[case['name']] = {'options': case['options'], 'cases': cases,
                                'paired_difference': pairs[0]}
        files.extend(candidate / p for p in ('evaluation_config.json', 'generation_state.json',
                     'indices.npy', 'prediction.npy', 'prediction_raw.npy'))
    result = {
        'time': timestamp(), 'scope': 'Matched mature E35 EMA-only validation diagnostic',
        'cells_each': 960, 'perturbations': 60, 'protocols': results,
        'bootstrap': plan['bootstrap'], 'seed': plan['seed'],
        'source_sha256': {str(p): file_sha256(p) for p in files},
        'note': 'Unadjusted descriptive paired perturbation intervals, fixed training and noise. '
                'Not repeated training-seed or generation-noise uncertainty; all nine metrics '
                'and both prespecified protocols retained. No final selection or test access.',
        'training_changed': False, 'EMA_or_protocol_adopted': False, 'test_access': False,
    }
    atomic(out / 'analysis.json', result)
    plot(result, out, helpers['ORDER'])
    return result


def plot(result, out, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8), layout='constrained')
    for ax, metric in zip(axes.flat, order):
        for i, (name, protocol) in enumerate(result['protocols'].items()):
            row = next(r for r in protocol['paired_difference']['rows'] if r['metric'] == metric)
            ax.plot(row['ci95'], [i, i], color='#3478a6')
            ax.plot(row['difference'], i, 'o', color='#3478a6')
        ax.axvline(0, color='#888888', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(result['protocols'])), list(result['protocols']))
        ax.set_title(metric + (' (lower is better)' if metric in ('MSE', 'MAE') else ''))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.set_xlabel('EMA0.9999 minus EMA0.9996; 95% perturbation interval')
    fig.suptitle('Mature E35 EMA-only diagnostic: 960 cells / 60 perturbations\n'
                 'Same indices, initial Gaussian draws, batch size32 and Heun50; clamp0')
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'paired_ema.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = read(plan_path)
    plan_digest = file_sha256(plan_path)
    torch.set_num_threads(1)
    def verify_current(prerequisites=True):
        verify_pins(plan, plan_path, plan_digest, prerequisites)
    verify_current()
    main_guard(plan)
    out = Path(plan['outdir'])
    with exclusive_run(out, 'ema_review'):
        cfg = checkpoint_proof(plan, out)
        data = BenchmarkData(cfg['data_dir'])
        if file_sha256(data.path / 'expression.npy') != plan['data_expression_sha256']:
            raise ValueError('Processed expression matrix changed')
        population, refs = references(plan, data)
        ready = readiness(plan)
        if args.check_only:
            atomic(out / 'check_only.json', {'time': timestamp(), 'cpu_only': True,
                   'plan_sha256': file_sha256(plan_path), 'references_verified': True,
                   'readiness': ready, 'GPU_generation_launched': False, 'test_access': False})
            print(json.dumps({'check_only_passed': True, 'readiness': ready}))
            return
        if not ready['preceding_full_review_complete'] or not ready['gpu_only_original_trainer']:
            raise ValueError('Wait for complete E35 full review and an available auxiliary GPU slot')
        if (out / 'completed.json').exists():
            raise FileExistsError('Keep the completed EMA review and its provenance')
        child = None
        def interrupt(signum, frame):
            if child is not None and child.poll() is None:
                child.terminate()
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            for case in plan['cases']:
                command = case['command']
                expected = [sys.executable, 'scripts/evaluate.py', '--checkpoint',
                            plan['full_checkpoint'], '--split', 'validation', '--outdir',
                            case['candidate'], '--cap', '16', '--batch-size', '32',
                            '--cfg-min', str(case['options']['cfg_min']), '--cfg-max',
                            str(case['options']['cfg_max']), '--clamp', '--de',
                            '--ema-decay', '0.9999']
                if case['options']['guidance'] is not None:
                    expected += ['--guidance', str(case['options']['guidance'])]
                if command != expected:
                    raise ValueError('Registered command differs from the exact EMA-only protocol')
                verify_current()
                main_guard(plan)
                if not readiness(plan)['gpu_only_original_trainer']:
                    raise ValueError('Another GPU job is active')
                with (out / (case['name'] + '.log')).open('a') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    atomic(out / 'process.json', {'time': timestamp(), 'pid': os.getpid(),
                           'stage': 'generating', 'case': case['name'], 'child_pid': child.pid,
                           'command': command, 'training_changed': False, 'test_access': False})
                    deadline = time.monotonic() + 24 * 3600
                    while child.poll() is None:
                        main_guard(plan)
                        verify_current(prerequisites=False)
                        if time.monotonic() > deadline:
                            raise TimeoutError('Retain cache; EMA case exceeded24 hours')
                        time.sleep(20)
                    if child.returncode:
                        raise RuntimeError(f"EMA case {case['name']} exited{child.returncode}; cache retained")
            atomic(out / 'process.json', {'time': timestamp(), 'pid': os.getpid(),
                   'stage': 'CPU_paired_analysis', 'training_changed': False, 'test_access': False})
            verify_current()
            main_guard(plan)
            if file_sha256(data.path / 'expression.npy') != plan['data_expression_sha256']:
                raise ValueError('Processed expression matrix changed during the EMA review')
            population, refs = references(plan, data)
            result = analyze(plan, out, data, population, refs, cfg['model'])
            outputs = [out / p for p in
                       ('analysis.json', 'paired_ema.png', 'paired_ema.pdf', 'checkpoint_verification.json')]
            atomic(out / 'completed.json', {'time': timestamp(), 'status': 'completed_review_required',
                   'plan_sha256': file_sha256(plan_path),
                   'output_sha256': {str(p): file_sha256(p) for p in outputs},
                   'training_changed': False, 'EMA_or_protocol_adopted': False, 'test_access': False})
            atomic(out / 'process.json', {'time': timestamp(), 'pid': os.getpid(),
                   'stage': 'completed_review_required', 'training_changed': False, 'test_access': False})
            print(json.dumps({k: v['paired_difference']['primary_difference']
                              for k, v in result['protocols'].items()}, indent=2))
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            atomic(out / 'process.json', {'time': timestamp(), 'pid': os.getpid(),
                   'stage': 'failed', 'error': repr(error), 'training_changed': False, 'test_access': False})
            raise


if __name__ == '__main__':
    main()
