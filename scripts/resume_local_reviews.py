"""Resume the registered E35/E40 validation workflow on the migrated node."""
import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from ocoot.distributed import live_worker_pids, review_trainer_pids
from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp


RUN = Path('runs/main_v1')
LOG = Path('logs/local_ddp_migration')
OLD_ROOT = '/home/whs/ocoo-t'
OLD_PYTHON = '/home/dataset-assist-0/whs/conda-env/ocoo-t/bin/python'
CONFIG = 'configs/main_selected.yaml'
POSTPROCESS = 'logs/postprocess_workflow_update/plan.json'
E35 = RUN / 'interim_epoch035_calibration'
EMA = RUN / 'ema_epoch035_review'
E40 = RUN / 'interim_epoch040_fixed_cfg'
MOVED = {p: p.with_name(p.name + '_local_ddp') for p in (E35, EMA, E40)}
LEGACY_SOURCES = {
    'src/ocoot/train.py': LOG / 'train_before.py',
    'logs/run_epoch035_validation_calibration.py': LOG / 'calibration_controller_before.py',
    'scripts/review_ema_weights.py': LOG / 'ema_review_before.py',
    'logs/follow_epoch040_fixed_cfg.py': LOG / 'epoch040_controller_before.py',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def relocate(value, output_map=()):
    if isinstance(value, dict):
        return {relocate(k, output_map): relocate(v, output_map) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(item, output_map) for item in value]
    if not isinstance(value, str):
        return value
    if value == OLD_PYTHON:
        return sys.executable
    if value.startswith(OLD_ROOT + '/'):
        value = str(Path.cwd()) + value[len(OLD_ROOT):]
    for source, target in output_map:
        for before, after in ((str(source), str(target)), (str(source.resolve()), str(target.resolve()))):
            if value == before or value.startswith(before + '/'):
                return after + value[len(before):]
    return value


def old_plan(path):
    plan = read(path)
    for group in ('source_sha256', 'prerequisite_sha256'):
        for source, digest in plan[group].items():
            physical_path = LEGACY_SOURCES.get(source, relocate(source))
            if file_sha256(physical_path) != digest:
                raise ValueError(f'Original registered input differs: {source}')
    return plan


def migrated_plan(source, output_map, **changes):
    original = old_plan(source)
    plan = relocate(original, output_map)
    plan.update(changes, migration={'time': timestamp(), 'original': str(source),
                                    'sha256': file_sha256(source), 'python': sys.executable,
                                    'project_root': str(Path.cwd())})
    # Keep original prerequisite pins at their original paths. Explicitly pin
    # each migrated counterpart in the plan that will consume it.
    plan['prerequisite_sha256'] = relocate(original['prerequisite_sha256'])
    plan['source_sha256'] = {path: file_sha256(path) for path in plan['source_sha256']}
    if any(path in plan['source_sha256'] for path in LEGACY_SOURCES):
        plan['source_sha256']['src/ocoot/distributed.py'] = file_sha256('src/ocoot/distributed.py')
    return plan


def copy_calibration():
    source = E35 / 'calibration'
    destination = MOVED[E35] / 'calibration'
    config = read(source / 'calibration_config.json')
    if file_sha256(relocate(config['checkpoint'])) != config['checkpoint_sha256']:
        raise ValueError('The relocated frozen E35 checkpoint differs')
    if (destination / 'calibration_config.json').exists():
        if read(destination / 'calibration_config.json') != relocate(config):
            raise ValueError('The migrated calibration configuration differs')
        return
    for record in read(source / 'all_results.json'):
        for name, digest in record['cache_provenance']['files'].items():
            if file_sha256(source / record['tag'] / name) != digest:
                raise ValueError(f'Original calibration cache differs: {record["tag"]}/{name}')
    marker = read(source / 'refine_01/generation_state.json')
    if marker != {'completed': 1024, 'total': 4825, 'options': {
            'noise_std': 1., 'batch_size': 32, 'steps': 50, 'guidance': 2.,
            'cfg_min': .1, 'cfg_max': .9, 'clamp': False, 'seed': 4242}}:
        raise ValueError('Unexpected partial E35 generation state')
    for case in sorted(source.iterdir()):
        if case.is_dir() and case.name.startswith(('screen_', 'refine_')):
            shutil.copytree(case, destination / case.name, dirs_exist_ok=True)
    cache_proof = {}
    for case in sorted(destination.iterdir()):
        cache_proof[case.name] = {}
        for filename in ('prediction.npy', 'indices.npy', 'generation_state.json'):
            digest = file_sha256(case / filename)
            if digest != file_sha256(source / case.name / filename):
                raise ValueError(f'Cache copy differs: {case.name}/{filename}')
            cache_proof[case.name][filename] = digest
    write(MOVED[E35] / 'copied_caches.json', {'files': cache_proof, 'partial': marker})
    write(destination / 'all_results.json', relocate(read(source / 'all_results.json'), [(E35, MOVED[E35])]))
    write(destination / 'calibration_config.json', relocate(config))


def register(workers, artifacts):
    rank_zero = workers[0]
    old_checkpoint = E35 / 'first_full_floor_diagnostic/metrics_floor_0p05/summary.json'
    full = migrated_plan(E35 / 'plan.json', [(E35, MOVED[E35])],
                         trainer_pid=rank_zero['pid'], trainer_start_tick=str(rank_zero['start_tick']),
                         trainer_workers=workers, main_pipeline_pid=None)
    floor_proof = read(E35 / 'first_full_floor_diagnostic/completed.json')
    digest = floor_proof['output_sha256'][str(old_checkpoint)]
    if file_sha256(old_checkpoint) != digest:
        raise ValueError('The earlier E35 floor metrics differ from their completion proof')
    full['prerequisite_sha256'][str(old_checkpoint)] = digest
    full['prerequisite_sha256'][str(MOVED[E35] / 'copied_caches.json')] = file_sha256(MOVED[E35] / 'copied_caches.json')
    full_path = MOVED[E35] / 'plan.json'
    write(full_path, full)
    analysis = migrated_plan(E35 / 'analysis_plan.json', [(E35, MOVED[E35])])
    analysis['prerequisite_sha256'][str(full_path)] = file_sha256(full_path)
    analysis_path = MOVED[E35] / 'analysis_plan.json'
    write(analysis_path, analysis)

    ema = migrated_plan(EMA / 'plan.json', [(EMA, MOVED[EMA]), (E35, MOVED[E35])],
                        main_pid=rank_zero['pid'], main_start_tick=rank_zero['start_tick'],
                        trainer_workers=workers, preceding_analysis_plan_sha256=file_sha256(analysis_path))
    coarse = MOVED[EMA] / 'screen_results_snapshot.json'
    write(coarse, relocate(read(E35 / 'coarse_review/screen_results_snapshot.json')))
    ema['coarse_snapshot'] = str(coarse)
    for path in (coarse, full_path, analysis_path, Path(ema['reference_calibration_config'])):
        ema['prerequisite_sha256'][str(path)] = file_sha256(path)
    ema_path = MOVED[EMA] / 'plan.json'
    write(ema_path, ema)

    later = migrated_plan(E40 / 'plan.json', [(E40, MOVED[E40])],
                          trainer_pid=rank_zero['pid'], trainer_start_tick=str(rank_zero['start_tick']),
                          trainer_workers=workers)
    reference = Path(later['reference'])
    copied_reference = MOVED[E40] / 'reference_epoch035'
    for name in ('indices.npy', 'generation_state.json', 'prediction.npy',
                 'prediction_raw.npy', 'metrics/summary.json', 'metrics/results.csv'):
        target = copied_reference / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to((reference / name).resolve())
    reference_config = copied_reference / 'evaluation_config.json'
    write(reference_config, relocate(read(reference / 'evaluation_config.json')))
    later['reference'] = str(copied_reference)
    later['fixed_comparison_command'][3] = str(copied_reference)
    later['prerequisite_sha256'][str(reference_config)] = file_sha256(reference_config)
    later['prerequisite_sha256'][str(full_path)] = file_sha256(full_path)
    later_path = MOVED[E40] / 'plan.json'
    write(later_path, later)
    write(LOG / 'review_registration.json', {'time': timestamp(), 'worker_identity': workers,
          'path_mapping': {OLD_ROOT: str(Path.cwd()), OLD_PYTHON: sys.executable,
                           **{str(k): str(v) for k, v in MOVED.items()}},
          'plans': {str(path): file_sha256(path) for path in (full_path, analysis_path, ema_path, later_path)},
          'artifacts': str(artifacts), 'test_access': False})


def verify_floors_and_record():
    root = MOVED[E35]
    review = read(root / 'calibration/postprocessing/review.json')
    original = read(E35 / 'first_full_floor_diagnostic/metrics_floor_0p05/summary.json')
    point = next(c for c in review['candidates'] if c['tag'] == 'refine_00' and c['zero_floor'] == .05)
    np.testing.assert_allclose([point['metrics'][k] for k in original], list(original.values()), rtol=0, atol=1e-10)
    audit = root / 'full_candidate_audit/analysis.json'
    write(root / 'decision.json', {'time': timestamp(), 'test_access': False,
          'earlier_floor_0p05_matches_formal': {k: point['metrics'][k] - value for k, value in original.items()},
          'all_candidates': [{'tag': c['tag'], 'floor': c['zero_floor'], 'metrics': c['metrics'],
                              'failed_checks': [k for k, passed in c['checks'].items() if not passed]}
                             for c in review['candidates']],
          'selector_choice': review['selected'], 'audit_sha256': file_sha256(audit),
          'final_protocol_adopted': False})
    readme_event(f'冻结 E35 四候选审计完成，提前与正式重算的 .05 九项逐一一致。'
                 f'各候选指标、配对区间和检查结果见 `{root}/decision.json` 及其审计目录。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--aux-gpu', required=True)
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--share-training-gpu', action='store_true',
                        help='Run GPU reviews alongside the registered DDP workers')
    args = parser.parse_args()
    artifacts = args.artifacts_root.resolve(strict=True)
    with exclusive_run(LOG, 'reviews'):
        execution = read(RUN / 'training_execution.json')
        workers = execution['workers']
        if len(workers) != 4 or args.aux_gpu not in execution['cuda_visible_devices'].split(','):
            raise ValueError('Expected the four registered training workers and a visible auxiliary GPU')
        if read(RUN / 'checkpoint_state.json')['step'] <= 41374:
            raise ValueError('Wait for a durable post-resume checkpoint')
        review_trainer_pids(workers, RUN)
        copy_calibration()
        registration = LOG / 'review_registration.json'
        if registration.exists():
            record = read(registration)
            if record['worker_identity'] != workers:
                raise ValueError('Review plans belong to an earlier training launch')
            for path, digest in record['plans'].items():
                if file_sha256(path) != digest:
                    raise ValueError(f'Migrated review plan changed: {path}')
        else:
            register(workers, artifacts)
        if args.register_only:
            return

        def execute(script, *arguments, gpu=False):
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.aux_gpu if gpu else '',
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            command = [sys.executable, script, *map(str, arguments)]
            write(LOG / 'reviews_state.json', {'time': timestamp(), 'pid': os.getpid(), 'script': script,
                                              'command': command, 'cuda_visible_devices': environment['CUDA_VISIBLE_DEVICES']})
            subprocess.run(command, env=environment, check=True)

        def preserve_epoch40():
            snapshot = RUN / 'snapshots/step_44720'
            while not (snapshot / 'full.pt').exists():
                write(LOG / 'reviews_state.json', {'time': timestamp(), 'pid': os.getpid(),
                                                  'stage': 'waiting_for_epoch40_checkpoint'})
                live_worker_pids(workers)
                time.sleep(20)
            if not (snapshot / 'preserved.json').exists():
                execute('scripts/preserve_checkpoint.py', '--run-dir', RUN, '--step', '44720',
                        '--destination', artifacts / 'main-snapshots/epoch040/full.pt')

        def wait_for_training():
            write(LOG / 'reviews_state.json', {'time': timestamp(), 'pid': os.getpid(),
                                              'stage': 'waiting_for_main_training_completion'})
            while not (RUN / 'finished.json').exists():
                live_worker_pids(workers)
                time.sleep(30)

        if not args.share_training_gpu:
            preserve_epoch40()
            wait_for_training()

        execute('logs/run_epoch035_validation_calibration.py', '--plan', MOVED[E35] / 'plan.json', gpu=True)
        if not (MOVED[E35] / 'full_candidate_audit/completed.json').exists():
            execute('scripts/analyze_full_calibration.py', '--plan', MOVED[E35] / 'analysis_plan.json')
        if not (MOVED[E35] / 'decision.json').exists():
            verify_floors_and_record()
        if not (MOVED[EMA] / 'completed.json').exists():
            execute('scripts/review_ema_weights.py', '--plan', MOVED[EMA] / 'plan.json', gpu=True)

        preserve_epoch40()
        execute('logs/follow_epoch040_fixed_cfg.py', '--plan', MOVED[E40] / 'plan.json', gpu=True)

        wait_for_training()
        execute('scripts/run_experiment.py', '--config', CONFIG, '--postprocess-plan', POSTPROCESS, gpu=True)


if __name__ == '__main__':
    main()
