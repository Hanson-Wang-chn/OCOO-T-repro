"""Run the registered matched Replogle–Nadig pilots on four local GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from calibrate import write_json_atomic
from ocoot.runtime import exclusive_run, file_sha256, gpu_command, readme_event, timestamp


def main():
    plan_path = Path('configs/repro3_pilots/plan.json')
    plan = json.loads(plan_path.read_text())
    root = Path(plan['run_dir'])
    with exclusive_run(root, 'pilots'):
        (root / 'pilots.pid').write_text(str(os.getpid()))

        def state(stage, case=None):
            write_json_atomic(root / 'pilot_state.json', {
                'time': timestamp(), 'stage': stage, 'case': case,
                'pid': os.getpid(), 'plan_sha256': file_sha256(plan_path),
            })

        for case, registered in plan['cases'].items():
            config = Path(registered['config'])
            if file_sha256(config) != registered['config_sha256']:
                raise ValueError(f'Pilot config changed for {case}')
            settings = yaml.safe_load(config.read_text())
            run = root / case
            if Path(settings['run_dir']) != run:
                raise ValueError(f'Pilot run directory mismatch for {case}')
            generation = run / f'validation_epoch_{plan["epochs"]:03d}'
            run.mkdir(exist_ok=True)
            with (run / 'pipeline.log').open('a') as log:
                try:
                    if not (run / 'finished.json').exists():
                        state('training', case)
                        subprocess.run(gpu_command(['scripts/train.py', '--config', str(config)], plan['gpu_workers']),
                                       stdout=log, stderr=log, check=True)
                    state('diagnosing_validation', case)
                    diagnostic = run / 'zero_floor_validation'
                    subprocess.run([sys.executable, 'scripts/diagnose_zero_floor.py', '--generation-dir', str(generation),
                                    '--outdir', str(diagnostic), '--floors', ','.join(map(str, plan['floors']))],
                                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdout=log, stderr=log, check=True)
                    cases = json.loads((diagnostic / 'comparison.json').read_text())
                    readme_event(f'小实验 `{case}` 完成 EMA 验证。裁负下 PDCorr={cases["floor_0"]["metrics"]["PDCorr"]:.5f}，'
                                 f'MSE={cases["floor_0"]["metrics"]["MSE"]:.5f}，'
                                 f'0.05 设置下 DEPrec={cases["floor_0p05"]["metrics"]["DEPrec"]:.5f}。')
                except subprocess.CalledProcessError:
                    state('failed', case)
                    raise
        state('complete')


if __name__ == '__main__':
    main()
