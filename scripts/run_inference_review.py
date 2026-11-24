"""Paired, validation-only inference diagnostics while full training continues.

The conditional-only reference is diagnostic. This workflow never evaluates
test cells or replaces the final positive-scale dual-CFG calibration.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from ocoot.runtime import exclusive_run, file_sha256, timestamp


def atomic_json(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--guidances', default='none,1,1.5')
    parser.add_argument('--cap', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cfg-min', type=float, default=.1)
    parser.add_argument('--cfg-max', type=float, default=.9)
    args = parser.parse_args()
    try:
        scales = [None if value.strip().lower() == 'none' else float(value) for value in args.guidances.split(',')]
    except ValueError:
        parser.error('guidances must contain none or positive finite numbers')
    if not scales or any(w is not None and (not math.isfinite(w) or w <= 0) for w in scales):
        parser.error('guidances must contain none or positive finite numbers')
    if min(args.cap, args.batch_size) < 1 or not 0 <= args.cfg_min < args.cfg_max <= 1:
        parser.error('invalid sampling dimensions or CFG interval')
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out, 'inference_review'):
        protocol = {
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': file_sha256(args.checkpoint),
            'guidances': scales, 'cap': args.cap, 'batch_size': args.batch_size,
            'cfg_min': args.cfg_min, 'cfg_max': args.cfg_max,
            'split': 'validation', 'clamp': True, 'de': True, 'steps': 50, 'seed': 4242,
            'source_sha256': {str(path): file_sha256(path) for path in [
                Path(__file__), Path('scripts/evaluate.py'), Path('src/ocoot/model.py'),
                Path('src/ocoot/flow.py'), Path('src/ocoot/evaluate.py'),
            ]},
            'note': 'Paired inference diagnostics only, using identical indices and initial-noise seeds. Not final test or final calibration.',
        }
        config_path = out / 'review_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != protocol:
            raise ValueError('Review directory is pinned to a different protocol')
        atomic_json(protocol, config_path)

        def state(stage, **extra):
            atomic_json({'time': timestamp(), 'pid': os.getpid(), 'stage': stage, **extra}, out / 'review_state.json')

        results = {}
        try:
            for w in scales:
                tag = 'ema_none' if w is None else 'ema_cfg_' + format(w, '.12g').replace('.', 'p')
                case = out / tag
                command = [sys.executable, 'scripts/evaluate.py', '--checkpoint', protocol['checkpoint'],
                           '--split', 'validation', '--outdir', str(case), '--cap', str(args.cap),
                           '--batch-size', str(args.batch_size), '--cfg-min', str(args.cfg_min),
                           '--cfg-max', str(args.cfg_max), '--clamp', '--de']
                if w is not None:
                    command += ['--guidance', str(w)]
                marker = case / 'review_completed.json'
                if not marker.exists():
                    with (out / f'{tag}.log').open('a') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        state('evaluating', case=tag, guidance=w, child_pid=child.pid, command=command)
                        code = child.wait()
                    if code:
                        raise RuntimeError(f'{tag} exited with {code}; its cache is retained')
                    atomic_json({'time': timestamp(), 'guidance': w}, marker)
                metrics = json.loads((case / 'metrics/summary.json').read_text())
                results[tag] = {'guidance': w, 'metrics': metrics,
                                'score': metrics['PDCorr'] + sum(metrics[key] for key in ['PDS_L1', 'PDS_L2', 'PDS_cos']) / 3}
                atomic_json(results, out / 'comparison.json')
            guided = [tag for tag, value in results.items() if value['guidance'] is not None]
            state('complete', cases=list(results), best_guided=max(guided, key=lambda tag: results[tag]['score']) if guided else None)
            print(json.dumps(results, indent=2), flush=True)
        except Exception as error:
            state('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
