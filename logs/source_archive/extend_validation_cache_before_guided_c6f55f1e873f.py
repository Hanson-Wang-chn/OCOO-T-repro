"""Extend a frozen, capped EMA validation cache to all validation cells.

The existing predictions are retained exactly; only missing cells are sampled.
This ordinary-field, nested-population diagnostic is separate from formal CFG
calibration, full-validation gates, and test. It never changes training.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import generate, official_metrics
from ocoot.runtime import exclusive_run, file_sha256, log_event, readme_event, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    source, out = Path(args.source_dir), Path(args.outdir)
    if source.resolve() == out.resolve():
        parser.error('source and output must differ')
    with exclusive_run(out, 'extension'):
        (out / 'extension.pid').write_text(str(os.getpid()))
        plan = json.loads((out / 'plan.json').read_text())
        if plan['script_sha256'] != file_sha256(__file__):
            raise ValueError('Extension implementation differs from registered plan')
        checkpoint_path = source / 'model.pt'
        marker = json.loads((source / 'generation_state.json').read_text())
        base = np.load(source / 'indices.npy')
        options = marker['options']
        if marker['completed'] != marker['total'] or marker['total'] != len(base):
            raise ValueError('Source generation is incomplete')
        if options['guidance'] is not None or not options['clamp'] or options['steps'] != 50:
            raise ValueError('This diagnostic requires ordinary-field, nonnegative Heun50 source samples')
        for name, digest in plan['source_sha256'].items():
            if file_sha256(source / name) != digest:
                raise ValueError(f'Registered source changed: {name}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        config = checkpoint['config']
        if config['training'].get('train_subset') or checkpoint.get('weights') != 'ema_0.9996':
            raise ValueError('Requires full-training EMA0.9996 weights')
        if checkpoint['step'] != plan['step']:
            raise ValueError('Registered checkpoint step differs')
        data = BenchmarkData(config['data_dir'])
        if len(np.unique(base)) != len(base) or not np.isin(base, data.val).all():
            raise ValueError('Source must contain distinct validation cells only')
        extra = data.val[~np.isin(data.val, base)]
        indices = np.concatenate([base, extra])
        np.testing.assert_array_equal(np.sort(indices), data.val)
        if len(extra) != plan['extra_cells'] or len(indices) != plan['full_cells']:
            raise ValueError('Population differs from registered plan')
        protocol = {'plan_sha256': file_sha256(out / 'plan.json'),
                    'source_dir': str(source.resolve()), 'source_options': options,
                    'source_checkpoint_sha256': plan['source_sha256']['model.pt'],
                    'step': checkpoint['step'], 'weights': checkpoint['weights'],
                    'base_cells': len(base), 'extra_cells': len(extra), 'full_cells': len(indices),
                    'extra_seed': plan['extra_seed'], 'extra_batch_size': options['batch_size'],
                    'data_metadata_sha256': file_sha256(data.path / 'metadata.npz'),
                    'note': 'All validation cells, preserving the complete capped prefix. '
                            'Extra cells use a separate RNG segment with seed+base_count. '
                            'Ordinary-field finite-population diagnostic, not formal dual-CFG or test.'}
        protocol_path = out / 'extension_config.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Output is pinned to a different extension protocol')
        atomic_json(protocol_path, protocol)
        if (out / 'completed.json').exists():
            print((out / 'metrics_full/summary.json').read_text())
            return
        np.save(out / 'indices.npy', indices)
        extra_out = out / 'extra'
        extra_marker = extra_out / 'generation_state.json'
        extra_complete = (extra_marker.exists() and
                          json.loads(extra_marker.read_text())['completed'] == len(extra))
        if not extra_complete:
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = True
            model = data.build_model(config['model']).cuda().eval()
            model.load_state_dict(checkpoint['model'])
            forward = torch.compile(model, fullgraph=True)
            log_event(out / 'events.jsonl', {'event': 'extra_generation_started', 'extra_cells': len(extra)})
            generate(forward, data, extra, extra_out, noise_std=options['noise_std'],
                     batch_size=options['batch_size'], steps=50, guidance=None,
                     cfg_min=options['cfg_min'], cfg_max=options['cfg_max'], clamp=True,
                     seed=plan['extra_seed'])
            del forward, model
            torch.cuda.empty_cache()
        else:
            np.testing.assert_array_equal(np.load(extra_out / 'indices.npy'), extra)
            extra_options = json.loads(extra_marker.read_text())['options']
            if extra_options != {**options, 'seed': plan['extra_seed']}:
                raise ValueError('Completed extra cache has different options')
        del checkpoint
        stitched = {}
        for filename, destination in [('prediction.npy', 'stitched_predictions.npy'),
                                      ('prediction_raw.npy', 'stitched_raw.npy')]:
            original = np.load(source / filename, mmap_mode='r')
            additional = np.load(extra_out / filename, mmap_mode='r')
            target = np.lib.format.open_memmap(out / destination, mode='w+', dtype=np.float32,
                                              shape=(len(indices), data.x.shape[1]))
            if original.shape != (len(base), data.x.shape[1]) or additional.shape != (len(extra), data.x.shape[1]):
                raise ValueError('Gene or population dimension mismatch')
            target[:len(base)] = original
            target[len(base):] = additional
            target.flush()
            np.testing.assert_array_equal(target[:len(base)], original)
            if not np.isfinite(target).all():
                raise ValueError('Nonfinite stitched predictions')
            stitched[destination] = target
        np.testing.assert_array_equal(stitched['stitched_predictions.npy'], np.maximum(stitched['stitched_raw.npy'], 0))
        metrics = official_metrics(data, indices, stitched['stitched_predictions.npy'], out / 'metrics_full', de=True)
        raw_metrics = official_metrics(data, indices, stitched['stitched_raw.npy'], out / 'metrics_raw_full')
        atomic_json(out / 'completed.json', {'time': timestamp(), **protocol, 'metrics': metrics,
                                           'raw_metrics': raw_metrics, 'prefix_predictions_exactly_preserved': True})
        readme_event(f'第15轮普通条件缓存扩展诊断完成：保留原{len(base)}个预测，只新增{len(extra)}个，覆盖全部{len(indices)}验证细胞；'
                     f'PDCorr={metrics["PDCorr"]:.4f}、PDS_L2={metrics["PDS_L2"]:.4f}、DEPrec={metrics["DEPrec"]:.4f}。'
                     f'这是有限生成数量诊断，非正式CFG或测试；结果 `{out}/completed.json`。')
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
