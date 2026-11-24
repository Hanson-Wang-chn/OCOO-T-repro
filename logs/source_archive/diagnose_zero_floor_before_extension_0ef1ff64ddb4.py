"""Validation-only near-zero support diagnostics on immutable predictions.

Every candidate lies below the smallest observed positive training expression
in the target cell line. This script does not change training or select the
final inference policy. All original predictions remain intact.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics
from ocoot.runtime import exclusive_run, file_sha256, log_event, timestamp


def atomic_json(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation-dir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--floors', default='0,.01,.02,.05,.07')
    args = parser.parse_args()
    try:
        floors = [float(value) for value in args.floors.split(',')]
    except ValueError:
        parser.error('floors must be finite nonnegative numbers')
    if not floors or any(not math.isfinite(value) or value < 0 for value in floors):
        parser.error('floors must be finite nonnegative numbers')
    generation = Path(args.generation_dir)
    out = Path(args.outdir)
    torch.set_num_threads(1)
    with exclusive_run(out, 'zero_floor'):
        checkpoint_path = generation / 'model.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        data = BenchmarkData(checkpoint['config']['data_dir'])
        indices = np.load(generation / 'indices.npy')
        predictions = np.load(generation / 'prediction.npy', mmap_mode='r')
        marker = json.loads((generation / 'generation_state.json').read_text())
        if marker['completed'] != len(indices) or marker['total'] != len(indices):
            raise ValueError('Generation cache is incomplete')
        if not np.all(data.meta['split'][indices] == 1):
            raise ValueError('Only validation predictions are accepted')
        lines = np.unique(data.meta['cell'][indices])
        if len(lines) != 1 or predictions.shape != (len(indices), data.x.shape[1]):
            raise ValueError('Prediction population or gene panel is inconsistent')
        np.testing.assert_array_equal(checkpoint['model']['gene_indices'].numpy(), data.meta['gene_indices'])
        if np.min(predictions) < 0 or not np.isfinite(predictions).all():
            raise ValueError('This study starts from finite, nonnegative cached predictions')
        train = data.train[data.meta['cell'][data.train] == lines[0]]
        minimum_positive = float('inf')
        for a in range(0, len(train), 2048):
            values = np.asarray(data.x[train[a:a+2048]])
            positive = values[values > 0]
            if len(positive):
                minimum_positive = min(minimum_positive, float(positive.min()))
        if not math.isfinite(minimum_positive) or any(value >= minimum_positive for value in floors):
            raise ValueError('A candidate would censor an observed positive training value')
        protocol = {
            'time_independent_settings': {
                'checkpoint': str(checkpoint_path.resolve()), 'checkpoint_sha256': file_sha256(checkpoint_path),
                'step': checkpoint['step'], 'weights': checkpoint.get('weights'),
                'generation_dir': str(generation.resolve()), 'sampling': marker['options'],
                'prediction_sha256': file_sha256(generation / 'prediction.npy'),
                'indices_sha256': file_sha256(generation / 'indices.npy'),
                'metadata_sha256': file_sha256(data.path / 'metadata.npz'),
                'source_sha256': file_sha256(__file__), 'metric_source_sha256': file_sha256('src/ocoot/evaluate.py'),
                'floors': floors, 'target_cell_line': data.vocab['cell_lines'][int(lines[0])],
                'minimum_positive_training_expression': minimum_positive,
                'training_cells_for_support_check': len(train),
            },
            'note': 'Diagnostic only; candidate floors are not paper-specified. No test cells or new generation; no observed positive training entries are removed. Original prediction files stay unchanged.',
        }
        config = out / 'study_config.json'
        if config.exists() and json.loads(config.read_text()) != protocol:
            raise ValueError('Study directory is pinned to different predictions or settings')
        atomic_json(protocol, config)
        atomic_json({'time': timestamp(), 'pid': os.getpid(), 'stage': 'running'}, out / 'study_state.json')
        results = {}
        for floor in floors:
            tag = 'floor_' + format(floor, '.12g').replace('.', 'p')
            case = out / tag
            case.mkdir(exist_ok=True)
            completed = case / 'completed.json'
            if completed.exists():
                record = json.loads(completed.read_text())
            else:
                projected = np.where(predictions < floor, 0, predictions)
                metrics = official_metrics(data, indices, projected, case / 'metrics', de=True)
                record = {'time': timestamp(), 'floor': floor, 'zero_fraction': float(np.mean(projected == 0)),
                          'mean_absolute_change': float(np.mean(np.abs(projected - predictions), dtype=np.float64)),
                          'metrics': metrics}
                atomic_json(record, completed)
                log_event(out / 'events.jsonl', {'event': 'zero_floor_case', **record})
            results[tag] = record
            atomic_json(results, out / 'comparison.json')
        atomic_json({'time': timestamp(), 'pid': os.getpid(), 'stage': 'complete', 'cases': list(results)}, out / 'study_state.json')
        print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
