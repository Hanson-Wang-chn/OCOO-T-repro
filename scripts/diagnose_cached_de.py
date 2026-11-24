"""Score complete validation caches with DE, without loading a model on GPU.

This is an intermediate diagnostic; capped or unguided predictions are not the
final paper benchmark. Original training-validation artifacts stay unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics
from ocoot.runtime import exclusive_run, file_sha256, log_event, timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation-dir', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--outdir')
    args = parser.parse_args()
    generation = Path(args.generation_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else generation / 'model.pt'
    out = Path(args.outdir) if args.outdir else generation / 'metrics_de_diagnostic'
    torch.set_num_threads(1)
    with exclusive_run(out, 'cached_de'):
        marker = json.loads((generation / 'generation_state.json').read_text())
        indices = np.load(generation / 'indices.npy')
        predictions = np.load(generation / 'prediction.npy', mmap_mode='r')
        if marker['completed'] != marker['total'] or marker['total'] != len(indices):
            raise ValueError('Generation cache is incomplete')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        data = BenchmarkData(checkpoint['config']['data_dir'])
        if not np.all(data.meta['split'][indices] == 1):
            raise ValueError('This diagnostic accepts validation cells only')
        np.testing.assert_array_equal(checkpoint['model']['gene_indices'].numpy(), data.meta['gene_indices'])
        if predictions.shape != (len(indices), data.x.shape[1]):
            raise ValueError('Prediction shape does not match the fixed gene panel')
        if not np.isfinite(predictions).all() or np.min(predictions) < 0:
            raise ValueError('DE requires finite, nonnegative cached predictions')
        protocol = {
            'checkpoint': str(checkpoint_path.resolve()),
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'step': checkpoint['step'], 'weights': checkpoint.get('weights'),
            'generation_dir': str(generation.resolve()), 'sampling': marker['options'],
            'prediction_sha256': file_sha256(generation / 'prediction.npy'),
            'indices_sha256': file_sha256(generation / 'indices.npy'),
            'metadata_sha256': file_sha256(data.path / 'metadata.npz'),
            'script_sha256': file_sha256(__file__),
            'metric_source_sha256': file_sha256('src/ocoot/evaluate.py'),
            'note': 'Validation-only cached diagnostic, no new generation. Sampling cap and guidance differ from the final calibrated full benchmark.',
        }
        config_path = out / 'diagnostic_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != protocol:
            raise ValueError('Output directory is pinned to a different cache or protocol')
        config_path.write_text(json.dumps(protocol, indent=2))
        if (out / 'completed.json').exists():
            print((out / 'summary.json').read_text())
            return
        del checkpoint
        log_event(out / 'events.jsonl', {'event': 'cached_de_started', 'generated_cells': len(indices)})
        metrics = official_metrics(data, indices, predictions, out, de=True)
        original_path = generation / 'metrics/summary.json'
        differences = {}
        if original_path.exists():
            original = json.loads(original_path.read_text())
            differences = {key: metrics[key] - value for key, value in original.items() if key in metrics}
            if not all(np.isclose(value, 0, atol=1e-9, rtol=0) for value in differences.values()):
                raise ValueError(f'Array versus DE-wrapper metrics disagree: {differences}')
        (out / 'array_metric_comparison.json').write_text(json.dumps(differences, indent=2))
        complete = out / 'completed.json'
        temp = complete.with_suffix('.tmp')
        temp.write_text(json.dumps({'time': timestamp(), 'generated_cells': len(indices), 'array_metric_agreement': True}, indent=2))
        temp.replace(complete)
        log_event(out / 'events.jsonl', {'event': 'cached_de_completed', **metrics})
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
