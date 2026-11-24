"""Training-derived near-zero truncation sensitivity, using fixed validation samples.

Unlike diagnose_zero_floor.py's below-minimum cleanup, these diagnostic cutoffs
may censor a measured small fraction of real positive training entries. They
are not specified by the paper and are never selected for formal inference by
this script. No model, original sample, metric definition, or test is changed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics
from ocoot.runtime import exclusive_run, file_sha256, log_event, timestamp


QUANTILES = [.0005, .001, .0025, .005, .01]
BASE_FLOORS = [0., .05, .07, .1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-dir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--evaluate', action='store_true', help='Otherwise audit support only; no DE scores')
    args = parser.parse_args()
    torch.set_num_threads(1)
    generation, out = Path(args.generation_dir).resolve(), Path(args.outdir)
    with exclusive_run(out, 'expression_truncation'):
        config = json.loads((generation / 'evaluation_config.json').read_text())
        if config['split'] != 'validation' or not config['clamp'] or not config['de']:
            raise ValueError('Requires completed nonnegative validation nine-metric samples')
        checkpoint_path = Path(config['checkpoint'])
        if file_sha256(checkpoint_path) != config['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        if checkpoint['step'] != config['step'] or checkpoint.get('weights') != config['weights']:
            raise ValueError('Checkpoint step or inference weights differ from the sample record')
        data = BenchmarkData(checkpoint['config']['data_dir'])
        indices = np.load(generation / 'indices.npy')
        np.testing.assert_array_equal(indices, data.evaluation_indices('validation', config['cap']))
        np.testing.assert_array_equal(checkpoint['model']['gene_indices'].numpy(), data.meta['gene_indices'])
        marker = json.loads((generation / 'generation_state.json').read_text())
        if marker['completed'] != len(indices) or marker['total'] != len(indices):
            raise ValueError('Incomplete generation cache')
        expected_options = {key: config[key] for key in
                            ('noise_std', 'batch_size', 'steps', 'guidance', 'cfg_min', 'cfg_max', 'clamp', 'seed')}
        if marker['options'] != expected_options:
            raise ValueError('Sampling provenance differs')
        lines = np.unique(data.meta['cell'][indices])
        if len(lines) != 1 or not np.all(data.meta['split'][indices] == 1):
            raise ValueError('Only one validation cell line is supported')
        predictions = np.load(generation / 'prediction.npy', mmap_mode='r')
        if (predictions.dtype != np.float32 or predictions.shape != (len(indices), data.x.shape[1])
                or not np.isfinite(predictions).all() or np.any(predictions < 0)):
            raise ValueError('Unexpected finite nonnegative sample schema')
        protocol = {'generation_dir': str(generation), 'checkpoint_sha256': config['checkpoint_sha256'],
                    'step': checkpoint['step'], 'weights': config['weights'], 'sampling': expected_options,
                    'quantiles_of_positive_training_values': QUANTILES, 'base_floors': BASE_FLOORS,
                    'maximum_positive_training_censored_fraction': .01,
                    'prediction_sha256': file_sha256(generation / 'prediction.npy'),
                    'indices_sha256': file_sha256(generation / 'indices.npy'),
                    'generation_config_sha256': file_sha256(generation / 'evaluation_config.json'),
                    'data_files_sha256': {name: file_sha256(data.path / name) for name in
                                          ('expression.npy', 'metadata.npz', 'vocab.json')},
                    'source_sha256': {name: file_sha256(name) for name in
                                      (__file__, 'src/ocoot/evaluate.py', 'src/ocoot/data.py', 'environment.lock.txt')},
                    'test_cells_used_for_statistics_or_evaluation': False,
                    'original_predictions_or_training_changed': False,
                    'formal_postprocess_selected': False}
        protocol_file = out / 'study_config.json'
        if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
            raise ValueError('Output is pinned to different data, samples or policy')
        atomic_json(protocol_file, protocol)
        audit_file = out / 'support_audit.json'
        if audit_file.exists():
            audit = json.loads(audit_file.read_text())
        else:
            train = data.train[data.meta['cell'][data.train] == lines[0]]
            positive_chunks = []
            zero_counts = np.zeros(data.x.shape[1], dtype=np.int64)
            for offset in range(0, len(train), 2048):
                values = np.asarray(data.x[train[offset:offset + 2048]])
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('Training expression schema differs')
                zero_counts += np.count_nonzero(values == 0, axis=0)
                positive_chunks.append(values[values > 0])
            positive = np.concatenate(positive_chunks)
            del positive_chunks
            quantiles = np.quantile(positive, QUANTILES, overwrite_input=True)
            floors = sorted(set(BASE_FLOORS + quantiles.tolist()))
            real_zero = zero_counts / len(train)
            entries = len(train) * data.x.shape[1]
            cases = []
            for floor in floors:
                removed = positive[positive < floor]
                fraction = len(removed) / len(positive)
                if fraction > .01 + 1 / len(positive):
                    raise ValueError('Cutoff would censor more than1% of positive training entries')
                projected_zero = np.mean(predictions < floor, axis=0) if floor else np.mean(predictions == 0, axis=0)
                cases.append({'floor': floor, 'positive_training_fraction_censored': fraction,
                              'all_training_fraction_censored': len(removed) / entries,
                              'training_mean_absolute_change': float(removed.sum(dtype=np.float64) / entries),
                              'generated_zero_fraction': float(projected_zero.mean()),
                              'gene_zero_fraction_correlation_with_training': float(np.corrcoef(real_zero, projected_zero)[0, 1]),
                              'gene_zero_fraction_mae_to_training': float(np.mean(np.abs(real_zero - projected_zero))),
                              'generated_mean_absolute_change': float(predictions[predictions < floor].sum(dtype=np.float64) / predictions.size)})
            audit = {'time': timestamp(), 'training_cells': len(train), 'training_entries': entries,
                     'positive_training_entries': len(positive), 'target_cell_line': data.vocab['cell_lines'][int(lines[0])],
                     'training_zero_fraction': float(real_zero.mean()), 'minimum_positive_training_value': float(positive.min()),
                     'training_positive_quantiles': dict(zip(map(str, QUANTILES), map(float, quantiles))),
                     'cases': cases, 'protocol_sha256': file_sha256(protocol_file),
                     'note': 'Train-derived sensitivity grid. Measured real positive entries may be censored; '
                             'this is an unreported postprocessing variant, not below-support numerical cleanup. '
                             'No validation DE metric has been computed by the audit.'}
            atomic_json(audit_file, audit)
        if audit['protocol_sha256'] != file_sha256(protocol_file):
            raise ValueError('Support audit provenance changed')
        if not args.evaluate:
            print(json.dumps(audit, indent=2), flush=True)
            return
        atomic_json(out / 'study_state.json', {'time': timestamp(), 'pid': os.getpid(), 'stage': 'evaluating'})
        results = {}
        for i, spec in enumerate(audit['cases']):
            floor = spec['floor']
            case = out / f'case_{i:02d}'
            case.mkdir(exist_ok=True)
            completed = case / 'completed.json'
            identity = {'support_audit_sha256': file_sha256(audit_file), 'floor': floor}
            if completed.exists():
                record = json.loads(completed.read_text())
                if record['identity'] != identity:
                    raise ValueError('Cached cutoff case differs')
            else:
                projected = np.where(predictions < floor, 0, predictions)
                metrics = official_metrics(data, indices, projected, case / 'metrics', de=True)
                if floor == 0:
                    original = json.loads((generation / 'metrics/summary.json').read_text())
                    np.testing.assert_allclose([metrics[k] for k in original], list(original.values()), rtol=0, atol=1e-10)
                record = {'time': timestamp(), 'identity': identity, **spec, 'metrics': metrics,
                          'score': metrics['PDCorr'] + sum(metrics[k] for k in ('PDS_L1', 'PDS_L2', 'PDS_cos')) / 3}
                atomic_json(completed, record)
                log_event(out / 'events.jsonl', {'event': 'expression_truncation_case', **record})
            results[case.name] = record
            atomic_json(out / 'comparison.json', results)
        pd.DataFrame([{'case': key, 'floor': r['floor'], 'positive_training_fraction_censored': r['positive_training_fraction_censored'],
                       'generated_zero_fraction': r['generated_zero_fraction'], 'score': r['score'], **r['metrics']}
                      for key, r in results.items()]).to_csv(out / 'comparison.csv', index=False)
        atomic_json(out / 'study_state.json', {'time': timestamp(), 'pid': os.getpid(), 'stage': 'complete',
                                              'cases': list(results), 'formal_postprocess_selected': False})
        print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
