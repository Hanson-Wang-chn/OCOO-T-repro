"""Strict paired nine-metric review of the registered mature batch-drop trial."""
import argparse
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ocoot.evaluate import METRICS
from ocoot.runtime import exclusive_run, file_sha256, timestamp


ORDER = ['PDCorr', 'DEOver', 'DEPrec', 'DirAgr', 'PDS_L1', 'PDS_L2', 'PDS_cos', 'MSE', 'MAE']


def paired_summary(reference, candidate, bootstrap=2000, seed=9232):
    if reference.shape != candidate.shape or reference.shape[1] != len(ORDER):
        raise ValueError('Expected equal perturbation-by-nine-metric arrays')
    helpers = runpy.run_path('scripts/analyze_inference_review.py')
    draws = np.random.default_rng(seed).integers(0, len(reference), (bootstrap, len(reference)))
    delta = helpers['resampled_means'](candidate, draws) - helpers['resampled_means'](reference, draws)
    means = [np.nanmean(array, axis=0) for array in (reference, candidate)]
    scores = [float(mean[0] + mean[[4, 5, 6]].mean()) for mean in means]
    score_draws = delta[:, 0] + delta[:, [4, 5, 6]].mean(axis=1)
    return {'reference_score': scores[0], 'candidate_score': scores[1],
            'score_difference': {'value': scores[1] - scores[0],
                                 'ci95': np.nanquantile(score_draws, [.025, .975]).tolist()},
            'rows': [{'metric': key, 'reference': float(means[0][i]), 'candidate': float(means[1][i]),
                      'difference': float(means[1][i] - means[0][i]),
                      'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()}
                     for i, key in enumerate(ORDER)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-dir', default='runs/batch_dropout_mature_trial')
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.trial_dir)
    with exclusive_run(out, 'paired_analysis'):
        plan = json.loads((out / 'plan.json').read_text())
        execution = json.loads((out / 'execution_config.json').read_text())
        completed = json.loads((out / 'completed.json').read_text())
        bounded = json.loads((out / 'bounded_training_completed.json').read_text())
        if (execution['plan_sha256'] != file_sha256(out / 'plan.json')
                or completed['step'] != plan['reference_step']
                or bounded['step'] != completed['step'] or bounded['updates_from_base'] != plan['updates']):
            raise ValueError('Unregistered or unequal optimization budget')
        for name, digest in execution['source_sha256'].items():
            if file_sha256(name) != digest:
                raise ValueError(f'Execution source changed: {name}')
        helper = runpy.run_path('scripts/run_batch_dropout_trial.py')
        models = [out / 'ema_models' / f'{tag}.pt' for tag in ('reference', 'candidate')]
        states = [torch.load(path, map_location='cpu', mmap=True, weights_only=False) for path in models]
        helper['configs_match'](states[0]['config'], states[1]['config'])
        sources = [out / name for name in ('plan.json', 'execution_config.json', 'completed.json',
                    'bounded_training_completed.json', 'fork_provenance.json', 'prelaunch_verification.json')]
        sources += models
        for tag, state, path in zip(('reference', 'candidate'), states, models):
            export = completed['exports'][tag]
            if (file_sha256(path) != export['export_sha256']
                    or state['source_checkpoint_sha256'] != export['source_checkpoint_sha256']
                    or state['step'] != completed['step'] or state['weights'] != 'ema_0.9996'):
                raise ValueError(f'Default-EMA export differs: {tag}')
            for name, digest in completed['case_sha256'][tag].items():
                if file_sha256(out / tag / name) != digest:
                    raise ValueError(f'Frozen evaluation changed: {tag}/{name}')
                sources.append(out / tag / name)
        for key in ('esm', 'gene_indices', 'missing', 'position'):
            if not torch.equal(states[0]['model'][key], states[1]['model'][key]):
                raise ValueError(f'Frozen input buffer changed: {key}')
        if states[0]['model'].keys() != states[1]['model'].keys():
            raise ValueError('Parameter schema changed')
        for key, tensor in states[0]['model'].items():
            other = states[1]['model'][key]
            if tensor.shape != other.shape or tensor.dtype != other.dtype:
                raise ValueError('Parameter shape or dtype changed')
        from ocoot.data import BenchmarkData
        data = BenchmarkData(states[0]['config']['data_dir'])
        expected_indices = data.evaluation_indices('validation', 16)
        expected_labels = sorted(data.vocab['all_genes'][int(p)] for p in np.unique(data.meta['pert'][expected_indices]))
        marker = None
        for tag in ('reference', 'candidate'):
            case = out / tag
            np.testing.assert_array_equal(np.load(case / 'indices.npy'), expected_indices)
            current = json.loads((case / 'generation_state.json').read_text())
            if current['completed'] != len(expected_indices) or current['total'] != len(expected_indices):
                raise ValueError('Generation budget is incomplete')
            if marker is not None and marker != current:
                raise ValueError('Noise, population or sampling options differ')
            marker = current
            expected_options = {'noise_std': states[0]['config']['training']['noise_std'],
                                **{key: plan['evaluation'][key] for key in
                                ('batch_size', 'steps', 'guidance', 'cfg_min', 'cfg_max', 'clamp', 'seed')}}
            if current['options'] != expected_options:
                raise ValueError('Actual sampling options differ from the registered plan')
            evaluation = json.loads((case / 'evaluation_config.json').read_text())
            for key in ('split', 'cap', 'batch_size', 'seed', 'guidance', 'cfg_min', 'cfg_max', 'steps', 'clamp', 'de'):
                if evaluation[key] != plan['evaluation'][key]:
                    raise ValueError(f'Inference setting differs: {tag}/{key}')
        columns = {value: key for key, value in METRICS.items()}
        comparisons = {}
        for floor, name in ((0, 'floor_0'), (.05, 'floor_0p05')):
            values = []
            for tag in ('reference', 'candidate'):
                case = out / tag / 'zero_floor' / name / 'metrics'
                frame = pd.read_csv(case / 'results.csv').set_index('perturbation').sort_index()
                if not frame.index.is_unique or frame.index.tolist() != expected_labels:
                    raise ValueError('Perturbation library differs')
                array = frame[[columns[key] for key in ORDER]].to_numpy(dtype=float)
                summary = json.loads((case / 'summary.json').read_text())
                registered_metrics = json.loads((out / tag / 'zero_floor/comparison.json').read_text())[name]['metrics']
                np.testing.assert_allclose([summary[k] for k in ORDER], [registered_metrics[k] for k in ORDER], atol=1e-10, rtol=0)
                np.testing.assert_allclose(np.nanmean(array, 0), [summary[k] for k in ORDER], atol=1e-10, rtol=0)
                if floor == 0:
                    original = json.loads((out / tag / 'metrics/summary.json').read_text())
                    np.testing.assert_allclose([summary[k] for k in ORDER], [original[k] for k in ORDER], atol=1e-10, rtol=0)
                values.append(array)
                sources.extend([case / 'results.csv', case / 'summary.json'])
            comparisons[name] = {'floor': floor, **paired_summary(*values)}
        result = {'time': timestamp(), 'step': completed['step'], 'continuation_updates': plan['updates'],
                  'generated_cells_each': len(expected_indices), 'perturbations': len(expected_labels),
                  'bootstrap': 2000, 'seed': 9232, 'comparisons': comparisons,
                  'source_sha256': {str(p): file_sha256(p) for p in sources},
                  'script_sha256': file_sha256(__file__),
                  'bootstrap_helper_sha256': file_sha256('scripts/analyze_inference_review.py'),
                  'automatic_main_switch': False, 'test_access': False,
                  'note': 'One mature-state interpretation trial with the same initial complete RNG state and schedule. '
                          'Compiled training random streams were not separately instrumented. '
                          'Inference indices and private noise seeds match exactly. '
                          'Exploratory perturbation intervals, not independent training seeds or generation-noise repeats.'}
        helper['atomic_json'](out / 'paired_comparison.json', result)
        pd.DataFrame([{'floor': case['floor'], **{k: v for k, v in row.items() if k != 'ci95'},
                       'ci95_low': row['ci95'][0], 'ci95_high': row['ci95'][1]}
                      for case in comparisons.values() for row in case['rows']]).to_csv(out / 'paired_comparison.csv', index=False)
        fig, axes = plt.subplots(3, 3, figsize=(12, 8), layout='constrained')
        for index, (key, ax) in enumerate(zip(ORDER, axes.flat)):
            for line, case in enumerate(comparisons.values()):
                row = case['rows'][index]
                ax.plot(row['ci95'], [line, line], color='#2c6c97')
                ax.plot(row['difference'], line, 'o', color='#2c6c97')
            ax.axvline(0, color='grey', linestyle='--', linewidth=1)
            ax.set_yticks([0, 1], ['cutoff 0', 'cutoff .05'])
            ax.set_title(key + (' (lower is better)' if key in ('MSE', 'MAE') else ''))
            ax.set_xlabel('Batch retained minus original combined dropout')
        fig.suptitle('One-epoch mature continuation; cap16; 95% perturbation intervals')
        fig.savefig(out / 'paired_comparison.png', dpi=180)
        plt.close(fig)
        print(json.dumps({name: case['score_difference'] for name, case in comparisons.items()}, indent=2))


if __name__ == '__main__':
    main()
