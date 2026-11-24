"""Audit and compare matched validation pilots without changing any training.

The raw-weight subset pilots select implementation details, not final paper
inference models. Bootstrap intervals describe held-out perturbations only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from ocoot.evaluate import METRICS
from ocoot.model import ModelConfig
from ocoot.runtime import file_sha256, timestamp


SCORE_METRICS = ['PDCorr', 'PDS_L1', 'PDS_L2', 'PDS_cos']
TRAIN_FIELDS = ['epochs', 'warmup_epochs', 'effective_batch', 'micro_batch', 'lr',
                'betas', 'weight_decay', 'p_mean', 'p_std', 'ema_decays',
                'train_subset', 'val_cells', 'val_generation_batch',
                'val_generation_cap', 'val_generation_steps']


def bootstrap_scores(values, draws):
    selected = values[draws]
    counts = np.isfinite(selected).sum(axis=1)
    means = np.divide(np.nansum(selected, axis=1), counts,
                      out=np.full(counts.shape, np.nan), where=counts > 0)
    return means[:, 0] + means[:, 1:].mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparisons', nargs='+', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--reference', default='sigma_1.0_cellinit_1.0')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=9232)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('bootstrap must be positive')
    records = {}
    sources = {}
    for value in args.comparisons:
        path = Path(value)
        addition = json.loads(path.read_text())
        if records.keys() & addition.keys():
            raise ValueError('Duplicate trials across comparison inputs')
        records.update(addition)
        sources[str(path)] = file_sha256(path)
    if args.reference not in records:
        raise ValueError('The reference must be included')
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    columns = {display: key for key, display in METRICS.items()}
    rows, arrays, budgets = [], {}, {}
    common = training_indices = prediction_indices = labels = None
    for tag, record in records.items():
        run = Path('runs') / ('pilot_' + tag)
        cfg = yaml.safe_load((run / 'config.yaml').read_text())
        if not cfg['training'].get('train_subset'):
            raise ValueError(f'Expected a subset pilot: {tag}')
        model_cfg = asdict(ModelConfig(**cfg['model']))
        factors = {'noise_std': cfg['training']['noise_std'],
                   'time_scale': model_cfg.pop('time_scale'),
                   'cell_init_std': model_cfg.pop('cell_init_std')}
        train = cfg['training']
        budget = {key: train[key] for key in TRAIN_FIELDS}
        case = run / f'validation_epoch_{train["epochs"]:03d}'
        generation = json.loads((case / 'generation_state.json').read_text())
        finished = json.loads((run / 'finished.json').read_text())
        subset = np.load(run / 'train_indices.npy')
        idx = np.load(case / 'indices.npy')
        expected_steps = len(subset) // train['effective_batch'] * train['epochs']
        if finished['step'] != expected_steps or generation['completed'] != len(idx):
            raise ValueError(f'Incomplete or unequal training budget: {tag}')
        options = dict(generation['options'])
        if options.pop('noise_std') != factors['noise_std']:
            raise ValueError(f'Sampling noise mismatch: {tag}')
        signature = {'model_other_parameters': model_cfg, 'training': budget,
                     'sampling_other_parameters': options,
                     'seed': cfg['seed'], 'data_dir': cfg['data_dir']}
        if common is None:
            common, training_indices, prediction_indices = signature, subset, idx
        else:
            if signature != common:
                raise ValueError(f'Unmatched pilot configuration beyond the three factors: {tag}')
            np.testing.assert_array_equal(subset, training_indices)
            np.testing.assert_array_equal(idx, prediction_indices)
        # mmap avoids copying the large weight tensors merely to check metadata.
        checkpoint_path = case / 'model.pt'
        if not checkpoint_path.exists():
            # Earlier pilots predate the per-epoch model copy. Their immutable
            # best-generation file records the same completed raw checkpoint.
            checkpoint_path = run / 'best_generation.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        if checkpoint['step'] != expected_steps or checkpoint.get('weights') != 'raw_pilot_only':
            raise ValueError(f'Expected the completed raw diagnostic weights: {tag}')
        if checkpoint['config'] != cfg:
            raise ValueError(f'Checkpoint configuration differs from the audited pilot: {tag}')
        for metric in SCORE_METRICS:
            np.testing.assert_allclose(checkpoint['metrics'][metric], record[metric], atol=1e-10, rtol=0)
        del checkpoint
        budgets[tag] = {'optimizer_steps': expected_steps, 'cells': len(subset),
                        'checkpoint': str(checkpoint_path), **factors}
        for path in [run / 'config.yaml', run / 'train_indices.npy',
                     case / 'indices.npy', case / 'generation_state.json', checkpoint_path]:
            sources[str(path)] = file_sha256(path)
        for postprocess in ['raw', 'clamped']:
            metrics = record if postprocess == 'raw' else record['clamped']
            metric_dir = case / ('metrics' if postprocess == 'raw' else
                                 'metrics_clamped_de' if 'DEPrec' in metrics else 'metrics_clamped')
            csv_path = metric_dir / 'results.csv'
            frame = pd.read_csv(csv_path).set_index('perturbation').sort_index()
            if not frame.index.is_unique:
                raise ValueError(f'Duplicate perturbation rows: {tag}')
            if labels is None:
                labels = frame.index
            elif not frame.index.equals(labels):
                raise ValueError(f'Unmatched perturbation library: {tag}')
            for metric in METRICS.values():
                if metric in metrics:
                    np.testing.assert_allclose(frame[columns[metric]].mean(), metrics[metric],
                                               atol=1e-10, rtol=0)
            values = frame[[columns[key] for key in SCORE_METRICS]].to_numpy(dtype=np.float64)
            score = float(np.nanmean(values, axis=0)[0] + np.nanmean(values, axis=0)[1:].mean())
            arrays[(tag, postprocess)] = values
            rows.append({'trial': tag, 'postprocess': postprocess, **factors,
                         **{key: metrics[key] for key in METRICS.values() if key in metrics},
                         'score': score})
            sources[str(csv_path)] = file_sha256(csv_path)
    draws = np.random.default_rng(args.seed).integers(0, len(labels), (args.bootstrap, len(labels)))
    boot = {key: bootstrap_scores(values, draws) for key, values in arrays.items()}
    for row in rows:
        key = (row['trial'], row['postprocess'])
        reference_key = (args.reference, row['postprocess'])
        delta = boot[key] - boot[reference_key]
        finite = delta[np.isfinite(delta)]
        lower, upper = np.quantile(finite, [.025, .975])
        reference_score = next(x['score'] for x in rows if
                               x['trial'] == args.reference and x['postprocess'] == row['postprocess'])
        row.update(score_delta_vs_reference=row['score'] - reference_score,
                   delta_ci95_low=float(lower), delta_ci95_high=float(upper),
                   valid_bootstrap_draws=len(finite))
    table = pd.DataFrame(rows).sort_values(['postprocess', 'score'], ascending=[True, False])
    table.to_csv(out / 'pilot_metrics.csv', index=False)
    best = {post: max((row for row in rows if row['postprocess'] == post), key=lambda r: r['score'])['trial']
            for post in ['raw', 'clamped']}
    combinations = {(row['noise_std'], row['time_scale'], row['cell_init_std']) for row in rows}
    levels = {key: sorted({row[key] for row in rows}) for key in
              ['noise_std', 'time_scale', 'cell_init_std']}
    result = {'time': timestamp(), 'reference': args.reference, 'budgets': budgets,
              'matched_protocol': common, 'perturbations': len(labels),
              'factor_levels': levels,
              'complete_2x2x2_grid': len(combinations) == 8 and all(len(x) == 2 for x in levels.values()),
              'best_among_supplied_trials': best, 'rows': rows,
              'bootstrap_draws': args.bootstrap, 'seed': args.seed,
              'source_sha256': sources, 'script_sha256': file_sha256(__file__),
              'note': 'Matched validation subset pilots only, with raw diagnostic weights. '
                      'Intervals bootstrap perturbations in the fixed comparison library; '
                      'they do not represent independent training seeds or generation noise. '
                      'This analysis does not replace the full-training model automatically.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    tags = sorted(records, key=lambda name: (budgets[name]['cell_init_std'],
                                           budgets[name]['time_scale'], budgets[name]['noise_std']))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .65 * len(tags))), layout='constrained')
    for j, post in enumerate(['raw', 'clamped']):
        color = ['#858d99', '#3478a6'][j]
        for i, tag in enumerate(tags):
            row = next(x for x in rows if x['trial'] == tag and x['postprocess'] == post)
            y = i + (j - .5) * .2
            axes[0].plot(row['score'], y, 'o', color=color, label=post if i == 0 else None)
            axes[1].plot([row['delta_ci95_low'], row['delta_ci95_high']], [y, y], color=color)
            axes[1].plot(row['score_delta_vs_reference'], y, 'o', color=color)
    names = [f'sigma={budgets[tag]["noise_std"]:g}, t={budgets[tag]["time_scale"]:g}, c={budgets[tag]["cell_init_std"]:g}' for tag in tags]
    axes[0].set_yticks(range(len(tags)), names)
    axes[0].set_xlabel('PDCorr + mean of three PDS')
    axes[0].legend()
    axes[1].set_yticks(range(len(tags)), [])
    axes[1].axvline(0, color='#858d99', linestyle='--')
    axes[1].set_xlabel('Paired score difference from original selected pilot (95% interval)')
    fig.suptitle('Validation implementation pilots; raw weights; matched training budget')
    fig.savefig(out / 'pilot_scores.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'trials': len(records), 'best': best,
                      'grid_complete': result['complete_2x2x2_grid'], 'outdir': str(out)}, indent=2))


if __name__ == '__main__':
    main()
