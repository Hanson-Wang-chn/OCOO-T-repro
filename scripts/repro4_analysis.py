"""Score retained validation samples and compare identical perturbation libraries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_inference_review import ORDER, resampled_means
from calibrate import write_json_atomic
from diagnose_batch_structure import batch_design, compare_vectors, fit_batch, group_mean
from ocoot.evaluate import METRICS, official_metrics
from ocoot.postprocess import apply_zero_floor, selection_score
from ocoot.runtime import file_sha256
from report import PAPER


def read_json(path):
    return json.loads(Path(path).read_text())


def case_metrics(data, case, out, mse_weight):
    """Evaluate both floors on one completed cap-16 validation generation."""
    case, out = Path(case), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    indices = np.load(case / 'indices.npy')
    np.testing.assert_array_equal(indices, data.evaluation_indices('validation', 16))
    generation = read_json(case / 'generation_state.json')
    if (generation['completed'] != len(indices) or generation['total'] != len(indices)
            or generation['options']['steps'] != 50 or generation['options']['seed'] != 4242):
        raise ValueError(f'Unexpected sampling protocol in {case}')
    raw_name = 'prediction_raw.npy' if generation['options']['clamp'] else 'prediction.npy'
    raw = np.load(case / raw_name, mmap_mode='r')
    if raw.shape != (len(indices), data.x.shape[1]):
        raise ValueError(f'Unexpected sample shape in {case}')
    protocol = {name: file_sha256(case / name) for name in ('indices.npy', raw_name, 'generation_state.json')}
    summary_path = out / 'summary.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary['sample_sha256'] != protocol or summary['mse_weight'] != mse_weight:
            raise ValueError(f'Pilot summary belongs to other samples or scoring settings at {out}')
        for name, digest in summary['metrics_sha256'].items():
            if file_sha256(name) != digest:
                raise ValueError(f'Pilot result changed at {name}')
        return summary

    unseen_ids = np.setdiff1d(data.meta['pert'][indices], data.meta['pert'][data.train])
    unseen = [data.vocab['all_genes'][int(pert)] for pert in unseen_ids]
    results, hashes = {}, {}
    for floor, label in ((0., 'clamp0'), (.05, 'floor_0p05')):
        predicted = apply_zero_floor(raw, floor)
        existing = case / 'metrics'
        if (floor == 0 and generation['options']['clamp'] and (existing / 'summary.json').exists()
                and set(read_json(existing / 'summary.json')) == set(METRICS.values())):
            metrics_dir = existing
            metrics = read_json(existing / 'summary.json')
        else:
            metrics_dir = out / label
            metrics = official_metrics(data, indices, predicted, metrics_dir, de=True)
        per_path = metrics_dir / 'results.csv'
        per = pd.read_csv(per_path).set_index('perturbation')
        if not per.index.is_unique or len(per) != len(np.unique(data.meta['pert'][indices])):
            raise ValueError(f'Incomplete validation perturbation rows in {per_path}')
        np.testing.assert_allclose([float(per[key].mean()) for key in METRICS],
                                   [metrics[name] for name in METRICS.values()], atol=1e-10, rtol=0)
        hashes[str(per_path)] = file_sha256(per_path)
        hashes[str(metrics_dir / 'summary.json')] = file_sha256(metrics_dir / 'summary.json')
        results[label] = {'floor': floor, 'metrics': metrics, 'score': selection_score(metrics, mse_weight),
                          'per_perturbation_csv': str(per_path),
                          'unseen': {name: {display: float(per.loc[name, key]) if pd.notna(per.loc[name, key]) else None
                                           for key, display in METRICS.items()} for name in unseen}}

    summary = {'test_access': False, 'cells': len(indices), 'cap': 16, 'generation': str(case),
               'sampling': generation['options'], 'sample_sha256': protocol, 'metrics_sha256': hashes,
               'mse_weight': mse_weight, 'floors': results}
    write_json_atomic(summary_path, summary)
    return summary


def batch_rms(data, case):
    """Use matched cells and report population-weighted, centered batch RMS."""
    indices = np.load(case / 'indices.npy')
    np.testing.assert_array_equal(indices, data.evaluation_indices('validation', 16))
    actual, prediction = data.x[indices], np.load(case / 'prediction.npy')
    design = batch_design(data.meta['pert'][indices], data.meta['batch'][indices])
    real_mean = group_mean(actual, design.b_inverse, len(design.batches))
    predicted_mean = group_mean(prediction, design.b_inverse, len(design.batches))
    moments = compare_vectors(real_mean, predicted_mean, design.b_counts)
    _, real_fit = fit_batch(actual, design)
    _, pred_fit = fit_batch(prediction, design)
    real_adjusted = real_fit['residualized_batch_component_rms']
    pred_adjusted = pred_fit['residualized_batch_component_rms']
    return {'batches': len(design.batches), 'batch_design_rank': design.rank, 'moments': moments,
            'inflation_ratio': moments['candidate_rms'] / moments['reference_rms'],
            'perturbation_adjusted': {'real_rms': real_adjusted, 'prediction_rms': pred_adjusted,
                                      'inflation_ratio': pred_adjusted / real_adjusted}}


def overfit_distance(probe):
    return float(np.median([abs(row['overfit_ratio'] - 1.) for row in probe['rows'] if row['t'] <= .5]))


def age_cutoff(probe):
    cutoff = probe['val_t_cutoff']
    return probe['rows'][-1]['t'] if cutoff is None else cutoff


def vu_error(probe):
    return float(np.mean([row['vu_mse_val'] for row in probe['rows']]))


def perturbation_effective_rank(checkpoint):
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    table = state['model']['pert_proj.weight']
    singular_values = torch.linalg.svdvals(table)
    probabilities = singular_values / singular_values.sum()
    effective_rank = float(torch.exp(-torch.special.xlogy(probabilities, probabilities).sum()))
    return {'value': effective_rank, 'max': len(singular_values), 'shape': list(table.shape),
            'definition': 'exp(entropy of singular values normalized by their sum)'}


def paired_bootstrap(reference, candidate, *, mse_weight=10., seed=9232):
    columns = {display: key for key, display in METRICS.items()}
    frames = [pd.read_csv(path).set_index('perturbation').sort_index() for path in (reference, candidate)]
    if not frames[0].index.is_unique or not frames[0].index.equals(frames[1].index):
        raise ValueError('Paired bootstrap requires identical unique perturbation labels')
    values = [frame[[columns[name] for name in ORDER]].to_numpy(dtype=float) for frame in frames]
    draws = np.random.default_rng(seed).integers(len(frames[0]), size=(2000, len(frames[0])))
    boot = [resampled_means(value, draws) for value in values]
    means = [np.nanmean(value, axis=0) for value in values]
    delta = boot[1] - boot[0]
    rows = [{'metric': name, 'reference': float(means[0][i]), 'candidate': float(means[1][i]),
             'difference': float(means[1][i] - means[0][i]),
             'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()} for i, name in enumerate(ORDER)]
    pdcorr, mse = [ORDER.index(name) for name in ('PDCorr', 'MSE')]
    pds = [ORDER.index(name) for name in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
    scores = [selection_score(dict(zip(ORDER, mean)), mse_weight) for mean in means]
    score_delta = delta[:, pdcorr] + delta[:, pds].mean(axis=1) - mse_weight * delta[:, mse]
    return {'reference': str(reference), 'candidate': str(candidate), 'draws': len(draws), 'seed': seed,
            'perturbations': len(frames[0]), 'metrics': rows,
            'score_delta': {'difference': float(scores[1] - scores[0]),
                            'ci95': np.nanquantile(score_delta, [.025, .975]).tolist()},
            'source_sha256': {str(path): file_sha256(path) for path in (reference, candidate)}}


def verify_training(run, config, evaluated_epochs):
    finished = read_json(run / 'finished.json')
    state = torch.load(run / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    step = finished['step']
    if (state['config'] != config or state['step'] != step or state['pending_generation_epoch'] is not None
            or set(state['ema']) != {'0.9996', '0.9999'} or len(state['rng_by_rank']) != 4
            or finished['planned_epochs'] != 80 or finished['epoch'] < 35 or step != finished['epoch'] * 1118):
        raise ValueError('Main training completion metadata differs from the registered run')
    tensors = state['model']
    parameter_names = set(tensors) - {'esm', 'gene_indices', 'missing', 'position'}
    for name, parameter in tensors.items():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f'Nonfinite model tensor {name}')
    for ema in state['ema'].values():
        if set(ema) != parameter_names or any(ema[name].shape != tensors[name].shape for name in ema):
            raise ValueError('EMA does not match model parameter schema')
        if any(not torch.isfinite(parameter).all() for parameter in ema.values()):
            raise FloatingPointError('Nonfinite EMA tensor')
    optimizer = state['optimizer']['state']
    if len(optimizer) != len(parameter_names):
        raise ValueError('Missing AdamW state')
    for entry in optimizer.values():
        if int(entry['step']) != step or not all(torch.isfinite(entry[key]).all() for key in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Invalid AdamW update history or tensor')
    if any(set(rng) != {'python', 'numpy', 'torch', 'cuda'} or len(rng['cuda']) != 1 for rng in state['rng_by_rank']):
        raise ValueError('Incomplete per-rank RNG history')
    best_epoch = state['best_generation_epoch']
    del state, tensors, optimizer

    expected_epochs = list(range(5, int(finished['epoch']) + 1, 5))
    if sorted(evaluated_epochs) != expected_epochs or best_epoch not in expected_epochs:
        raise ValueError('Main training snapshots do not cover the registered validation cadence')
    files = [run / name for name in ('config.yaml', 'last.pt', 'best_generation.pt',
                                    'best_denoise.pt', 'finished.json', 'training_execution.json', 'train_indices.npy')]
    snapshots = [run / f'validation_epoch_{epoch:03d}/model.pt' for epoch in expected_epochs]
    for path, epoch in zip(snapshots, expected_epochs, strict=True):
        snapshot = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if snapshot['step'] != epoch * 1118 or snapshot.get('weights') != 'ema_0.9996':
            raise ValueError(f'Snapshot step or EMA differs at {path}')
        del snapshot
    files.extend(snapshots)
    files.extend(run / f'velocity_probe/epoch_{epoch:03d}/summary.json' for epoch in expected_epochs if epoch % 10 == 0)
    proof = {'finished': finished, 'best_generation_epoch': best_epoch, 'rng_ranks': 4,
             'optimizer_slots': len(parameter_names), 'ema_decays': [.9999, .9996],
             'snapshot_epochs': expected_epochs, 'test_access': False,
             'training_event_log': {'bytes': (run / 'events.jsonl').stat().st_size,
                                    'sha256': file_sha256(run / 'events.jsonl')},
             'source_sha256': read_json(run / 'source_manifest_step_0.json'),
             'checkpoint_output_sha256': {str(path): file_sha256(path) for path in files}}
    write_json_atomic(run / 'training_completion_verification.json', proof)
    return proof


def verify_and_compare_test(run):
    selected = run / 'test_full_postprocess'
    clamp0 = run / 'test_full_clamp0'
    completion = read_json(selected / 'postprocess_test_completed.json')
    protocol = completion['protocol']
    if (protocol['test_cells'] != 26878 or protocol['test_perturbations'] != 380
            or not all(protocol['validation_checks'].values()) or completion['selection_changed_after_test']
            or completion['new_postprocess_generated_samples'] != 0):
        raise ValueError('Frozen evaluation completion differs from the registered protocol')
    for path, digest in completion['output_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Completed evaluation artifact changed at {path}')
    for folder in (selected, clamp0):
        counts = read_json(folder / 'metrics/population_counts.json')
        if counts != {'real_perturbed_cells': 26878, 'generated_cells': 26878, 'shared_real_controls': 4976}:
            raise ValueError(f'Unexpected full-test populations at {folder}')
    for folder, out in ((selected, run / 'results'), (clamp0, run / 'results_clamp0')):
        current = folder / 'metrics/results.csv'
        comparisons = {}
        rows = []
        for prior in (2, 3):
            previous = Path(f'runs/repro{prior}_main') / folder.name / 'metrics/results.csv'
            pair = paired_bootstrap(previous, current)
            comparisons[f'repro{prior}'] = pair
            for row in pair['metrics']:
                rows.append({**row, 'reference_run': f'repro{prior}', 'ci_low': row['ci95'][0], 'ci_high': row['ci95'][1]})
        pd.DataFrame(rows).drop(columns='ci95').to_csv(out / 'paired_previous_rounds.csv', index=False)
        # Table 3 supplies point estimates. Their difference intervals use the
        # bootstrap for this run, with the published point held fixed.
        table3 = pd.read_csv(out / 'benchmark_comparison.csv').set_index('metric')
        paper = {name: {'table3': value, 'difference': float(table3.loc[name, 'reproduction'] - value),
                        'ci95': [float(table3.loc[name, col] - value) for col in ('CI_low', 'CI_high')]}
                 for name, value in PAPER.items()}
        write_json_atomic(out / 'paired_previous_rounds.json', {
            'paired': comparisons, 'paper_table3_fixed_estimates': paper,
            'paper_table3_pairing': 'The published table has aggregate values. Perturbation pairs are available for runs 2 and 3.'})
    proof = {'protocol': protocol, 'completion_sha256': file_sha256(selected / 'postprocess_test_completed.json'),
             'raw_prediction_sha256': file_sha256(clamp0 / 'prediction_raw.npy'),
             'metrics': completion['metrics'], 'baseline_clamp0_metrics': completion['baseline_metrics'],
             'reports': [str(run / folder / 'REPORT.md') for folder in ('results', 'results_clamp0')],
             'paired_comparisons': [str(run / folder / 'paired_previous_rounds.json')
                                    for folder in ('results', 'results_clamp0')],
             'additional_test_samples_for_floor': 0}
    write_json_atomic(run / 'test_completion_verification.json', proof)
    return proof
