"""Compare identical denoising interventions across two frozen checkpoints.

Cells, noising times, and interventions must match exactly. The paired cluster
bootstrap describes these cells' denoising dependence, never generation quality.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ocoot.runtime import file_sha256, timestamp


def load_arrays(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--outdir')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=62341)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('bootstrap must be positive')
    reference, candidate = Path(args.reference), Path(args.candidate)
    out = Path(args.outdir) if args.outdir else candidate / 'paired_progress'
    out.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((p / 'summary.json').read_text()) for p in (reference, candidate)]
    old_protocol, new_protocol = [s['protocol'] for s in summaries]
    for key in ('cells', 'batch_size', 'seed', 'cases', 'ema_decay', 'model_sha256',
                'flow_sha256', 'data_code_sha256', 'metadata_sha256', 'vocab_sha256'):
        if old_protocol[key] != new_protocol[key]:
            raise ValueError(f'Intervention protocols differ: {key}')
    checkpoints = [torch.load(p / 'source_snapshot.pt', map_location='cpu', mmap=True, weights_only=False)
                   for p in (reference, candidate)]
    if checkpoints[0]['config'] != checkpoints[1]['config']:
        raise ValueError('Training configurations differ')
    if checkpoints[0]['step'] >= checkpoints[1]['step']:
        raise ValueError('Candidate must be a later checkpoint')
    if set(summaries[0]['weights']) != set(summaries[1]['weights']):
        raise ValueError('Weight variants differ')
    records, provenance = [], {}
    for weights in summaries[0]['weights']:
        for split in summaries[0]['weights'][weights]:
            paths = [p / f'{weights}_{split}.npz' for p in (reference, candidate)]
            before, after = map(load_arrays, paths)
            for key in ('indices', 'pert', 'shuffled_pert', 'batch', 't', 'shuffled_t'):
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            provenance[f'{weights}/{split}'] = dict(zip(('reference', 'candidate'), map(file_sha256, paths)))
            _, inverse = np.unique(before['pert'], return_inverse=True)
            counts = np.bincount(inverse)
            draws = np.random.default_rng(args.seed).integers(0, len(counts), (args.bootstrap, len(counts)))
            old_losses = before['velocity_losses'].astype(np.float64)
            new_losses = after['velocity_losses'].astype(np.float64)
            if not np.isfinite(old_losses).all() or not np.isfinite(new_losses).all():
                raise ValueError('Nonfinite intervention losses')
            # Retain cell weighting from diagnose_conditions.summarize and
            # resample perturbations with all of their paired cells together.
            for space, factor in [('velocity', np.ones(len(inverse))),
                                  ('endpoint', (1 - before['t'].astype(np.float64)) ** 2)]:
                old = old_losses * factor[:, None]
                new = new_losses * factor[:, None]
                for i, case in enumerate(old_protocol['cases']):
                    first = old[:, i] if i == 0 else old[:, i] - old[:, 0]
                    second = new[:, i] if i == 0 else new[:, i] - new[:, 0]
                    delta = second - first
                    totals = np.bincount(inverse, weights=delta, minlength=len(counts))
                    resampled = totals[draws].sum(1) / counts[draws].sum(1)
                    low, high = np.quantile(resampled, [.025, .975])
                    records.append({'weights': weights, 'split': split, 'space': space, 'case': case,
                                    'quantity': 'matched_loss' if i == 0 else 'intervention_minus_matched_loss',
                                    'cells': len(delta), 'perturbations': len(counts),
                                    'reference': float(first.mean()), 'candidate': float(second.mean()),
                                    'difference': float(delta.mean()), 'ci95_low': float(low), 'ci95_high': float(high)})
    result = {'time': timestamp(), 'reference': str(reference), 'candidate': str(candidate),
              'reference_step': old_protocol['step'], 'candidate_step': new_protocol['step'],
              'reference_checkpoint_sha256': old_protocol['checkpoint_sha256'],
              'candidate_checkpoint_sha256': new_protocol['checkpoint_sha256'],
              'bootstrap': args.bootstrap, 'seed': args.seed, 'array_sha256': provenance,
              'script_sha256': file_sha256(__file__),
              'note': 'Exact paired cells/times/interventions. Matched loss reductions indicate better denoising; '
                      'positive intervention-minus-matched changes indicate stronger conditional dependence. '
                      'Cell-weighted, perturbation-cluster bootstrap; not independent training seeds, generation metrics, or test results.',
              'records': records}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps([r for r in records if r['space'] == 'velocity' and r['case'] == 'shuffled_pert'], indent=2))


if __name__ == '__main__':
    main()
