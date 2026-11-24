"""Compare one controlled inference change using immutable validation caches."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from ocoot.data import BenchmarkData
from ocoot.evaluate import METRICS
from ocoot.runtime import exclusive_run, file_sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--factor', choices=('guidance', 'cfg_interval'), default='guidance')
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    folders = [Path(args.reference), Path(args.candidate)]
    if folders[0].resolve() == folders[1].resolve():
        raise ValueError('A controlled comparison requires two different caches')
    configs = [json.loads((p / 'evaluation_config.json').read_text()) for p in folders]
    if any(c['split'] != 'validation' or not c['de'] or not c['clamp'] for c in configs):
        raise ValueError('Requires completed nonnegative nine-metric validation cases')
    allowed = ('guidance',) if args.factor == 'guidance' else ('cfg_min', 'cfg_max')
    normalized = [{key: value for key, value in c.items() if key not in allowed} for c in configs]
    if normalized[0] != normalized[1]:
        raise ValueError('Cases differ beyond the registered inference factor')
    if all(configs[0][key] == configs[1][key] for key in allowed):
        raise ValueError('The registered factor is unchanged')
    if file_sha256(configs[0]['checkpoint']) != configs[0]['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    indices = [np.load(p / 'indices.npy') for p in folders]
    np.testing.assert_array_equal(*indices)
    data = BenchmarkData()
    if not np.all(data.meta['split'][indices[0]] == 1):
        raise ValueError('Cache contains cells outside validation')
    markers = [json.loads((p / 'generation_state.json').read_text()) for p in folders]
    if any(m['completed'] != m['total'] or m['total'] != len(indices[0]) for m in markers):
        raise ValueError('Incomplete generation cache')
    options = [{key: value for key, value in m['options'].items() if key not in allowed} for m in markers]
    if options[0] != options[1]:
        raise ValueError('Generation options differ beyond the registered factor')
    helpers = runpy.run_path('scripts/analyze_inference_review.py')
    order = helpers['ORDER']
    columns = {value: key for key, value in METRICS.items()}
    frames = [pd.read_csv(p / 'metrics/results.csv').set_index('perturbation').sort_index() for p in folders]
    if not frames[0].index.equals(frames[1].index) or not frames[0].index.is_unique:
        raise ValueError('Perturbation libraries differ')
    values = [frame[[columns[key] for key in order]].to_numpy(dtype=float) for frame in frames]
    means = [np.nanmean(value, axis=0) for value in values]
    for folder, mean in zip(folders, means):
        summary = json.loads((folder / 'metrics/summary.json').read_text())
        np.testing.assert_allclose(mean, [summary[key] for key in order], rtol=0, atol=1e-10)
    draws = np.random.default_rng(9232).integers(0, len(frames[0]), (2000, len(frames[0])))
    boot = [helpers['resampled_means'](value, draws) for value in values]
    delta = boot[1] - boot[0]
    rows = [{'metric': key, 'reference': float(means[0][i]), 'candidate': float(means[1][i]),
             'difference': float(means[1][i] - means[0][i]),
             'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()} for i, key in enumerate(order)]
    pd_id = order.index('PDCorr')
    pds = [order.index(key) for key in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
    scores = [float(mean[pd_id] + mean[pds].mean()) for mean in means]
    score_delta = delta[:, pd_id] + delta[:, pds].mean(1)
    sources = [p / name for p in folders for name in ('evaluation_config.json', 'indices.npy',
               'generation_state.json', 'prediction.npy', 'prediction_raw.npy', 'metrics/summary.json', 'metrics/results.csv')]
    hashes = {str(p): file_sha256(p) for p in sources}
    result = {'time': timestamp(), 'reference': str(folders[0]), 'candidate': str(folders[1]),
              'factor': args.factor, 'reference_options': {key: configs[0][key] for key in allowed},
              'candidate_options': {key: configs[1][key] for key in allowed},
              'checkpoint_sha256': configs[0]['checkpoint_sha256'], 'step': configs[0]['step'],
              'cells': len(indices[0]), 'perturbations': len(frames[0]), 'rows': rows,
              'reference_score': scores[0], 'candidate_score': scores[1],
              'score_difference': {'value': scores[1] - scores[0],
                                   'ci95': np.nanquantile(score_delta, [.025, .975]).tolist()},
              'source_sha256': hashes, 'metadata_sha256': file_sha256(data.path / 'metadata.npz'),
              'script_sha256': file_sha256(__file__), 'bootstrap': 2000, 'seed': 9232,
              'note': 'One controlled validation-only inference factor. Matched indices/noise, fixed reference library. '
                      'Intervals do not measure independent training seeds or generation noise. '
                      'No automatic formal-setting selection and no test access.'}
    out = Path(args.outdir)
    fingerprint = {key: result[key] for key in ('reference', 'candidate', 'factor', 'source_sha256',
                                               'metadata_sha256', 'script_sha256', 'bootstrap', 'seed')}
    with exclusive_run(out, 'comparison'):
        protocol = out / 'comparison_config.json'
        if protocol.exists() and json.loads(protocol.read_text()) != fingerprint:
            raise ValueError('Output is pinned to a different comparison')
        temporary = protocol.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(fingerprint, indent=2)); temporary.replace(protocol)
        temporary = out / 'comparison.json.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)); temporary.replace(out / 'comparison.json')
        pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    print(json.dumps({'factor': args.factor, 'score_difference': result['score_difference']}, indent=2))


if __name__ == '__main__':
    main()
