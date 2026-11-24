"""Controlled CPU metric-sensitivity experiment on measured validation cells.

Only exact zero entries receive rectified tiny Gaussian jitter; every measured
positive stays bitwise unchanged. These are synthetic diagnostics, NOT model
predictions, performance estimates, or a postprocessing proposal.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics
from ocoot.runtime import exclusive_run, file_sha256, log_event, readme_event, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    out = Path(args.outdir)
    with exclusive_run(out, 'zero_tie_diagnostic'):
        torch.set_num_threads(1)
        plan = json.loads((out / 'plan.json').read_text())
        if file_sha256(__file__) != plan['script_sha256']:
            raise ValueError('Implementation changed from registered plan')
        data = BenchmarkData()
        if file_sha256(data.path / 'metadata.npz') != plan['metadata_sha256']:
            raise ValueError('Data metadata changed')
        indices = data.val
        np.save(out / 'indices.npy', indices)
        real = np.asarray(data.x[indices]).copy()
        zero = real == 0
        noise = np.random.default_rng(plan['seed']).standard_normal(int(zero.sum()), dtype=np.float32)
        real_hash = hashlib.sha256(real.tobytes()).hexdigest()
        records = []
        for sigma in plan['sigmas']:
            folder = out / ('identity' if sigma == 0 else f'zero_jitter_{sigma:g}')
            folder.mkdir(exist_ok=True)
            completed = folder / 'completed.json'
            if completed.exists():
                records.append(json.loads(completed.read_text()))
                continue
            pred = real.copy()
            if sigma:
                pred[zero] = np.maximum(noise * sigma, 0)
            np.testing.assert_array_equal(pred[~zero], real[~zero])
            if not np.isfinite(pred).all() or np.min(pred) < 0:
                raise ValueError('Invalid synthetic sensitivity sample')
            metrics = official_metrics(data, indices, pred, folder / 'metrics', de=True)
            frames = [pd.read_csv(folder / 'metrics' / name) for name in ('real_de.csv', 'pred_de.csv')]
            sets = []
            labels = sorted(frames[0]['target'].unique())
            for frame in frames:
                significant = frame[(frame.fdr < .05) & (frame.target != frame.feature)]
                sets.append({name: set(group.feature) for name, group in significant.groupby('target')})
            counts = [{'perturbation': p, 'real_significant': len(sets[0].get(p, set())),
                       'synthetic_significant': len(sets[1].get(p, set())),
                       'shared_significant': len(sets[0].get(p, set()) & sets[1].get(p, set()))}
                      for p in labels]
            pd.DataFrame(counts).to_csv(folder / 'significant_counts.csv', index=False)
            record = {'sigma': sigma, 'metrics': metrics,
                      'zero_fraction': float(np.mean(pred == 0)),
                      'positive_entries_unchanged': True,
                      'mean_abs_change': float(np.abs(pred - real).mean(dtype=np.float64)),
                      'source_real_sha256': real_hash,
                      'synthetic_sha256': hashlib.sha256(pred.tobytes()).hexdigest(),
                      'median_real_significant': float(np.median([c['real_significant'] for c in counts])),
                      'median_synthetic_significant': float(np.median([c['synthetic_significant'] for c in counts])),
                      'note': 'Measured-data zero-tie sensitivity, NOT OCOO-T predictions or a model score.'}
            if sigma == 0 and (abs(metrics['MSE']) > 1e-12 or abs(metrics['PDCorr'] - 1) > 1e-6):
                raise ValueError('Identity reference failed')
            temporary = completed.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
            temporary.replace(completed)
            records.append(record)
            log_event(out / 'events.jsonl', {'event': 'zero_tie_sensitivity_case', **record})
        summary = {'time': timestamp(), 'plan_sha256': file_sha256(out / 'plan.json'),
                   'real_validation_cells': len(indices), 'perturbations': len(np.unique(data.meta['pert'][indices])),
                   'real_zero_fraction': float(zero.mean()), 'seed': plan['seed'],
                   'records': records, 'model_predictions_used': False, 'test_access': False,
                   'note': 'Controls stay exactly real, as in the paper protocol. Identity and its zero-only perturbations '
                           'share the same measured cells; this is a controlled metric sensitivity check, not independent generation.'}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        readme_event(f'CPU近零秩敏感性检查完成：仅对实测验证零值加微扰，所有正表达保持原样；'
                     f'这不是模型预测或成绩。结果 `{out}/summary.json`，主训练与指标定义均未修改。')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
