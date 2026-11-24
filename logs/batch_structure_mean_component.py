"""Post-hoc descriptor linking within-perturbation effects to group mean error.

No corrected predictions or official benchmark scores are produced.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('runs/main_v1/batch_structure_epoch025')
plan = json.loads((root / 'plan.json').read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name, digest in plan['source_sha256'].items():
    assert sha(name) == digest
assert json.loads((root / 'completed.json').read_text())['summary_sha256'] == sha(root / 'summary.json')
meta = dict(np.load(Path(plan['data_dir']) / 'metadata.npz'))
idx = np.load(plan['indices_file'])
np.testing.assert_array_equal(np.sort(idx), np.flatnonzero(meta['split'] == 1))
x = np.load(plan['expression_file']['path'], mmap_mode='r')
error = np.asarray(x[idx], dtype=np.float64) - np.load(plan['clamped_prediction_file'])
perturbations, pi = np.unique(meta['pert'][idx], return_inverse=True)
counts = np.bincount(pi)
mean_error = np.zeros((len(perturbations), error.shape[1]), dtype=np.float64)
np.add.at(mean_error, pi, error)
mean_error /= counts[:, None]
effects = dict(np.load(root / 'batch_effects.npz'))
bi = np.searchsorted(effects['batch_ids'], meta['batch'][idx])
np.testing.assert_array_equal(effects['batch_ids'][bi], meta['batch'][idx])
np.testing.assert_allclose(effects['clamp0_error'], effects['true'] - effects['clamp0_prediction'], atol=1e-11, rtol=0)
np.testing.assert_allclose(np.average(effects['clamp0_error'], axis=0,
    weights=effects['validation_counts']), 0, atol=1e-12, rtol=0)
batch_composition = np.zeros_like(mean_error)
np.add.at(batch_composition, pi, effects['clamp0_error'][bi])
batch_composition /= counts[:, None]
common = mean_error.mean(0)
energy = float(np.square(mean_error).mean())
composition_energy = float(np.square(batch_composition).mean())
common_energy = float(np.square(common).mean())
result = {'time': datetime.now().astimezone().isoformat(timespec='seconds'),
          'post_hoc_descriptive_supplement': True,
          'purpose': 'Relate the registered within-perturbation batch decomposition to pseudobulk mean errors; prevent treating a cell-variance decomposition as a benchmark error attribution.',
          'all_gene_perturbation_mean_error_rms': float(np.sqrt(energy)),
          'centered_batch_composition_mean_component_rms': float(np.sqrt(composition_energy)),
          'batch_composition_component_squared_norm_over_mean_error_squared_norm': composition_energy / energy,
          'common_across_perturbations_gene_bias_rms': float(np.sqrt(common_energy)),
          'common_gene_bias_squared_norm_fraction': common_energy / energy,
          'source_sha256': {'plan.json': sha(root / 'plan.json'), 'summary.json': sha(root / 'summary.json'),
                            'batch_effects.npz': sha(root / 'batch_effects.npz'), 'script': sha(__file__)},
          'note': 'All2000 genes, macro perturbation means; these descriptors are not official metric values. Batch effects have a cell-count-weighted zero-mean constraint. Ratios are descriptive norms, not causal explained-error fractions. Coefficients use validation truth and must not be adopted as a test-time correction.',
          'cpu_only': True, 'test_access': False, 'corrected_prediction_produced': False}
output = root / 'mean_component_supplement.json'
assert not output.exists(), 'Keep the completed descriptor immutable'
output.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(result))
