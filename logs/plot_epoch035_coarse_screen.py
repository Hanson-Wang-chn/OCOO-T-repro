"""Plot the six completed E35 validation screen cases without changing metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from ocoot.runtime import file_sha256, timestamp


ORDER = ('PDCorr', 'DEOver', 'DEPrec', 'DirAgr', 'PDS_L1', 'PDS_L2', 'PDS_cos', 'MSE', 'MAE')
root = Path('runs/main_v1/interim_epoch035_calibration')
calibration = root / 'calibration'
out = root / 'coarse_review'
out.mkdir(exist_ok=True)
completed = out / 'plot_completed.json'
if completed.exists():
    proof = json.loads(completed.read_text())
    for path, digest in proof['output_sha256'].items():
        assert file_sha256(path) == digest
    print(json.dumps(proof))
    raise SystemExit(0)

source = calibration / 'all_results.json'
records = json.loads(source.read_text())
screens = [r for r in records if r['tag'].startswith('screen')]
assert [r['tag'] for r in screens] == ['screen_00', 'screen_01', 'screen_02', 'screen_03', 'screen_04', 'screen_full_interval']
configuration = json.loads((calibration / 'calibration_config.json').read_text())
assert configuration['step'] == 39130 and configuration['weights'] == 'ema_0.9996'
assert configuration['batch_size'] == 32 and configuration['screen_cap'] == 16
checksums = {}
reference_indices = None
for record in screens:
    assert record['cap'] == 16 and record['clamp'] and set(record['metrics']) == set(ORDER)
    directory = Path(record['cache_provenance']['directory'])
    assert directory.resolve() == (calibration / record['tag']).resolve()
    assert record['cache_provenance']['split'] == 'validation'
    for name, digest in record['cache_provenance']['files'].items():
        path = directory / name
        assert file_sha256(path) == digest
        checksums[str(path)] = digest
    indices = np.load(directory / 'indices.npy')
    if reference_indices is None:
        reference_indices = indices
    np.testing.assert_array_equal(indices, reference_indices)
    state = json.loads((directory / 'generation_state.json').read_text())
    assert state['completed'] == state['total'] == len(indices) == 960
    assert state['options'] == {'noise_std': 1., 'batch_size': 32, 'steps': 50,
        'guidance': record['guidance'], 'cfg_min': record['cfg_min'], 'cfg_max': record['cfg_max'],
        'clamp': False, 'seed': 4242}
    summary = directory / 'metrics_clamped/summary.json'
    assert json.loads(summary.read_text()) == record['metrics']
    checksums[str(summary)] = file_sha256(summary)

# The original all_results file later appends full refinements. Keep the exact
# six-case content used for this figure in an immutable, separate snapshot.
snapshot = out / 'screen_results_snapshot.json'
if snapshot.exists():
    assert json.loads(snapshot.read_text()) == screens
else:
    snapshot.write_text(json.dumps(screens, indent=2, allow_nan=False))
eligible = [r for r in screens if r['guidance'] is not None]
settings = [{key: r[key] for key in ('tag', 'guidance', 'cfg_min', 'cfg_max', 'score')}
            for r in sorted(eligible, key=lambda r: r['score'], reverse=True)[:2]]
selection = out / 'full_settings_from_screen.json'
selection.write_text(json.dumps({'time': timestamp(), 'settings_in_original_algorithm_order': settings,
    'policy': 'Unchanged original positive-CFG primary-score ranking; ordinary field excluded',
    'full_refinement_cells_each': 4825, 'full_metrics_pending': True, 'final_protocol_adopted': False,
    'screen_snapshot_sha256': file_sha256(snapshot), 'test_access': False}, indent=2))

labels = ['cond', '2/3', '1', '1.5', '2', '2\nfull']
fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), layout='constrained')
for ax, metric in zip(axes.flat, ORDER):
    values = [r['metrics'][metric] for r in screens]
    ax.scatter([0], values[:1], color='#596b88', s=32)
    ax.plot(range(1, 5), values[1:5], 'o-', color='#287f94', markersize=4, linewidth=1.2)
    ax.scatter([5], values[5:], color='#bd722d', s=36)
    ax.set_xticks(range(6), labels)
    ax.set_xlim(-.4, 5.4)
    ax.set_title(metric + (' (lower is better)' if metric in ('MSE', 'MAE') else ''))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.grid(axis='y', alpha=.18)
    ax.tick_params(axis='x', labelsize=9)
fig.suptitle('E35 matched validation screen: 960 predictions, 60 perturbations\n'
             'EMA0.9996 / batch32 / seed4242 / Heun50 / nonnegative predictions\n'
             'Partial interval [0.1,0.9]; orange = full [0,1]; points only, full refinements pending')
for suffix in ('png', 'pdf'):
    fig.savefig(out / f'screen_tradeoffs.{suffix}', dpi=180, bbox_inches='tight')
plt.close(fig)
outputs = [snapshot, selection, out / 'screen_tradeoffs.png', out / 'screen_tradeoffs.pdf']
proof = {'time': timestamp(), 'purpose': 'Descriptive display of the six original screen results; no metric or selection changes',
         'source_all_results_sha256_at_plot_time': file_sha256(source), 'source_all_results_count_at_plot_time': len(records),
         'source_may_later_append_full_refinements': True, 'screen_snapshot_sha256': file_sha256(snapshot),
         'calibration_config_sha256': file_sha256(calibration / 'calibration_config.json'),
         'verified_matching_generation_and_metric_files': checksums, 'script_sha256': file_sha256(__file__),
         'output_sha256': {str(path): file_sha256(path) for path in outputs}, 'test_access': False}
completed.write_text(json.dumps(proof, indent=2))
print(json.dumps({'settings': settings, 'figure': str(out / 'screen_tradeoffs.png'), 'test_access': False}))
