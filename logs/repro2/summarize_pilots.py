"""Summarize the registered matched Replogle–Nadig pilots."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
import yaml

from ocoot.evaluate import METRICS
from ocoot.runtime import file_sha256, timestamp

root = Path('runs/repro2_pilots')
registered = json.loads((root / 'plan.json').read_text())
helpers = runpy.run_path('scripts/analyze_inference_review.py')
metric_order = list(METRICS.values())
pd_id = metric_order.index('PDCorr')
pds_ids = [metric_order.index(k) for k in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
baseline = root / 'control'
subset = np.load(baseline / 'train_indices.npy')
generation = baseline / 'validation_epoch_012'
indices = np.load(generation / 'indices.npy')
options = json.loads((generation / 'generation_state.json').read_text())
config = yaml.safe_load((baseline / 'config.yaml').read_text())
rows = []
for key, entry in registered['cases'].items():
    run = root / key
    finished = run / 'finished.json'
    if not finished.exists():
        continue
    case = run / 'validation_epoch_012'
    de_folder = case / 'metrics_de_diagnostic'
    summary = de_folder / 'summary.json'
    if not (de_folder / 'completed.json').exists():
        continue
    registered_config = Path(entry['config'])
    assert file_sha256(registered_config) == entry['config_sha256']
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    assert cfg == yaml.safe_load(registered_config.read_text())
    expected_model = dict(config['model'], **entry['change'])
    assert cfg['model'] == expected_model
    assert {k: v for k, v in cfg['training'].items() if k != 'distributed_generation'} == config['training']
    assert json.loads(finished.read_text())['step'] == registered['updates']
    state = torch.load(case / 'model.pt', map_location='cpu', mmap=True, weights_only=False)
    assert state['weights'] == 'raw_pilot_only' and state['config'] == cfg and state['step'] == registered['updates']
    del state
    marker = json.loads((case / 'generation_state.json').read_text())
    assert marker == options and marker['completed'] == marker['total'] == 960
    np.testing.assert_array_equal(np.load(run / 'train_indices.npy'), subset)
    np.testing.assert_array_equal(np.load(case / 'indices.npy'), indices)
    frame = pd.read_csv(de_folder / 'results.csv').set_index('perturbation').sort_index()
    assert frame.index.is_unique and len(frame) == 60
    values = frame[list(METRICS)].to_numpy(dtype=np.float64)
    metrics = json.loads(summary.read_text())
    means = np.nanmean(values, axis=0)
    np.testing.assert_allclose(means, [metrics[k] for k in metric_order], atol=1e-10, rtol=0)
    if key == 'control':
        labels = frame.index
        draws = np.random.default_rng(9232).integers(0, len(labels), (2000, len(labels)))
        boots_ref = helpers['resampled_means'](values, draws)
        ref_score = means[pd_id] + means[pds_ids].mean()
    else:
        assert frame.index.equals(labels)
    score = means[pd_id] + means[pds_ids].mean()
    delta = helpers['resampled_means'](values, draws) - boots_ref
    score_deltas = delta[:, pd_id] + delta[:, pds_ids].mean(1)
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    probes = [event for event in events if event['event'] == 'validation']
    final_probe = next(event for event in reversed(probes) if event['step'] == registered['updates'])
    timings = [event for event in events if event['event'] == 'train' and event['step'] >= 40]
    started = next(event for event in events if event['event'] == 'start')
    part_counts = [json.loads(p.read_text())['completed'] for p in case.glob('parts_w*/rank_*/generation_state.json')]
    rows.append({'case':key, **metrics, 'score':float(score), 'score_delta':float(score-ref_score),
                 'score_delta_ci95':np.quantile(score_deltas,[.025,.975]).tolist(),
                 'final_raw_endpoint_mse':final_probe['raw']['endpoint_mse'],
                 'final_ema_endpoint_mse':final_probe['ema_0.9996']['endpoint_mse'],
                 'median_log_cells_s':float(np.median([event['cells_per_sec'] for event in timings])),
                 'peak_at_log_GB':max(event['peak_gpu_GB'] for event in timings),
                 'parameters':started['parameters'], 'evaluation_parts':part_counts,
                 'source_sha256':{p.name: file_sha256(p) for p in (summary, de_folder/'results.csv', case/'model.pt')}})
report = {'time':timestamp(), 'scope':'Matched raw-weight diagnostics on 960 validation cells / 60 perturbations.',
          'paired_interval':'2,000 bootstrap draws over perturbations from the same library with seed 9232.',
          'plan_sha256':file_sha256(root / 'plan.json'), 'rows':rows}
path = root / 'summary.json'
temp = path.with_suffix('.tmp')
temp.write_text(json.dumps(report,indent=2,allow_nan=False))
temp.replace(path)
pd.DataFrame(rows).drop(columns='source_sha256').to_csv(root / 'summary.csv',index=False)
print(json.dumps(report,indent=2))
