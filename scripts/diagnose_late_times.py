"""Frozen-checkpoint velocity diagnostics at fixed times, including the last ODE time.

These are paired denoising measurements, not generation scores. The plan pins
weights, cells, noise, precision and time grid before this program runs.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.runtime import exclusive_run, file_sha256, log_event, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def measurements(prediction, x, noise, time):
    target = x - noise
    z = np.float32(time) * x + np.float32(1 - time) * noise
    endpoint = z + np.float32(1 - time) * prediction
    result = {'velocity_mse': float(np.square(prediction - target).mean(dtype=np.float64)),
              'endpoint_mse': float(np.square(endpoint - x).mean(dtype=np.float64))}
    for name, mask in [('zero', x == 0), ('positive', x > 0)]:
        error = (prediction - target)[mask].astype(np.float64)
        terminal_error = (endpoint - x)[mask].astype(np.float64)
        result[name] = {'entries': int(mask.sum()), 'velocity_mse': float(np.mean(error ** 2)),
                        'endpoint_mse': float(np.mean(terminal_error ** 2)),
                        'endpoint_bias': float(np.mean(terminal_error)),
                        'endpoint_abs_error': float(np.mean(np.abs(terminal_error)))}
    mask = x == 0
    v, reference = prediction[mask].astype(np.float64), -noise[mask].astype(np.float64)
    covariance = float(np.mean((v - v.mean()) * (reference - reference.mean())))
    result['zero'].update(
        velocity_noise_slope=covariance / float(np.var(reference)),
        velocity_noise_correlation=covariance / max(float(np.std(v) * np.std(reference)), 1e-20),
        endpoint_positive_fraction=float(np.mean(endpoint[mask] > 0)))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    out = Path(args.outdir)
    with exclusive_run(out, 'late_time_probe'):
        plan = json.loads((out / 'plan.json').read_text())
        if plan['script_sha256'] != file_sha256(__file__):
            raise ValueError('Probe implementation differs from preregistration')
        protocol_path = out / 'probe_config.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != plan:
            raise ValueError('Probe directory already has different registered settings')
        atomic_json(protocol_path, plan)
        torch.set_num_threads(4)
        data = BenchmarkData(plan['data_dir'])
        if file_sha256(data.path / 'metadata.npz') != plan['metadata_sha256']:
            raise ValueError('Benchmark metadata changed')
        indices = np.sort(np.random.default_rng(plan['cell_seed']).choice(data.val, plan['cells'], replace=False))
        assert np.all(data.meta['split'][indices] == 1)
        np.save(out / 'indices.npy', indices)
        x, pert, cell, batch = data.batch(indices)
        generator = torch.Generator(device='cuda').manual_seed(plan['noise_seed'])
        noise = torch.randn(x.shape, device='cuda', generator=generator) * plan['noise_std']
        x_cpu, noise_cpu = x.cpu().numpy(), noise.cpu().numpy()
        np.savez_compressed(out / 'fixed_inputs.npz', indices=indices, x=x_cpu, noise=noise_cpu)
        oracle = measurements(x_cpu - noise_cpu, x_cpu, noise_cpu, .98)
        if oracle['velocity_mse'] != 0 or oracle['endpoint_mse'] > 1e-12:
            raise ValueError('Analytic velocity oracle failed; diagnostic units are incorrect')
        records = {}
        for case in plan['cases']:
            path = Path(case['checkpoint'])
            if file_sha256(path) != case['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint changed')
            folder = out / case['tag']
            folder.mkdir(exist_ok=True)
            if (folder / 'summary.json').exists():
                records[case['tag']] = json.loads((folder / 'summary.json').read_text())
                continue
            state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            np.testing.assert_array_equal(state['model']['gene_indices'].numpy(), data.meta['gene_indices'])
            if state.get('weights') != case['weights']:
                raise ValueError('Checkpoint weight variant differs from plan')
            model = data.build_model(state['config']['model'])
            model.load_state_dict(state['model'], strict=True)
            values = model.value(torch.tensor([[-.001], [0.], [.001]], dtype=torch.float32))
            gain = float(((values[2] - values[0]) / .002).square().mean().sqrt())
            model = model.cuda().eval()
            del state
            torch.backends.cuda.matmul.allow_tf32 = case['precision'] == 'bf16'
            forward = torch.compile(model, fullgraph=True)
            marker = folder / 'progress.json'
            start = json.loads(marker.read_text())['completed_times'] if marker.exists() else 0
            prediction = np.lib.format.open_memmap(folder / 'velocity.npy', mode='r+' if start else 'w+',
                                                  dtype=np.float32, shape=(len(plan['times']), len(indices), data.x.shape[1]))
            for i in range(start, len(plan['times'])):
                t = plan['times'][i]
                z = t * x + (1 - t) * noise
                times = torch.full((len(x),), t, device='cuda')
                for offset in range(0, len(x), plan['batch_size']):
                    sl = slice(offset, offset + plan['batch_size'])
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=case['precision'] == 'bf16'):
                        v = forward(z[sl], times[sl], pert[sl], cell[sl], batch[sl]).float()
                    prediction[i, sl] = v.cpu().numpy()
                prediction.flush()
                if not np.isfinite(prediction[i]).all():
                    raise ValueError('Nonfinite velocity diagnostic')
                atomic_json(marker, {'completed_times': i + 1, 'time': timestamp()})
                log_event(out / 'events.jsonl', {'event': 'fixed_time_completed', 'case': case['tag'], 't': t})
            rows = [{'t': t, **measurements(prediction[i], x_cpu, noise_cpu, t)}
                    for i, t in enumerate(plan['times'])]
            record = {'time': timestamp(), 'case': case, 'cells': len(indices), 'rows': rows,
                      'value_encoder_f32_small_signal_rms_gain': gain,
                      'velocity_sha256': file_sha256(folder / 'velocity.npy'),
                      'note': 'Fixed-time, matched denoising only. Oracle verifies units, not attainable conditional Bayes error. '
                              'Raw short pilots and mature EMA have different training budgets; only matched cases support comparisons.'}
            atomic_json(folder / 'summary.json', record)
            records[case['tag']] = record
            del forward, model, prediction
            gc.collect()
            torch.cuda.empty_cache()
        result = {'time': timestamp(), 'plan_sha256': file_sha256(out / 'plan.json'),
                  'oracle': oracle, 'indices_sha256': file_sha256(out / 'indices.npy'),
                  'records': records, 'model_or_training_change': False, 'test_access': False}
        atomic_json(out / 'summary.json', result)
        print(json.dumps({tag: {'gain': r['value_encoder_f32_small_signal_rms_gain'], 'last_time': r['rows'][-1]}
                          for tag, r in records.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
