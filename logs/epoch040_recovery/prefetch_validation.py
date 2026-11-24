"""Precompute independent validation cases using the registered sampling protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import generate
from ocoot.flow import sample
from ocoot.runtime import file_sha256, timestamp


RUN = Path('runs/main_ddp_recovery')
CALIBRATION = RUN / 'calibration'
WORK = RUN / 'calibration_prefetch'


def read(path):
    return json.loads(path.read_text())


def load():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    constitution = CALIBRATION / 'calibration_config.json'
    protocol = read(constitution)
    for path, digest in protocol['source_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Unmatched calibration source: {path}')
    path = protocol['checkpoint']
    if file_sha256(path) != protocol['checkpoint_sha256']:
        raise ValueError('Frozen calibration checkpoint changed')
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert checkpoint['step'] == protocol['step'] and checkpoint['weights'] == protocol['weights'] == 'ema_0.9996'
    cfg = checkpoint['config']
    data = BenchmarkData(cfg['data_dir'])
    model = data.build_model(cfg['model'])
    model.load_state_dict(checkpoint['model'])
    return protocol, cfg, data, torch.compile(model.cuda().eval(), fullgraph=True)


def verify_device(protocol, cfg, data, model):
    reference = CALIBRATION / 'screen_00'
    state = read(reference / 'generation_state.json')
    assert state['completed'] >= protocol['batch_size'] and state['options']['guidance'] is None
    batch = protocol['batch_size']
    indices = data.evaluation_indices('validation', protocol['screen_cap'])[:batch]
    np.testing.assert_array_equal(indices, np.load(reference / 'indices.npy')[:batch])
    p, c, b = [torch.from_numpy(data.meta[k][indices]).cuda() for k in ('pert', 'cell', 'batch')]
    pred = sample(model, p, c, b, data.x.shape[1], noise_std=cfg['training']['noise_std'],
                  steps=50, seed=4242).cpu().numpy()
    expected = np.array(np.load(reference / 'prediction.npy', mmap_mode='r')[:batch])
    np.testing.assert_array_equal(pred, expected)
    return {'device': torch.cuda.get_device_name(), 'n': batch, 'bitwise_equal_screen_00': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=int, choices=(2, 3, 4))
    args = p.parse_args()
    tag = f'screen_{args.case:02d}'
    protocol, cfg, data, model = load()
    same = verify_device(protocol, cfg, data, model)
    scales = [float(s) for s in protocol['scales'].split(',')]
    guidance = scales[args.case-1]
    indices = data.evaluation_indices('validation', protocol['screen_cap'])
    case = WORK / tag
    prediction = generate(model, data, indices, case, noise_std=cfg['training']['noise_std'],
                          steps=50, batch_size=protocol['batch_size'], guidance=guidance,
                          cfg_min=.1, cfg_max=.9, clamp=False, seed=4242)
    marker = read(case / 'generation_state.json')
    assert marker['completed'] == marker['total'] == len(indices) and np.isfinite(prediction).all()
    proof = {'time': timestamp(), 'tag': tag, 'marker': marker, 'device_match': same,
             'validation_only': True, 'checkpoint_sha256': protocol['checkpoint_sha256'],
             'files': {n: file_sha256(case/n) for n in ('prediction.npy','indices.npy','generation_state.json')}}
    (case / 'prefetch_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    try:
        (CALIBRATION / tag).symlink_to(case.resolve(), target_is_directory=True)
    except FileExistsError:
        print('Primary calibration already created the case. Independent cache kept unpromoted.', flush=True)
    else:
        print('Published validation prediction cache:', tag, json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
