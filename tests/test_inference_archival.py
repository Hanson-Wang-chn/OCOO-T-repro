"""Preserve inference bytes and refuse active/full runs or conflicting copies."""
import json
from pathlib import Path
import runpy

import pytest
import torch
import yaml

from ocoot.runtime import file_sha256


relocate = runpy.run_path('scripts/archive_finished_pilot_inference.py')['relocate']


def pilot(tmp_path, subset=16, finished=True):
    run = tmp_path / 'pilot'
    run.mkdir()
    cfg = {'run_dir': str(run), 'training': {'train_subset': subset}}
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg))
    if finished:
        (run / 'finished.json').write_text(json.dumps({'step': 3}))
    for name, step in [('best_denoise.pt', 2), ('final_ema.pt', 3)]:
        torch.save({'config': cfg, 'step': step, 'model': {'w': torch.arange(7)}}, run / name)
    return run


def test_exact_copy_loadability_and_idempotence(tmp_path):
    run = pilot(tmp_path)
    original = {p.name: file_sha256(p) for p in run.glob('*.pt')}
    result = relocate([run], tmp_path / 'archive', tmp_path / 'audit')
    assert len(result['records']) == 2 and result['bytes_newly_relocated'] > 0
    for name, digest in original.items():
        assert (run / name).is_symlink() and file_sha256(run / name) == digest
        assert torch.equal(torch.load(run / name, weights_only=False)['model']['w'], torch.arange(7))
    assert relocate([run], tmp_path / 'archive', tmp_path / 'audit')['bytes_newly_relocated'] == 0


@pytest.mark.parametrize('subset,finished', [(16, False), (None, True)])
def test_refuse_active_or_full_data_runs(tmp_path, subset, finished):
    run = pilot(tmp_path, subset, finished)
    with pytest.raises(ValueError, match='ineligible|subset'):
        relocate([run], tmp_path / 'archive', tmp_path / 'audit')
    assert all(not p.is_symlink() for p in run.glob('*.pt'))


def test_conflicting_destination_keeps_original_bytes(tmp_path):
    run = pilot(tmp_path)
    digest = file_sha256(run / 'best_denoise.pt')
    dest = tmp_path / 'archive' / run.name
    dest.mkdir(parents=True)
    (dest / 'best_denoise.pt').write_bytes(b'different artifact')
    with pytest.raises(FileExistsError, match='different bytes'):
        relocate([run], tmp_path / 'archive', tmp_path / 'audit')
    assert not (run / 'best_denoise.pt').is_symlink()
    assert file_sha256(run / 'best_denoise.pt') == digest
