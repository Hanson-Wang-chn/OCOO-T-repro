import copy
import runpy

import numpy as np
import pytest


@pytest.fixture
def review():
    return runpy.run_path('scripts/review_ema_weights.py')


def protocol(tmp_path):
    options = dict(noise_std=1., batch_size=32, steps=50, guidance=2.,
                   cfg_min=0., cfg_max=1., seed=4242)
    source = dict(completed=960, total=960, options={**options, 'clamp': False})
    candidate = dict(completed=960, total=960, options={**options, 'clamp': True})
    full = str(tmp_path / 'full.pt')
    config = dict(checkpoint=full, checkpoint_sha256='frozen', step=39130,
                  weights='ema_0.9999', split='validation', cap=16, guidance=2.,
                  cfg_min=0., cfg_max=1., clamp=True, de=True, seed=4242,
                  steps=50, batch_size=32, noise_std=1.)
    return options, source, config, candidate, full


def test_only_ema_changes_and_storage_clamp_does_not_change_raw_protocol(review, tmp_path):
    options, source, config, marker, full = protocol(tmp_path)
    review['validate_protocol'](options, source, config, marker, full, 'frozen')
    for field, invalid in [('weights', 'ema_0.9996'), ('step', 39131), ('split', 'test'),
                           ('batch_size', 16), ('guidance', 1.5), ('cfg_min', .1),
                           ('checkpoint_sha256', 'changed')]:
        wrong = {**config, field: invalid}
        with pytest.raises(ValueError, match='beyond'):
            review['validate_protocol'](options, source, wrong, marker, full, 'frozen')


def test_incomplete_or_different_rng_generation_is_rejected(review, tmp_path):
    options, source, config, marker, full = protocol(tmp_path)
    for side in ('reference', 'candidate'):
        for field, invalid in [('seed', 4243), ('steps', 51), ('noise_std', .5)]:
            changed = copy.deepcopy(source if side == 'reference' else marker)
            changed['options'][field] = invalid
            with pytest.raises(ValueError):
                review['validate_protocol'](options, changed if side == 'reference' else source,
                    config, changed if side == 'candidate' else marker, full, 'frozen')
    marker['completed'] = 928
    with pytest.raises(ValueError, match='incomplete'):
        review['validate_protocol'](options, source, config, marker, full, 'frozen')


def test_exact_distinct_indices_reject_shuffle_duplicate_and_population_size(review):
    indices = np.arange(960)
    review['validate_indices'](indices, indices.copy())
    with pytest.raises(AssertionError):
        review['validate_indices'](indices, indices[::-1])
    for wrong in (indices[:-1], np.zeros(960, dtype=int)):
        with pytest.raises(ValueError, match='960 distinct'):
            review['validate_indices'](wrong, wrong)


def test_source_and_plan_drift_are_rejected_during_sampling(review, tmp_path):
    import hashlib
    plan_file, source, prerequisite = [tmp_path / p for p in ('plan.json', 'source.py', 'reference.csv')]
    plan_file.write_text('{}')
    source.write_text('original source')
    prerequisite.write_text('frozen reference')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    digest = sha(plan_file)
    plan = {'source_sha256': {str(source): sha(source)},
            'prerequisite_sha256': {str(prerequisite): sha(prerequisite)}}
    review['verify_pins'](plan, plan_file, digest)
    plan_file.write_text('changed')
    with pytest.raises(ValueError, match='plan changed'):
        review['verify_pins'](plan, plan_file, digest)
    plan_file.write_text('{}')
    prerequisite.write_text('drift')
    review['verify_pins'](plan, plan_file, digest, prerequisites=False)
    with pytest.raises(ValueError, match='source changed'):
        review['verify_pins'](plan, plan_file, digest)
    source.write_text('drift')
    with pytest.raises(ValueError, match='source changed'):
        review['verify_pins'](plan, plan_file, digest, prerequisites=False)
