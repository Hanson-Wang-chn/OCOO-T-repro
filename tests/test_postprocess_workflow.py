"""CPU checks for explicit, validation-selected zero floors and test safeguards."""
import copy
import fcntl
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocoot.evaluate import official_metrics
from ocoot.postprocess import (DEFAULT_THRESHOLDS, apply_zero_floor, choose_feasible, choose_validation_candidate,
                              primary_score, quality_checks, zero_floor)
from ocoot.runtime import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def good_metrics():
    return {'PDCorr': .4, 'DEOver': .21, 'DEPrec': .22, 'DirAgr': .76,
            'PDS_L1': .8, 'PDS_L2': .82, 'PDS_cos': .81, 'MSE': .009, 'MAE': .06}


def test_feasible_selection_keeps_all_checks_and_prefers_original_on_exact_ties():
    original = {'tag': 'refine_00', 'cap': None, 'guidance': 2., 'clamp': True,
                'zero_floor': 0., 'metrics': good_metrics()}
    higher_but_bad = copy.deepcopy(original)
    higher_but_bad.update(tag='refine_01', zero_floor=.05)
    higher_but_bad['metrics'].update(PDCorr=.5, DEPrec=.17)
    assert primary_score(higher_but_bad['metrics']) > primary_score(original['metrics'])
    assert choose_feasible([higher_but_bad, original]) == original
    equal = {**copy.deepcopy(original), 'zero_floor': .05}
    assert choose_feasible([equal, original]) == original
    assert choose_feasible([higher_but_bad]) is None
    capped = {**original, 'cap': 16}
    ordinary = {**original, 'guidance': None}
    for bad in (capped, ordinary):
        with pytest.raises(ValueError, match='full positive-CFG'):
            choose_feasible([bad])


def test_reporting_shortfalls_consults_full_validation_without_lowering_thresholds():
    first = {'tag': 'refine_00', 'cap': None, 'guidance': 2., 'clamp': True,
             'zero_floor': .05, 'metrics': {**good_metrics(), 'DEPrec': .16}}
    second = {'tag': 'refine_01', 'cap': None, 'guidance': 1.5, 'clamp': True,
              'zero_floor': 0., 'metrics': {**good_metrics(), 'DEPrec': .14, 'PDCorr': .45}}
    assert choose_validation_candidate([first, second], DEFAULT_THRESHOLDS) is None
    selected = choose_validation_candidate([first, second], DEFAULT_THRESHOLDS, report_shortfall=True)
    assert selected == first
    assert not quality_checks(selected['metrics'], DEFAULT_THRESHOLDS)['DEPrec']


def test_zero_floor_preserves_legacy_values_and_rejects_unregistered_inputs():
    raw = np.array([[-1., -0., 0., .049, .05, .051, 1.]], dtype=np.float32)
    np.testing.assert_array_equal(apply_zero_floor(raw, 0), np.maximum(raw, 0))
    np.testing.assert_array_equal(apply_zero_floor(raw, .05), [[0, 0, 0, 0, raw[0, 4], raw[0, 5], 1]])
    assert apply_zero_floor(raw, .05).dtype == raw.dtype
    for bad in (-.1, .07, .05001, float('nan'), True):
        with pytest.raises(ValueError):
            zero_floor(bad)
    with pytest.raises(ValueError, match='Finite'):
        apply_zero_floor(np.array([[np.nan]], dtype=np.float32), .05)


@pytest.fixture
def postprocess_case(tmp_path, monkeypatch):
    x = np.array([[200, 200, 0]] + [[3, 1, 0]] * 12 + [[1, 3, 0]] * 12 +
                 [[1, 1, 0]] * 20 + [[3, 1, 0]] * 12 + [[1, 3, 0]] * 12, dtype=np.float32)
    meta = {'cell': np.zeros(len(x), dtype=np.int64), 'batch': np.zeros(len(x), dtype=np.int64),
            'pert': np.array([0] + [0] * 12 + [1] * 12 + [2] * 20 + [0] * 12 + [1] * 12),
            'split': np.array([0] + [1] * 24 + [3] * 20 + [2] * 24)}
    data_path = tmp_path / 'data'
    data_path.mkdir()
    vocab = {'all_genes': ['X', 'Y', 'non-targeting'], 'genes': ['G1', 'G2', 'G3']}
    np.save(data_path / 'expression.npy', x)
    np.savez(data_path / 'metadata.npz', **meta)
    (data_path / 'vocab.json').write_text(json.dumps(vocab))
    data = SimpleNamespace(path=data_path, x=x, meta=meta, val=np.arange(1, 25),
                           control=np.arange(25, 45), test=np.arange(45, 69), vocab=vocab)
    data.evaluation_indices = lambda split, cap: data.val if split == 'validation' and cap is None else data.test if split == 'test' and cap is None else None
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'finished.json').write_text('{}')
    cfg = {'run_dir': str(run), 'data_dir': str(data_path), 'model': {}, 'training': {'noise_std': 1.}}
    checkpoint = tmp_path / 'frozen.pt'
    torch.save({'config': cfg, 'step': 123, 'weights': 'ema_0.9996', 'model': {}}, checkpoint)
    pinned = tmp_path / 'source.py'
    pinned.write_text('pass\n')
    provenance = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
                  'step': 123, 'weights': 'ema_0.9996', 'data_dir': str(data_path),
                  'source_sha256': {str(pinned): file_sha256(pinned)},
                  'formal_guidance_policy': 'literal_dual_cfg_positive_scale',
                  'refine_cap': None, 'batch_size': 32}
    calibration = tmp_path / 'calibration'
    calibration.mkdir()
    records = []
    for i, (low, high) in enumerate([(.1, .9), (0., 1.)]):
        case = calibration / f'refine_{i:02d}'
        case.mkdir()
        raw = x[data.val].copy()
        raw[:, 2] = .01 * (i + 1)
        np.save(case / 'prediction.npy', raw)
        np.save(case / 'indices.npy', data.val)
        options = {'noise_std': 1., 'batch_size': 32, 'steps': 50, 'guidance': 1.5,
                   'cfg_min': low, 'cfg_max': high, 'clamp': False, 'seed': 4242}
        (case / 'generation_state.json').write_text(json.dumps({'completed': 24, 'total': 24, 'options': options}))
        metrics = official_metrics(data, data.val, np.maximum(raw, 0), case / 'metrics_clamped', de=True)
        raw_metrics = official_metrics(data, data.val, raw, case / 'metrics_raw', de=False)
        records.append({'tag': case.name, 'guidance': 1.5, 'cfg_min': low, 'cfg_max': high,
                        'clamp': True, 'cap': None, 'metrics': metrics, 'raw_metrics': raw_metrics,
                        'score': primary_score(metrics), 'cache_provenance': {
                            'directory': str(case), 'raw_unclamped': True, 'split': 'validation',
                            'files': {name: file_sha256(case / name) for name in
                                      ('prediction.npy', 'indices.npy', 'generation_state.json')}}})
    selected = {**copy.deepcopy(max(records, key=lambda r: r['score'])), **provenance,
                'sampling_steps': 50, 'noise_std': 1., 'seed': 4242}
    (calibration / 'calibration_config.json').write_text(json.dumps(provenance))
    (calibration / 'all_results.json').write_text(json.dumps(records))
    (calibration / 'selected.json').write_text(json.dumps(selected))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'floors': [0., .05], 'quality_thresholds': DEFAULT_THRESHOLDS}))
    select = runpy.run_path(str(ROOT / 'scripts/select_validation_postprocess.py'))['select']
    promote = runpy.run_path(str(ROOT / 'scripts/promote_validation_cache.py'))['promote']
    run_test = runpy.run_path(str(ROOT / 'scripts/evaluate_postprocess.py'))['run_test']
    for function in (select, promote, run_test):
        monkeypatch.setitem(function.__globals__, 'BenchmarkData', lambda _: data)
    return SimpleNamespace(data=data, cfg=cfg, run=run, checkpoint=checkpoint, calibration=calibration,
                           plan=plan, records=records, select=select, promote=promote, run_test=run_test,
                           postprocess=tmp_path / 'postprocessing', validation=tmp_path / 'validation',
                           baseline_test=tmp_path / 'test_clamp0', selected_test=tmp_path / 'test_selected')


def prepare_validated(c):
    selected, _ = c.select(c.calibration, c.postprocess, c.plan)
    assert selected['zero_floor'] == .05
    assert all(quality_checks(selected['metrics']).values())
    c.promote(c.calibration, c.validation, c.postprocess / 'selected.json')
    return selected


def test_full_selection_and_promotion_match_real_metrics_and_are_idempotent(postprocess_case):
    c = postprocess_case
    original_sha = file_sha256(c.calibration / 'selected.json')
    selected = prepare_validated(c)
    assert selected['metrics']['MSE'] == 0 and selected['metrics']['DEPrec'] == 1
    prediction = np.load(c.validation / 'prediction.npy')
    np.testing.assert_array_equal(prediction, c.data.x[c.data.val])
    assert json.loads((c.validation / 'evaluation_config.json').read_text())['zero_floor'] == .05
    assert json.loads((c.validation / 'generation_state.json').read_text())['options']['zero_floor'] == .05
    frozen = file_sha256(c.postprocess / 'selected.json')
    c.select(c.calibration, c.postprocess, c.plan)
    c.promote(c.calibration, c.validation, c.postprocess / 'selected.json')
    assert file_sha256(c.postprocess / 'selected.json') == frozen
    assert file_sha256(c.calibration / 'selected.json') == original_sha


def test_postprocessing_cannot_write_inside_an_original_raw_refinement(postprocess_case):
    c = postprocess_case
    source = c.calibration / 'refine_00'
    raw_sha = file_sha256(source / 'prediction.npy')
    destination = source / 'nested_postprocessing'
    with pytest.raises(ValueError, match='separate'):
        c.select(c.calibration, destination, c.plan)
    assert not destination.exists() and file_sha256(source / 'prediction.npy') == raw_sha


@pytest.mark.parametrize('fault', ['capped', 'ordinary', 'corrupt_raw', 'mixed_marker', 'threshold_change', 'wrong_perturbation_library'])
def test_selector_rejects_invalid_or_unregistered_refinements(postprocess_case, fault):
    c = postprocess_case
    if fault == 'threshold_change':
        plan = json.loads(c.plan.read_text())
        plan['quality_thresholds']['min_val_deprec'] = .1
        c.plan.write_text(json.dumps(plan))
    else:
        records = json.loads((c.calibration / 'all_results.json').read_text())
        if fault == 'capped': records[0]['cap'] = 16
        if fault == 'ordinary': records[0]['guidance'] = None
        source = Path(records[0]['cache_provenance']['directory'])
        if fault == 'corrupt_raw': np.save(source / 'prediction.npy', np.ones((24, 3), dtype=np.float32))
        if fault == 'mixed_marker':
            marker = json.loads((source / 'generation_state.json').read_text())
            marker['options']['extra_seed'] = 62525
            (source / 'generation_state.json').write_text(json.dumps(marker))
            records[0]['cache_provenance']['files']['generation_state.json'] = file_sha256(source / 'generation_state.json')
        if fault == 'wrong_perturbation_library':
            import pandas as pd
            file = source / 'metrics_clamped/results.csv'
            frame = pd.read_csv(file)
            frame.loc[0, 'perturbation'] = 'another_target'
            frame.to_csv(file, index=False)
        (c.calibration / 'all_results.json').write_text(json.dumps(records))
    with pytest.raises((ValueError, AssertionError)):
        c.select(c.calibration, c.postprocess, c.plan)
    assert not (c.postprocess / 'selected.json').exists()


def install_fake_test_generator(c, monkeypatch):
    calls = []
    module = c.run_test.__globals__

    class Child:
        pid = 12345
        def __init__(self, command, **kwargs):
            assert (c.selected_test / 'test_protocol.json').exists(), 'Freeze settings before generation'
            assert '--cap' not in command and command[command.index('--split') + 1] == 'test'
            calls.append(command)
            c.baseline_test.mkdir(exist_ok=True)
            selected = json.loads((c.postprocess / 'selected.json').read_text())
            options = {'noise_std': 1., 'batch_size': 32, 'steps': 50, 'guidance': selected['guidance'],
                       'cfg_min': selected['cfg_min'], 'cfg_max': selected['cfg_max'], 'clamp': True, 'seed': 4242}
            evaluation = {'checkpoint': str(c.checkpoint.resolve()), 'checkpoint_sha256': file_sha256(c.checkpoint),
                          'step': 123, 'weights': 'ema_0.9996', 'split': 'test', 'cap': None, 'de': True,
                          'model': {}, **options}
            raw = c.data.x[c.data.test].copy()
            raw[:, 2] = .01
            np.save(c.baseline_test / 'indices.npy', c.data.test)
            np.save(c.baseline_test / 'prediction_raw.npy', raw)
            np.save(c.baseline_test / 'prediction.npy', np.maximum(raw, 0))
            (c.baseline_test / 'evaluation_config.json').write_text(json.dumps(evaluation))
            (c.baseline_test / 'generation_state.json').write_text(json.dumps({'completed': 24, 'total': 24, 'options': options}))
            official_metrics(c.data, c.data.test, np.maximum(raw, 0), c.baseline_test / 'metrics', de=True)
        def wait(self): return 0

    monkeypatch.setattr(module['subprocess'], 'Popen', Child)
    return calls


def test_locked_test_reuses_baseline_after_cpu_interruption(postprocess_case, monkeypatch):
    c = postprocess_case
    prepare_validated(c)
    calls = install_fake_test_generator(c, monkeypatch)
    module = c.run_test.__globals__
    original_metrics = module['official_metrics']
    def interrupted(*args, **kwargs): raise RuntimeError('interrupted CPU postprocessing')
    monkeypatch.setitem(module, 'official_metrics', interrupted)
    with pytest.raises(RuntimeError, match='interrupted CPU'):
        c.run_test(c.postprocess / 'selected.json', c.validation, c.baseline_test, c.selected_test)
    assert len(calls) == 1 and not (c.selected_test / 'postprocess_test_completed.json').exists()
    raw_sha = file_sha256(c.baseline_test / 'prediction_raw.npy')
    monkeypatch.setitem(module, 'official_metrics', original_metrics)
    result = c.run_test(c.postprocess / 'selected.json', c.validation, c.baseline_test, c.selected_test)
    assert len(calls) == 1 and result['metrics']['MSE'] == 0 and result['metrics']['DEPrec'] == 1
    assert result['new_postprocess_generated_samples'] == 0 and not result['selection_changed_after_test']
    assert file_sha256(c.selected_test / 'prediction_raw.npy') == raw_sha
    assert not (c.selected_test / 'prediction_raw.npy').is_symlink()
    assert c.run_test(c.postprocess / 'selected.json', c.validation, c.baseline_test, c.selected_test) == result
    assert len(calls) == 1


@pytest.mark.parametrize('fault', ['training_unfinished', 'training_locked', 'validation_changed', 'selection_changed'])
def test_formal_test_refuses_before_any_generation(postprocess_case, monkeypatch, fault):
    c = postprocess_case
    prepare_validated(c)
    calls = install_fake_test_generator(c, monkeypatch)
    handle = None
    if fault == 'training_unfinished': (c.run / 'finished.json').unlink()
    if fault == 'training_locked':
        handle = (c.run / 'training.lock').open('a+')
        fcntl.flock(handle, fcntl.LOCK_EX)
    if fault == 'validation_changed':
        with (c.validation / 'prediction.npy').open('ab') as stream: stream.write(b'changed')
    if fault == 'selection_changed':
        selected = json.loads((c.postprocess / 'selected.json').read_text())
        selected['cfg_min'] = 0.02
        (c.postprocess / 'selected.json').write_text(json.dumps(selected))
    try:
        with pytest.raises((ValueError, AssertionError)):
            c.run_test(c.postprocess / 'selected.json', c.validation, c.baseline_test, c.selected_test)
        assert calls == []
    finally:
        if handle is not None: handle.close()
