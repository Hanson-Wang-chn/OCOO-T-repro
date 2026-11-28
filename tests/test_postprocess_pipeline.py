"""The optional parent must preserve stage order and stop before failed tests."""
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import yaml

from ocoot.postprocess import DEFAULT_THRESHOLDS, choose_validation_candidate
from ocoot.runtime import file_sha256


def metrics():
    return {'PDCorr': .4, 'DEOver': .21, 'DEPrec': .22, 'DirAgr': .76,
            'PDS_L1': .8, 'PDS_L2': .82, 'PDS_cos': .81, 'MSE': .009, 'MAE': .06}


def test_round_three_full_validation_uses_registered_mse_weight():
    evidence = Path(__file__).resolve().parents[1] / 'runs/repro3_main/calibration/postprocessing/review.json'
    candidates = json.loads(evidence.read_text())['candidates']
    assert len(candidates) == 4
    legacy = choose_validation_candidate(candidates, DEFAULT_THRESHOLDS)
    weighted = choose_validation_candidate(candidates, DEFAULT_THRESHOLDS, mse_weight=10.0)
    assert (legacy['guidance'], legacy['zero_floor']) == (1.375, .05)
    assert (weighted['guidance'], weighted['zero_floor']) == (1.125, .05)


def setup_pipeline(tmp_path, monkeypatch, use_postprocess, no_feasible=False, changed_gate=False,
                   report_shortfall=False):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'finished.json').write_text('{}')
    checkpoint = run / 'best_generation.pt'
    checkpoint.write_bytes(b'full-data frozen default EMA placeholder')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'run_dir': str(run), 'training': {}}))
    plan = tmp_path / 'postprocess_plan.json'
    thresholds = copy.deepcopy(DEFAULT_THRESHOLDS)
    if changed_gate: thresholds['min_val_deprec'] = .1
    plan.write_text(json.dumps({'floors': [0., .05], 'quality_thresholds': thresholds,
                                'allow_validation_shortfall': report_shortfall}))
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/run_experiment.py'))
    module = namespace['main'].__globals__
    monkeypatch.setitem(module, 'readme_event', lambda *args, **kwargs: None)
    calls = []

    def fake_run(command, **kwargs):
        script = Path(command[1]).name
        calls.append(command)
        option = lambda key: Path(command[command.index(key) + 1])
        if script == 'calibrate.py':
            out = option('--outdir')
            out.mkdir()
            selected = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
                        'formal_guidance_policy': 'literal_dual_cfg_positive_scale',
                        'weights': 'ema_0.9996', 'cap': None, 'cache_provenance': {'directory': 'fixture'},
                        'guidance': 1.5, 'cfg_min': .1, 'cfg_max': .9, 'batch_size': 32, 'clamp': True}
            (out / 'selected.json').write_text(json.dumps(selected))
        elif script == 'select_validation_postprocess.py':
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            out = option('--outdir')
            out.mkdir()
            selected = json.loads((option('--calibration') / 'selected.json').read_text())
            if no_feasible:
                (out / 'review.json').write_text(json.dumps({'selected': None}))
                raise subprocess.CalledProcessError(2, command)
            selected['zero_floor'] = .05
            (out / 'selected.json').write_text(json.dumps(selected))
        elif script == 'promote_validation_cache.py':
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            out = option('--outdir')
            (out / 'metrics').mkdir(parents=True)
            validation = metrics()
            if report_shortfall:
                validation['DEPrec'] = .16
            (out / 'metrics/summary.json').write_text(json.dumps(validation))
            if use_postprocess:
                assert option('--selection') == run / 'calibration/postprocessing/selected.json'
            else:
                assert '--selection' not in command
        elif script in ('evaluate.py', 'evaluate_postprocess.py'):
            out = option('--outdir')
            (out / 'metrics').mkdir(parents=True)
            (out / 'metrics/summary.json').write_text(json.dumps(metrics()))
        elif script == 'report.py':
            out = option('--outdir')
            out.mkdir(parents=True)
            (out / 'REPORT.md').write_text('synthetic pipeline-order test')
        elif script != 'plot_progress.py':
            raise AssertionError('Unexpected child task: ' + script)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module['subprocess'], 'run', fake_run)
    arguments = ['run_experiment.py', '--config', str(config)]
    if use_postprocess: arguments += ['--postprocess-plan', str(plan)]
    monkeypatch.setattr(sys, 'argv', arguments)
    return namespace['main'], run, calls


@pytest.mark.parametrize('use_postprocess', [False, True])
def test_parent_uses_correct_full_validation_and_test_routes(tmp_path, monkeypatch, use_postprocess):
    main, run, calls = setup_pipeline(tmp_path, monkeypatch, use_postprocess)
    main()
    scripts = [Path(command[1]).name for command in calls]
    assert 'train.py' not in scripts and scripts[0] == 'calibrate.py'
    if use_postprocess:
        assert scripts[:4] == ['calibrate.py', 'select_validation_postprocess.py',
                              'promote_validation_cache.py', 'evaluate_postprocess.py']
        assert scripts.count('report.py') == 2
        review = json.loads((run / 'validation_postprocess_review.json').read_text())
    else:
        assert scripts[:3] == ['calibrate.py', 'promote_validation_cache.py', 'evaluate.py']
        assert 'select_validation_postprocess.py' not in scripts and scripts.count('report.py') == 1
        review = json.loads((run / 'validation_review.json').read_text())
    assert review['thresholds'] == DEFAULT_THRESHOLDS and all(review['checks'].values())
    assert json.loads((run / 'pipeline_state.json').read_text())['stage'] == 'evaluated'


def test_parent_stops_before_test_when_no_postprocess_candidate_passes(tmp_path, monkeypatch):
    main, run, calls = setup_pipeline(tmp_path, monkeypatch, True, no_feasible=True)
    with pytest.raises(SystemExit) as stopped: main()
    assert stopped.value.code == 2
    assert [Path(command[1]).name for command in calls] == ['calibrate.py', 'select_validation_postprocess.py']
    assert json.loads((run / 'pipeline_state.json').read_text())['stage'] == 'validation_needs_review'


def test_parent_refuses_relaxed_postprocess_quality_plan_before_any_child(tmp_path, monkeypatch):
    main, run, calls = setup_pipeline(tmp_path, monkeypatch, True, changed_gate=True)
    with pytest.raises(ValueError, match='original validation threshold'): main()
    assert calls == [] and not (run / 'pipeline_state.json').exists()


def test_registered_full_test_preserves_and_reports_validation_shortfall(tmp_path, monkeypatch):
    main, run, calls = setup_pipeline(tmp_path, monkeypatch, True, report_shortfall=True)
    main()
    review = json.loads((run / 'validation_postprocess_review.json').read_text())
    assert review['thresholds'] == DEFAULT_THRESHOLDS
    assert not review['checks']['DEPrec']
    states = [json.loads(line).get('stage') for line in (run / 'events.jsonl').read_text().splitlines()]
    assert states.index('validation_shortfall_reported') < states.index('full_test_de_with_frozen_postprocess')
    assert states[-1] == 'evaluated'
