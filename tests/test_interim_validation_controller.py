"""CPU process-safety and no-feasible-result checks for the interim controller."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest

from ocoot.postprocess import DEFAULT_THRESHOLDS
from ocoot.runtime import file_sha256


def namespace():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / 'logs/run_epoch035_validation_calibration.py'))


def test_real_guard_rejects_finished_or_test_directory(tmp_path):
    ns = namespace()
    tick = Path(f'/proc/{os.getpid()}/stat').read_text().split(') ')[1].split()[19]
    plan = {'trainer_pid': os.getpid(), 'trainer_start_tick': tick, 'main_run': str(tmp_path),
            'main_config_sha256': file_sha256('configs/main_selected.yaml')}
    ns['guard_main'](plan)
    (tmp_path / 'finished.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Training finished'):
        ns['guard_main'](plan)
    (tmp_path / 'finished.json').unlink()
    (tmp_path / 'test_full_fixture').mkdir()
    with pytest.raises(RuntimeError, match='test directory'):
        ns['guard_main'](plan)


def test_interrupt_terminates_only_validation_child_and_retains_cache(tmp_path, monkeypatch):
    ns = namespace()
    monkeypatch.setitem(ns['Controller'].execute.__globals__, 'guard_main', lambda plan: None)
    controller = ns['Controller']({}, tmp_path)
    marker = tmp_path / 'cached_batch.json'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    timer = threading.Timer(.8, lambda: setattr(controller, 'stop', True))
    try:
        timer.start()
        command = [sys.executable, '-c',
                   'import pathlib,sys,time,os;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)',
                   str(marker)]
        with pytest.raises(InterruptedError):
            controller.execute(command, 'fixture', cpu=True, poll_seconds=.01)
        assert marker.exists() and unrelated.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    finally:
        timer.cancel()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_selector_exit_two_is_an_expected_review_result(tmp_path, monkeypatch):
    ns = namespace()
    monkeypatch.setitem(ns['Controller'].execute.__globals__, 'guard_main', lambda plan: None)
    controller = ns['Controller']({}, tmp_path)
    assert controller.execute([sys.executable, '-c', 'raise SystemExit(2)'], 'selector',
                              cpu=True, allowed=(0, 2), poll_seconds=.01) == 2


def test_no_feasible_selection_finishes_without_completion_marker_or_promotion(tmp_path, monkeypatch):
    ns = namespace()
    module = ns['main'].__globals__
    root = tmp_path / 'interim'
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'immutable E35 fixture')
    plan = {'outdir': str(root), 'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
            'source_sha256': {}, 'prerequisite_sha256': {}, 'trainer_pid': 123,
            'floors': [0., .05], 'quality_thresholds': DEFAULT_THRESHOLDS}
    plan['commands'] = ns['commands'](plan)
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text(json.dumps(plan))
    real_read = ns['read']
    def read(path):
        if str(path).endswith('interim_epoch035_fixed_cfg/checkpoint_verification.json'):
            return {'step': 39130, 'checkpoint_sha256': plan['checkpoint_sha256'],
                    'all193_tensors_equal_default_ema_and_buffers': True}
        if str(path) == 'logs/postprocess_workflow_update/plan.json':
            return {'quality_thresholds': DEFAULT_THRESHOLDS, 'floors': [0., .05]}
        return real_read(path)
    monkeypatch.setitem(module, 'read', read)
    monkeypatch.setitem(module, 'guard_main', lambda plan: None)
    monkeypatch.setitem(module, 'readme_event', lambda *args, **kwargs: None)
    monkeypatch.setattr(module['subprocess'], 'check_output', lambda *args, **kwargs: '123\n')
    monkeypatch.setattr(module['signal'], 'signal', lambda *args, **kwargs: None)
    calls = []
    def execute(self, command, stage, **kwargs):
        calls.append(stage)
        calibration = root / 'calibration'
        if stage == 'validation_cfg_calibration':
            calibration.mkdir()
            for name in ('calibration_config.json', 'all_results.json', 'selected.json'):
                (calibration / name).write_text('{}')
            return 0
        if stage == 'validation_postprocessing':
            destination = calibration / 'postprocessing'
            destination.mkdir()
            (destination / 'review.json').write_text(json.dumps({'selected': None}))
            (destination / 'execution_config.json').write_text('{}')
            return 2
        raise AssertionError('An infeasible result must not be promoted or tested')
    monkeypatch.setattr(ns['Controller'], 'execute', execute)
    monkeypatch.setattr(sys, 'argv', ['controller', '--plan', str(plan_file)])
    ns['main']()
    completed = real_read(root / 'completed.json')
    assert not completed['any_original_gate_feasible_candidate']
    assert not completed['test_access'] and not completed['final_test_protocol_adopted']
    assert calls == ['validation_cfg_calibration', 'validation_postprocessing']
    assert not (root / 'calibration/postprocessing/completed.json').exists()
