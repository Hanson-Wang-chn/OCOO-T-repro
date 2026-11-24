"""Reports must label optional validation-selected processing transparently."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from ocoot.evaluate import METRICS
from ocoot.runtime import file_sha256


@pytest.mark.parametrize('floor', [0., .05])
def test_report_preserves_baseline_and_labels_the_optional_choice(tmp_path, monkeypatch, floor):
    ev = tmp_path / 'evaluation'
    baseline = tmp_path / 'baseline'
    for directory in (ev, baseline):
        (directory / 'metrics').mkdir(parents=True)
    values = {'PDCorr': .4, 'MSE': .009, 'MAE': .06, 'PDS_L1': .8, 'PDS_L2': .82,
              'PDS_cos': .81, 'DEOver': .21, 'DEPrec': .22, 'DirAgr': .76}
    frame = pd.DataFrame({'perturbation': [f'P{i:03d}' for i in range(380)],
        **{key: np.linspace(values[display] - .001, values[display] + .001, 380) for key, display in METRICS.items()}})
    frame.to_csv(ev / 'metrics/results.csv', index=False)
    (ev / 'metrics/population_counts.json').write_text(json.dumps({
        'real_perturbed_cells': 26878, 'generated_cells': 26878, 'shared_real_controls': 4976}))
    common = {'split': 'test', 'cap': None, 'de': True, 'clamp': True,
              'guidance': 1.5, 'cfg_min': .1, 'cfg_max': .9, 'weights': 'ema_0.9996'}
    config = {**common, 'zero_floor': floor}
    if floor:
        (baseline / 'evaluation_config.json').write_text(json.dumps(common))
        (baseline / 'metrics/summary.json').write_text(json.dumps(values))
        config.update(baseline_evaluation_dir=str(baseline), postprocess_selection_sha256='frozen_before_test')
    (ev / 'evaluation_config.json').write_text(json.dumps(config))
    if floor:
        files = [ev / 'evaluation_config.json', ev / 'metrics/results.csv',
                 ev / 'metrics/population_counts.json', baseline / 'evaluation_config.json',
                 baseline / 'metrics/summary.json']
        (ev / 'postprocess_test_completed.json').write_text(json.dumps({
            'protocol': {'zero_floor': floor, 'selection_sha256': 'frozen_before_test'},
            'selection_changed_after_test': False,
            'output_sha256': {str(p): file_sha256(p) for p in files}}))
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/report.py'))
    out = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['report.py', '--evaluation-dir', str(ev), '--outdir', str(out)])
    namespace['main']()
    text = (out / 'REPORT.md').read_text()
    comparison = pd.read_csv(out / 'benchmark_comparison.csv')
    assert len(comparison) == 9 and (comparison['valid_perturbations'] == 380).all()
    assert (out / 'benchmark_comparison.png').exists() and (out / 'benchmark_comparison.pdf').exists()
    if floor:
        assert '原文未公开这一处理细节' in text and '未根据测试结果选择阈值' in text
        assert 'baseline_clamp0' in comparison and '原始clamp0' in text
    else:
        assert 'baseline_clamp0' not in comparison and '原文未公开这一处理细节' not in text


def test_positive_floor_report_refuses_missing_frozen_test_provenance(tmp_path, monkeypatch):
    ev = tmp_path / 'evaluation'
    (ev / 'metrics').mkdir(parents=True)
    (ev / 'evaluation_config.json').write_text(json.dumps({'split': 'test', 'cap': None, 'de': True, 'zero_floor': .05}))
    pd.DataFrame({'perturbation': [f'P{i}' for i in range(380)]}).to_csv(ev / 'metrics/results.csv', index=False)
    (ev / 'metrics/population_counts.json').write_text(json.dumps({'real_perturbed_cells': 26878, 'generated_cells': 26878}))
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/report.py'))
    out = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['report.py', '--evaluation-dir', str(ev), '--outdir', str(out)])
    with pytest.raises(FileNotFoundError): namespace['main']()
    assert not (out / 'REPORT.md').exists()
