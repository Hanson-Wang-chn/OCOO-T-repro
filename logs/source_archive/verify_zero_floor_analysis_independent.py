import copy, json, runpy, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from ocoot.runtime import file_sha256,timestamp
study=Path('runs/main_v1/interim_epoch015_full_conditional/zero_floor')
result=json.loads((study/'paired_comparison.json').read_text())
order=['PDCorr','DEOver','DEPrec','DirAgr','PDS_L1','PDS_L2','PDS_cos','MSE','MAE']
columns=['pearson_delta','overlap_at_N','precision_at_N','de_direction_match','discrimination_score_l1','discrimination_score_l2','discrimination_score_cosine','mse','mae']
rng=np.random.default_rng(9232)
frame=pd.read_csv(study/'floor_0/metrics/results.csv').sort_values('perturbation')
base=frame[columns].to_numpy()
draws=rng.integers(0,len(frame),(2000,len(frame)))
errors=[]
for comparison in result['comparisons_to_zero']:
    cand=pd.read_csv(study/comparison['candidate']/'metrics/results.csv').sort_values('perturbation')[columns].to_numpy()
    # Independent explicit resampling, without the production bootstrap helper.
    d=np.stack([np.nanmean(cand[indices],0)-np.nanmean(base[indices],0) for indices in draws])
    ci=np.quantile(d,[.025,.975],axis=0).T
    expected=np.array([row['ci95'] for row in comparison['rows']])
    errors.append(float(np.max(np.abs(ci-expected))))
    np.testing.assert_allclose(ci,expected,atol=1e-12,rtol=0)
    score=np.quantile(d[:,0]+d[:,[4,5,6]].mean(1),[.025,.975])
    np.testing.assert_allclose(score,comparison['primary_difference']['ci95'],atol=1e-12,rtol=0)
module=runpy.run_path('scripts/analyze_zero_floor.py')
rejected=[]
for kind in ['incomplete','changed_source','wrong_floor']:
    with tempfile.TemporaryDirectory(prefix='zero_floor_invalid_') as temporary:
        out=Path(temporary)
        for path in study.iterdir():
            if path.name not in ('study_config.json','study_state.json','comparison.json'):
                (out/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
        cfg=json.loads((study/'study_config.json').read_text())
        state=json.loads((study/'study_state.json').read_text())
        records=json.loads((study/'comparison.json').read_text())
        if kind=='incomplete':state['stage']='running'
        elif kind=='changed_source':cfg['time_independent_settings']['prediction_sha256']='0'*64
        else:cfg['time_independent_settings']['floors']=[0,.5]
        for filename,obj in [('study_config.json',cfg),('study_state.json',state),('comparison.json',records)]:
            (out/filename).write_text(json.dumps(obj))
        try:module['analyze'](out)
        except ValueError as exc:rejected.append({'case':kind,'exception':str(exc)})
        else:raise AssertionError('Invalid protocol accepted: '+kind)
record={'time':timestamp(),'source_sha256':file_sha256('scripts/analyze_zero_floor.py'),
        'real_cache':str(study),'cases':len(result['comparisons_to_zero']),
        'all9_paired_ci_independent_reference_max_abs_error':max(errors),
        'absolute_tolerance':1e-12,'rejected_invalid_protocols':rejected,
        'cuda_visible_devices':'','training_or_inference_defaults_changed':False}
Path('logs/zero_floor_analysis_verification.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
