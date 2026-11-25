"""Choose inference settings using validation only; preserve every raw sample.

Screen dual-CFG scales, compare the best scale's interval, then refine the two
best settings on all validation cells by default. Formal test evaluation is separate.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from ocoot.data import BenchmarkData
from ocoot.distributed import TrainingRanks
from ocoot.evaluate import generate,official_metrics
from ocoot.postprocess import primary_score
from ocoot.runtime import file_sha256,readme_event


def write_json_atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def refinement_cap(value):
    if value.lower()=='full':return None
    try:cap=int(value)
    except ValueError:raise argparse.ArgumentTypeError('refine-cap must be full or a positive integer') from None
    if cap<1:raise argparse.ArgumentTypeError('refine-cap must be full or a positive integer')
    return cap


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--outdir',required=True)
    p.add_argument('--screen-cap',type=int,default=16)
    p.add_argument('--refine-cap',type=refinement_cap,default=None,help='full (default), or a positive per-perturbation cap')
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--scales',default='0.6666666667,1,1.5,2')
    p.add_argument('--mse-weight', type=float, default=0.0,
                   help='MSE penalty in the validation selection score')
    p.add_argument('--compare-windows', action='store_true',
                   help='Screen each positive scale in two windows and refine one lower-MSE alternative')
    a=p.parse_args()
    if a.screen_cap<1 or a.batch_size<1:p.error('screen-cap and batch-size must be positive')
    if not math.isfinite(a.mse_weight) or a.mse_weight < 0:p.error('mse-weight must be nonnegative and finite')
    try:scales=[float(x) for x in a.scales.split(',')]
    except ValueError:p.error('scales must be comma-separated positive finite numbers')
    if not scales or any(not math.isfinite(w) or w<=0 for w in scales):
        p.error('formal dual-CFG scales must be positive and finite')
    if len(set(scales))!=len(scales):
        p.error('duplicate guidance scales')
    ranks=TrainingRanks()
    with ranks.initialize() if ranks.size>1 else nullcontext():
        calibrate(a,scales,ranks)


def calibrate(a, scales, ranks):
    torch.set_num_threads(8 if ranks.primary else 2)
    torch.backends.cuda.matmul.allow_tf32=True
    out=Path(a.outdir);out.mkdir(exist_ok=True,parents=True)
    s=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False);cfg=s['config']
    if cfg['training'].get('train_subset'):raise ValueError('Formal inference calibration requires a full-data model')
    if s.get('weights') in ['raw_pilot_only','raw_diagnostic']:raise ValueError('Formal calibration requires EMA inference weights')
    weights='ema_0.9996' if '0.9996' in s.get('ema',{}) else s.get('weights')
    if weights!='ema_0.9996':raise ValueError('Formal calibration requires the default EMA0.9996 weights')
    if ranks.primary:
        provenance={'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':file_sha256(a.checkpoint),'step':s['step'],
                    'weights':weights,'data_dir':cfg['data_dir'],
                    'screen_cap':a.screen_cap,'refine_cap':a.refine_cap,'batch_size':a.batch_size,'scales':a.scales,
                    'refinement_population':'all validation cells' if a.refine_cap is None else 'capped validation predictions',
                    'objective':'PDCorr + mean(PDS_L1,PDS_L2,PDS_cos)','data':'validation only',
                    'formal_guidance_policy':'literal_dual_cfg_positive_scale',
                    'source_sha256':{str(path):file_sha256(path) for path in [Path(__file__),
                                      Path('src/ocoot/model.py'),Path('src/ocoot/flow.py'),
                                      Path('src/ocoot/evaluate.py'),Path('src/ocoot/data.py'),
                                      Path('src/ocoot/postprocess.py'),
                                      Path('src/ocoot/runtime.py'),Path('environment.lock.txt')]},
                    'conditional_only_role':'diagnostic reference; excluded from formal refinement and selection',
                    'metric_protocol':'all nine paper metrics on nonnegative predictions; six array diagnostics on raw predictions',
                    'formal_postprocess':'clamp_min_0, matching the nonnegative expression domain; raw arrays are retained'}
        if a.mse_weight:
            provenance['objective'] += ' - weighted MSE'
            provenance['mse_weight'] = a.mse_weight
        if a.compare_windows:
            provenance['screen_windows']=[[.1,.9],[0.,1.]]
            provenance['refinement_policy']='Highest selection score plus lowest MSE among other cases within 0.04, or next highest score'
        fp=out/'calibration_config.json'
        if fp.exists() and json.loads(fp.read_text())!=provenance:raise ValueError('Calibration output already has different settings')
        write_json_atomic(fp,provenance)
    ranks.barrier()
    m=BenchmarkData(cfg['data_dir']);model=m.build_model(cfg['model']);model.load_state_dict(s['model'])
    if 'ema' in s:
        with torch.no_grad():
            for name,param in model.named_parameters():param.copy_(s['ema']['0.9996'][name])
    model=model.cuda().eval();del s
    forward=torch.compile(model,fullgraph=True)
    parallel_options={'ranks':ranks} if ranks.size>1 else {}
    records=[]

    def selection_score(record):
        return record.get('clamp0_selection_score', record['score'])

    def evaluate(w,low,high,cap,tag):
        case=out/tag;idx=m.evaluation_indices('validation',cap)
        raw=generate(forward,m,idx,case,noise_std=cfg['training']['noise_std'],batch_size=a.batch_size,
                     steps=50,guidance=w,cfg_min=low,cfg_max=high,clamp=False,**parallel_options)
        record=None
        if ranks.primary:
            raw_metrics=official_metrics(m,idx,raw,case/'metrics_raw')
            metrics=official_metrics(m,idx,np.maximum(raw,0),case/'metrics_clamped',de=True)
            cache={'directory':str(case.resolve()),'raw_unclamped':True,'split':'validation',
                   'files':{name:file_sha256(case/name) for name in
                            ('prediction.npy','indices.npy','generation_state.json')}}
            record={'tag':tag,'guidance':w,'cfg_min':low,'cfg_max':high,'clamp':True,'cap':cap,
                    'metrics':metrics,'raw_metrics':raw_metrics,'score':primary_score(metrics),'cache_provenance':cache}
            if a.mse_weight:
                record['clamp0_selection_score'] = record['score'] - a.mse_weight * metrics['MSE']
            write_json_atomic(out/'all_results.json',[*records,record])
            readme_event(f'验证集 CFG 校准 `{tag}`：w={w}，区间 [{low},{high}]，PDCorr={metrics["PDCorr"]:.4f}，PDS_L2={metrics["PDS_L2"]:.4f}，MSE={metrics["MSE"]:.5f}（非测试）。')
        record=ranks.from_primary(record)
        records.append(record)
        return record
    candidates=[None]+scales
    for i,w in enumerate(candidates):evaluate(w,.1,.9,a.screen_cap,f'screen_{i:02d}')
    guided=[r for r in records if r['guidance'] is not None]
    best_guided=max(guided,key=selection_score)
    if a.compare_windows:
        for i,w in enumerate(scales,1):evaluate(w,0.,1.,a.screen_cap,f'screen_full_{i:02d}')
    else:
        evaluate(best_guided['guidance'],0.,1.,a.screen_cap,'screen_full_interval')
    # Section 3.3 specifies dual CFG at inference. The ordinary conditional
    # field remains a diagnostic reference, including when it scores higher.
    eligible=sorted((r for r in records if r['guidance'] is not None),key=selection_score,reverse=True)
    top=eligible[:2]
    if a.compare_windows:
        near=[r for r in eligible[1:] if selection_score(r)>=selection_score(eligible[0])-.04]
        if near:
            top[1]=min(near,key=lambda r:(r['metrics']['MSE'],-selection_score(r)))
    refined=[]
    for i,r in enumerate(top):refined.append(evaluate(r['guidance'],r['cfg_min'],r['cfg_max'],a.refine_cap,f'refine_{i:02d}'))
    if not ranks.primary:
        return
    selected=max(refined,key=selection_score)
    selected.update(checkpoint=provenance['checkpoint'],checkpoint_sha256=provenance['checkpoint_sha256'],step=provenance['step'],
                    weights=weights,data_dir=cfg['data_dir'],
                    noise_std=cfg['training']['noise_std'],sampling_steps=50,seed=4242,batch_size=a.batch_size,
                    formal_guidance_policy=provenance['formal_guidance_policy'],
                    source_sha256=provenance['source_sha256'],
                    conditional_reference_tag=records[0]['tag'],
                    note='Locked using validation only within the literal dual-CFG family; the conditional-only field is diagnostic. '
                         f'Final two candidates used {provenance["refinement_population"]}. Run full validation DE and full test with these settings.')
    if a.mse_weight:
        selected['mse_weight'] = a.mse_weight
    write_json_atomic(out/'selected.json',selected)
    readme_event(f'完成验证集采样校准。两组全量候选和校准选择见 `{out}/selected.json` 与 `{out}/all_results.json`。')
    print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()

