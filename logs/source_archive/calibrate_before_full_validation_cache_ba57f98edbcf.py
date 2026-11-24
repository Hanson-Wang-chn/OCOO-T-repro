"""Choose inference settings using validation only; preserve every raw sample.

Screen dual-CFG scales, compare the best scale's interval, then refine the two
best settings on all validation cells by default. Formal test evaluation is separate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from ocoot.data import BenchmarkData
from ocoot.evaluate import generate,official_metrics
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


def score(metrics):
    return metrics['PDCorr']+sum(metrics[k] for k in ['PDS_L1','PDS_L2','PDS_cos'])/3


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
    a=p.parse_args();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    if a.screen_cap<1 or a.batch_size<1:p.error('screen-cap and batch-size must be positive')
    try:scales=[float(x) for x in a.scales.split(',')]
    except ValueError:p.error('scales must be comma-separated positive finite numbers')
    if not scales or any(not math.isfinite(w) or w<=0 for w in scales):
        p.error('formal dual-CFG scales must be positive and finite')
    out=Path(a.outdir);out.mkdir(exist_ok=True,parents=True)
    s=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False);cfg=s['config']
    if cfg['training'].get('train_subset'):raise ValueError('Formal inference calibration requires a full-data model')
    if s.get('weights') in ['raw_pilot_only','raw_diagnostic']:raise ValueError('Formal calibration requires EMA inference weights')
    provenance={'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':file_sha256(a.checkpoint),'step':s['step'],
                'screen_cap':a.screen_cap,'refine_cap':a.refine_cap,'batch_size':a.batch_size,'scales':a.scales,
                'refinement_population':'all validation cells' if a.refine_cap is None else 'capped validation predictions',
                'objective':'PDCorr + mean(PDS_L1,PDS_L2,PDS_cos)','data':'validation only',
                'formal_guidance_policy':'literal_dual_cfg_positive_scale',
                'source_sha256':{str(path):file_sha256(path) for path in [Path(__file__),
                                  Path('src/ocoot/model.py'),Path('src/ocoot/flow.py'),
                                  Path('src/ocoot/evaluate.py'),Path('src/ocoot/data.py'),
                                  Path('src/ocoot/runtime.py'),Path('environment.lock.txt')]},
                'conditional_only_role':'diagnostic reference; excluded from formal refinement and selection',
                'metric_protocol':'all nine paper metrics on nonnegative predictions; six array diagnostics on raw predictions',
                'formal_postprocess':'clamp_min_0, matching the nonnegative expression domain; raw arrays are retained'}
    fp=out/'calibration_config.json'
    if fp.exists() and json.loads(fp.read_text())!=provenance:raise ValueError('Calibration output already has different settings')
    write_json_atomic(fp,provenance)
    m=BenchmarkData(cfg['data_dir']);model=m.build_model(cfg['model']);model.load_state_dict(s['model'])
    if 'ema' in s:
        with torch.no_grad():
            for name,param in model.named_parameters():param.copy_(s['ema']['0.9996'][name])
    model=model.cuda().eval();del s
    forward=torch.compile(model,fullgraph=True)
    records=[]
    def evaluate(w,low,high,cap,tag):
        case=out/tag;idx=m.evaluation_indices('validation',cap)
        raw=generate(forward,m,idx,case,noise_std=cfg['training']['noise_std'],batch_size=a.batch_size,
                     steps=50,guidance=w,cfg_min=low,cfg_max=high,clamp=False)
        raw_metrics=official_metrics(m,idx,raw,case/'metrics_raw')
        # DE is inexpensive relative to generation and can fail independently
        # of pseudobulk scores. Record it during screening, keeping the locked
        # primary selection objective and the full-validation gate unchanged.
        metrics=official_metrics(m,idx,np.maximum(raw,0),case/'metrics_clamped',de=True)
        record={'tag':tag,'guidance':w,'cfg_min':low,'cfg_max':high,'clamp':True,'cap':cap,
                'metrics':metrics,'raw_metrics':raw_metrics,'score':score(metrics)}
        records.append(record);write_json_atomic(out/'all_results.json',records)
        readme_event(f'验证集 CFG 校准 `{tag}`：w={w}，区间 [{low},{high}]，PDCorr={metrics["PDCorr"]:.4f}，PDS_L2={metrics["PDS_L2"]:.4f}，MSE={metrics["MSE"]:.5f}（非测试）。')
        return record
    candidates=[None]+scales
    for i,w in enumerate(candidates):evaluate(w,.1,.9,a.screen_cap,f'screen_{i:02d}')
    guided=[r for r in records if r['guidance'] is not None]
    best_guided=max(guided,key=lambda r:r['score'])
    evaluate(best_guided['guidance'],0.,1.,a.screen_cap,'screen_full_interval')
    # Section 3.3 specifies dual CFG at inference. The ordinary conditional
    # field remains a diagnostic reference, including when it scores higher.
    eligible=[r for r in records if r['guidance'] is not None]
    top=sorted(eligible,key=lambda r:r['score'],reverse=True)[:2]
    refined=[]
    for i,r in enumerate(top):refined.append(evaluate(r['guidance'],r['cfg_min'],r['cfg_max'],a.refine_cap,f'refine_{i:02d}'))
    selected=max(refined,key=lambda r:r['score'])
    selected.update(checkpoint=provenance['checkpoint'],checkpoint_sha256=provenance['checkpoint_sha256'],step=provenance['step'],
                    noise_std=cfg['training']['noise_std'],sampling_steps=50,seed=4242,batch_size=a.batch_size,
                    formal_guidance_policy=provenance['formal_guidance_policy'],
                    source_sha256=provenance['source_sha256'],
                    conditional_reference_tag=records[0]['tag'],
                    note='Locked using validation only within the literal dual-CFG family; the conditional-only field is diagnostic. '
                         f'Final two candidates used {provenance["refinement_population"]}. Run full validation DE and full test with these settings.')
    write_json_atomic(out/'selected.json',selected)
    readme_event(f'完成验证集采样校准；设置已锁定于 `{out}/selected.json`，尚未据此计算测试结果。')
    print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()

