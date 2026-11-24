"""Generate the registered second full refinement on idle devices in B32 shards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ocoot.data import BenchmarkData
from ocoot.evaluate import generate
from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp
from prefetch_validation import CALIBRATION, WORK, load, read, verify_device


ROOT = WORK / 'refine_01_shards'
STAGED = WORK / 'refine_01_complete'
SLOTS = ('1', '2', '3')
SCREEN_TAGS = ('screen_00', 'screen_01', 'screen_02', 'screen_03', 'screen_04', 'screen_full_interval')


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def registered_case():
    current = CALIBRATION / 'all_results.json'
    if not current.exists():
        return None
    rows = read(current)
    if tuple(x['tag'] for x in rows[:6]) != SCREEN_TAGS:
        return None
    guided = [r for r in rows[:6] if r['guidance'] is not None]
    # Match Python's stable sort and the registered primary objective.
    return sorted(guided, key=lambda row: row['score'], reverse=True)[1]


def run_shard(part):
    plan = read(ROOT / 'plan.json')
    protocol, cfg, data, model = load()
    if protocol['checkpoint_sha256'] != plan['checkpoint_sha256'] or plan['options']['noise_std'] != cfg['training']['noise_std']:
        raise ValueError('The refinement has another checkpoint or noise scale')
    check = verify_device(protocol, cfg, data, model)
    start, end = plan['ranges'][part]
    case = ROOT / f'part_{part}'
    indices = data.evaluation_indices('validation', None)[start:end]
    options = {**plan['options'], 'seed': plan['options']['seed'] + start}
    result = generate(model, data, indices, case, **options)
    marker = read(case / 'generation_state.json')
    assert marker == {'completed':len(indices),'total':len(indices),'options':options}
    assert np.isfinite(result).all()
    write(case / 'shard_proof.json', {'time':timestamp(),'range':[start,end],'options':options,
          'device_match':check,'checkpoint_sha256':plan['checkpoint_sha256'],
          'files':{name:file_sha256(case/name) for name in ('prediction.npy','indices.npy','generation_state.json')}})


def merge(plan, data, indices):
    if registered_case() != plan['screen_record'] or file_sha256(CALIBRATION/'calibration_config.json') != plan['calibration_config_sha256']:
        raise ValueError('Registered inference selection changed during prefetch')
    STAGED.mkdir(exist_ok=True)
    result = np.lib.format.open_memmap(STAGED/'prediction.npy',mode='w+',dtype=np.float32,
                                       shape=(len(indices),data.x.shape[1]))
    proofs = []
    for part,(start,end) in enumerate(plan['ranges']):
        case = ROOT / f'part_{part}'
        proof = read(case/'shard_proof.json')
        expected = {**plan['options'],'seed':plan['options']['seed']+start}
        if (proof['range']!=[start,end] or proof['checkpoint_sha256']!=plan['checkpoint_sha256']
                or proof['options']!=expected or read(case/'generation_state.json') != {
                    'completed':end-start,'total':end-start,'options':expected}):
            raise ValueError('Inconsistent refinement shard provenance')
        for name,digest in proof['files'].items():
            if file_sha256(case/name)!=digest:
                raise ValueError(f'Shard changed: {case/name}')
        np.testing.assert_array_equal(np.load(case/'indices.npy'),indices[start:end])
        block=np.load(case/'prediction.npy',mmap_mode='r')
        if block.dtype!=np.float32 or block.shape!=result[start:end].shape or not np.isfinite(block).all():
            raise ValueError('Invalid prediction array in a completed shard')
        result[start:end]=block
        proofs.append(proof)
    result.flush()
    del result
    np.save(STAGED/'indices.npy',indices)
    write(STAGED/'generation_state.json',{'completed':len(indices),'total':len(indices),'options':plan['options']})
    write(STAGED/'prefetch_proof.json',{'time':timestamp(),'plan_sha256':file_sha256(ROOT/'plan.json'),
          'validation_only':True,'shards':proofs,
          'files':{name:file_sha256(STAGED/name) for name in ('prediction.npy','indices.npy','generation_state.json')}})
    first = CALIBRATION/'refine_00/generation_state.json'
    if not first.exists() or read(first)['completed'] > len(indices)-512:
        raise RuntimeError('Primary refinement is too close to the next case; keep independent cache intact')
    target = CALIBRATION/'refine_01'
    subprocess.run(['mv','--no-clobber','--no-target-directory',str(STAGED),str(target)],check=True)
    if STAGED.exists() or not (target/'prefetch_proof.json').exists():
        raise FileExistsError('Primary calibration already created the second refinement')
    readme_event('正式校准的第二个完整验证候选已由三张空闲 GPU 生成并逐段核验，原校准流程将对同一 raw 缓存计算指标。原 B32 批次、全局种子偏移和双 CFG 选型保持。')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',type=int,choices=range(len(SLOTS)))
    a=p.parse_args()
    if a.part is not None:
        run_shard(a.part)
        return
    with exclusive_run(ROOT,'prefetch'):
        while (case:=registered_case()) is None:
            pipeline=read(CALIBRATION.parent/'pipeline_state.json')
            if pipeline['stage']!='calibrating_validation':
                raise RuntimeError('Primary validation calibration has exited')
            time.sleep(15)
        protocol_path=CALIBRATION/'calibration_config.json'
        protocol=read(protocol_path)
        data=BenchmarkData(protocol['data_dir'])
        indices=data.evaluation_indices('validation',None)
        batch=protocol['batch_size']
        assert len(indices)==4825 and batch==32 and protocol['refine_cap'] is None
        blocks=np.array_split(np.arange((len(indices)+batch-1)//batch),len(SLOTS))
        ranges=[[int(block[0])*batch,min((int(block[-1])+1)*batch,len(indices))] for block in blocks]
        plan={'checkpoint_sha256':protocol['checkpoint_sha256'],'calibration_config_sha256':file_sha256(protocol_path),
              'ranges':ranges,'screen_record':case,
              'options':{'noise_std':1.,'batch_size':batch,'steps':50,'guidance':case['guidance'],
                         'cfg_min':case['cfg_min'],'cfg_max':case['cfg_max'],'clamp':False,'seed':4242}}
        path=ROOT/'plan.json'
        if path.exists() and read(path)!=plan:
            raise ValueError('A different refinement plan is already registered')
        write(path,plan)
        children=[]
        try:
            for part,device in enumerate(SLOTS):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
                with (ROOT/f'part_{part}.log').open('a') as log:
                    children.append(subprocess.Popen([sys.executable,__file__,'--part',str(part)],env=env,
                                                        stdout=log,stderr=subprocess.STDOUT))
            for child in children:
                if child.wait():
                    raise RuntimeError(f'Prefetch worker exited {child.returncode}')
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
        merge(plan,data,indices)


if __name__=='__main__':
    main()
