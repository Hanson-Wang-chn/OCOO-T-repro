"""Complete a validation cache from its durable prefix on the available GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import generate
from ocoot.flow import sample
from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp
from prefetch_validation import CALIBRATION, WORK, load, read


CASE = CALIBRATION / 'refine_00'
ROOT = WORK / 'refine_00_suffix'
LOG = Path(__file__).parent


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verified_prefix(plan):
    path = CASE / 'generation_state.json'
    marker = read(path)
    if marker != plan['prefix_marker'] or file_sha256(CASE/'prediction.npy') != plan['prefix_sha256']:
        raise ValueError('Existing validation prefix changed')
    return marker


def worker(part):
    plan = read(ROOT/'plan.json')
    marker = verified_prefix(plan)
    protocol, cfg, data, model = load()
    if plan['checkpoint_sha256'] != protocol['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    indices = data.evaluation_indices('validation',None)
    batch = marker['options']['batch_size']
    options = {k:v for k,v in marker['options'].items() if k!='batch_size'}
    x = indices[:batch]
    p,c,b = [torch.from_numpy(data.meta[k][x]).cuda() for k in ('pert','cell','batch')]
    pred = sample(model,p,c,b,data.x.shape[1],**options).cpu().numpy()
    reference = np.load(CASE/'prediction.npy',mmap_mode='r')[:batch]
    np.testing.assert_array_equal(pred,reference)
    start,end=plan['ranges'][part]
    case=ROOT/f'part_{part}'
    options={**marker['options'],'seed':marker['options']['seed']+start}
    generated=generate(model,data,indices[start:end],case,**options)
    assert np.isfinite(generated).all()
    assert read(case/'generation_state.json')=={'completed':end-start,'total':end-start,'options':options}
    write(case/'proof.json',{'time':timestamp(),'range':[start,end],'options':options,
                            'bitwise_prefix_batch_match':True,'checkpoint_sha256':plan['checkpoint_sha256'],
                            'files':{k:file_sha256(case/k) for k in ('indices.npy','prediction.npy','generation_state.json')}})


def merge(plan, indices, genes):
    marker=verified_prefix(plan)
    staged=ROOT/'complete'
    staged.mkdir(exist_ok=True)
    prediction=np.lib.format.open_memmap(staged/'prediction.npy',mode='w+',dtype=np.float32,shape=(len(indices),genes))
    prefix=marker['completed']
    prediction[:prefix]=np.load(CASE/'prediction.npy',mmap_mode='r')[:prefix]
    assert np.isfinite(prediction[:prefix]).all()
    proofs=[]
    for part,(start,end) in enumerate(plan['ranges']):
        shard=ROOT/f'part_{part}'; proof=read(shard/'proof.json')
        expected={**marker['options'],'seed':marker['options']['seed']+start}
        if (proof['range']!=[start,end] or proof['options']!=expected
                or proof['checkpoint_sha256']!=plan['checkpoint_sha256']
                or read(shard/'generation_state.json')!={'completed':end-start,'total':end-start,'options':expected}):
            raise ValueError('Refinement shard provenance differs')
        for name,digest in proof['files'].items():
            if file_sha256(shard/name)!=digest:
                raise ValueError('Completed shard changed')
        np.testing.assert_array_equal(np.load(shard/'indices.npy'),indices[start:end])
        values=np.load(shard/'prediction.npy',mmap_mode='r')
        assert values.dtype==np.float32 and values.shape==(end-start,genes) and np.isfinite(values).all()
        prediction[start:end]=values
        proofs.append(proof)
    prediction.flush(); del prediction
    np.save(staged/'indices.npy',indices)
    write(staged/'generation_state.json',{**marker,'completed':len(indices)})
    write(staged/'prefetch_proof.json',{'time':timestamp(),'prefix':prefix,'test_access':False,
         'plan_sha256':file_sha256(ROOT/'plan.json'),'shards':proofs,
         'files':{name:file_sha256(staged/name) for name in ('prediction.npy','indices.npy','generation_state.json')}})
    verified_prefix(plan)
    archive=ROOT/'serial_prefix'
    if archive.exists():
        raise FileExistsError('The original calibration prefix is already archived')
    CASE.rename(archive)
    try:
        staged.rename(CASE)
    except BaseException:
        archive.rename(CASE)
        raise
    readme_event(f'第一组正式全量验证已保留前 {prefix} 个样本并用四卡完成剩余批次。四张卡的 w2/[0,1] 首批逐元素一致，按原索引及种子偏移合并 4825 个 raw 样本。恢复正式校准评分。')


def active(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().rsplit(')',1)[1].split()[0]!='Z'


def repartition():
    state=read(CALIBRATION.parent/'pipeline_state.json')
    parent=state['pid']
    if state['stage']!='calibrating_validation' or not active(parent):
        raise RuntimeError('Expected a running original calibration')
    candidates=Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
    child=next((int(pid) for pid in candidates if b'scripts/calibrate.py' in Path(f'/proc/{pid}/cmdline').read_bytes()),None)
    if child is None or not (CALIBRATION/'refine_01/prefetch_proof.json').exists():
        raise RuntimeError('Could not confirm the active serial first refinement and completed second case')
    before=read(CASE/'generation_state.json')
    protocol=read(CALIBRATION/'calibration_config.json')
    if before['options']!={'noise_std':1.,'batch_size':32,'steps':50,'guidance':2.,'cfg_min':0.,'cfg_max':1.,'clamp':False,'seed':4242} or not 512<=before['completed']<=3000 or before['total']!=4825:
        raise ValueError('Unexpected first-refinement protocol or progress')
    write(ROOT/'intent.json',{'time':timestamp(),'parent_pid':parent,'child_pid':child,'reason':'repartition validation suffix; preserve committed prefix','marker':before,'test_access':False})
    os.kill(child,signal.SIGTERM)
    deadline=time.monotonic()+90
    while active(child) or active(parent):
        if time.monotonic()>deadline:
            raise TimeoutError('Original serial calibration did not exit')
        time.sleep(.5)
    marker=read(CASE/'generation_state.json')
    assert marker['options']==before['options'] and marker['total']==before['total']
    data=BenchmarkData(protocol['data_dir']); indices=data.evaluation_indices('validation',None)
    np.testing.assert_array_equal(np.load(CASE/'indices.npy'),indices)
    batch=marker['options']['batch_size']; n=len(indices); first=marker['completed']
    assert first%batch==0 and n==4825
    blocks=np.array_split(np.arange(first//batch,(n+batch-1)//batch),4)
    ranges=[[int(block[0])*batch,min((int(block[-1])+1)*batch,n)] for block in blocks]
    plan={'prefix_marker':marker,'prefix_sha256':file_sha256(CASE/'prediction.npy'),
          'checkpoint_sha256':protocol['checkpoint_sha256'],'ranges':ranges,'test_access':False}
    write(ROOT/'plan.json',plan)
    children=[]
    try:
        for part in range(4):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(part))
            with (ROOT/f'part_{part}.log').open('a') as log:
                children.append(subprocess.Popen([sys.executable,__file__,'--part',str(part)],env=env,
                                                    stdout=log,stderr=subprocess.STDOUT))
        for child in children:
            if child.wait():
                raise RuntimeError(f'Parallel refinement worker exited {child.returncode}')
        merge(plan,indices,data.x.shape[1])
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate(); child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',type=int,choices=range(4))
    args=parser.parse_args()
    if args.part is not None:
        worker(args.part)
        return
    with exclusive_run(ROOT,'repartition'):
        repartition()


if __name__=='__main__':
    main()
