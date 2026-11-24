"""Resumable generation and Cell Eval on a locked checkpoint and split."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from ocoot.data import BenchmarkData
from ocoot.distributed import TrainingRanks
from ocoot.evaluate import generate, official_metrics
from ocoot.runtime import readme_event,file_sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--outdir',required=True);p.add_argument('--cap',type=int);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--guidance',type=float);p.add_argument('--cfg-min',type=float,default=.1);p.add_argument('--cfg-max',type=float,default=.9)
    p.add_argument('--clamp',action='store_true');p.add_argument('--de',action='store_true');p.add_argument('--no-compile',action='store_true')
    p.add_argument('--ema-decay',default='0.9996');a=p.parse_args()
    ranks = TrainingRanks()
    with ranks.initialize() if ranks.size > 1 else nullcontext():
        evaluate(a, ranks)


def evaluate(a, ranks):
    torch.set_num_threads(8 if ranks.primary else 2)
    torch.backends.cuda.matmul.allow_tf32=True
    state=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False);cfg=state['config']
    if a.split=='test' and cfg['training'].get('train_subset'):
        raise ValueError('Subset pilot checkpoints are diagnostic; test evaluation requires a full-data model')
    data=BenchmarkData(cfg['data_dir']);model=data.build_model(cfg['model'])
    model.load_state_dict(state['model'])
    if 'ema' in state:
        with torch.no_grad():
            for name,param in model.named_parameters():param.copy_(state['ema'][a.ema_decay][name])
        weights=f'ema_{a.ema_decay}'
    else:weights=state.get('weights','saved_inference_weights')
    if a.split=='test' and weights in ['raw_pilot_only','raw_diagnostic']:raise ValueError('Diagnostic raw weights are not a reproduction checkpoint')
    out=Path(a.outdir)
    if ranks.primary:
        provenance={'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':file_sha256(a.checkpoint),
                    'step':state['step'],'weights':weights,'split':a.split,'cap':a.cap,'guidance':a.guidance,
                    'cfg_min':a.cfg_min,'cfg_max':a.cfg_max,'clamp':a.clamp,'de':a.de,
                    'seed':4242,'steps':50,'batch_size':a.batch_size,'noise_std':cfg['training']['noise_std'],'model':cfg['model']}
        out.mkdir(exist_ok=True,parents=True)
        fp=out/'evaluation_config.json'
        if fp.exists() and json.loads(fp.read_text())!=provenance:raise ValueError('Evaluation folder has a different configuration')
        temporary=fp.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(provenance,indent=2))
        temporary.replace(fp)
    del state
    ranks.barrier()
    model=model.cuda().eval()
    forward=torch.compile(model,fullgraph=True) if not a.no_compile else model
    indices=data.evaluation_indices(a.split,a.cap)
    pred=generate(forward,data,indices,out,noise_std=cfg['training']['noise_std'],batch_size=a.batch_size,steps=50,
                  guidance=a.guidance,cfg_min=a.cfg_min,cfg_max=a.cfg_max,clamp=a.clamp,ranks=ranks)
    if ranks.primary:
        summary=official_metrics(data,indices,pred,out/'metrics',de=a.de)
        print(json.dumps(summary,indent=2),flush=True)
        readme_event(f'完成 `{a.outdir}` 的 {a.split} 评估（细胞 {len(indices):,}，weights={weights}，CFG={a.guidance}）；指标见 `{out}/metrics/summary.json`。PDCorr={summary["PDCorr"]:.4f}，MSE={summary["MSE"]:.5f}。')
    ranks.barrier()


if __name__=='__main__':main()

