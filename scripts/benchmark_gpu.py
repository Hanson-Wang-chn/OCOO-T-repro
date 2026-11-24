"""Real architecture throughput and memory; does not tune scientific settings."""
import argparse
import json
import time
from pathlib import Path
import torch
from ocoot.model import OCOOT, ModelConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=16);p.add_argument('--checkpoint',action='store_true');p.add_argument('--compile',action='store_true');p.add_argument('--compile-mode',default='default');p.add_argument('--steps',type=int,default=12);a=p.parse_args()
    if a.steps<4:raise ValueError('Need warmup and at least two timed steps')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(991);torch.cuda.manual_seed_all(991)
    m=OCOOT(ModelConfig(grad_checkpoint=a.checkpoint),torch.randn(2100,5120),torch.arange(2000),4,56).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0,fused=True)
    f=torch.compile(m,fullgraph=True,mode=a.compile_mode) if a.compile else m
    x=torch.randn(a.batch,2000,device='cuda');t=torch.rand(a.batch,device='cuda');i=torch.zeros(a.batch,device='cuda',dtype=torch.long)
    ts=[]
    for s in range(a.steps):
        torch.cuda.synchronize();start=time.monotonic()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=(f(x,t,i,i,i).float()-x).square().mean()
        loss.backward();opt.step();torch.cuda.synchronize();ts.append(time.monotonic()-start)
        print(json.dumps({'step':s,'sec':ts[-1],'loss':loss.item(),'peak_GB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
    result={'batch':a.batch,'checkpoint':a.checkpoint,'compile':a.compile,'compile_mode':a.compile_mode,'seconds':ts,'cells_per_sec':a.batch/(sum(ts[2:])/len(ts[2:])), 'peak_GB':torch.cuda.max_memory_allocated()/1e9,'parameters':sum(p.numel() for p in m.parameters())}
    Path('runs/benchmarks').mkdir(exist_ok=True,parents=True)
    suffix=f'_{a.compile_mode}' if a.compile_mode!='default' else ''
    Path(f'runs/benchmarks/b{a.batch}_cp{a.checkpoint}_compile{a.compile}{suffix}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()

