"""Single-A100 trainer with exact effective batch size and restartable state."""
import contextlib
import json
import hashlib
import math
from pathlib import Path
import signal
import time
import tarfile
import numpy as np
import torch
import yaml
from .data import BenchmarkData
from .flow import flow_batch
from .runtime import DualEMA, atomic_save, log_event, readme_event, seed_all, rng_state, restore_rng, timestamp, update_readme_live,exclusive_run


@contextlib.contextmanager
def ema_parameters(model, ema):
    saved={n:p.detach().clone() for n,p in model.named_parameters()}
    ema.copy_to(model)
    try: yield
    finally:
        with torch.no_grad():
            for n,p in model.named_parameters():p.copy_(saved[n])


def update_live_status(run_dir, event):
    update_readme_live(f'实时训练：`{run_dir}`，epoch **{event["epoch"]:.3f}**，优化步 **{event["step"]}**，'
                       f'loss **{event["loss"]:.5f}**，约 **{event["cells_per_sec"]:.1f} cells/s**，更新于 {event["time"]}。')


@torch.no_grad()
def validate(model, forward, data, indices, config):
    model.eval();stats=[]
    # A fixed RNG isolates validation from changes in training RNG.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(9821);torch.cuda.manual_seed_all(9821)
        for a in range(0,len(indices),config['micro_batch']):
            x,p,c,b=data.batch(indices[a:a+config['micro_batch']])
            z,t,v=flow_batch(x,config['noise_std'],config['p_mean'],config['p_std'])
            with torch.autocast('cuda',dtype=torch.bfloat16):vhat=forward(z,t,p,c,b).float()
            stats.append((len(x),float((vhat-v).square().mean()),float(((z+(1-t[:,None])*vhat)-x).square().mean())))
    model.train()
    return {key:sum(n*s[i] for n,*s in stats)/sum(x[0] for x in stats) for i,key in enumerate(['velocity_mse','endpoint_mse'])}


def train(config, resume=True):
    with exclusive_run(config['run_dir'],'training'):
        return _train_locked(config,resume)


def _train_locked(config, resume=True):
    tr=config['training'];run=Path(config['run_dir']);run.mkdir(parents=True,exist_ok=True)
    if not resume and ((run/'last.pt').exists() or (run/'finished.json').exists()):
        raise FileExistsError('Fresh training requires a new run_dir; existing experiment artifacts are retained')
    if resume and (run/'finished.json').exists():
        if yaml.safe_load((run/'config.yaml').read_text())!=config:
            raise ValueError('Completed run config differs; use a new run directory')
        return
    import os
    (run/'training.pid').write_text(str(os.getpid()))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    seed_all(config['seed'])
    data=BenchmarkData(config['data_dir'])
    indices=data.train.copy()
    if tr.get('train_subset'):
        indices=np.sort(np.random.default_rng(config['seed']).choice(indices,tr['train_subset'],replace=False))
    np.save(run/'train_indices.npy',indices)
    model=data.build_model(config['model']).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),weight_decay=tr['weight_decay'],fused=True)
    ema=DualEMA(model,tuple(tr['ema_decays']))
    effective=tr['effective_batch'];micro=tr['micro_batch']
    assert effective%micro==0
    steps_per_epoch=len(indices)//effective
    assert steps_per_epoch>0
    total_steps=steps_per_epoch*tr['epochs'];warmup=steps_per_epoch*tr['warmup_epochs']
    step=0;best=float('inf');best_generation=-float('inf');best_generation_epoch=0
    early_reference=-float('inf');early_reference_epoch=0
    pending_generation_epoch=None
    saved_config=run/'config.yaml'
    if resume and (run/'last.pt').exists():
        checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['config']!=config:raise ValueError('Resume config differs; use a new run directory')
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        ema.load_state_dict(checkpoint['ema'],'cuda')
        step=checkpoint['step'];best=checkpoint['best'];best_generation=checkpoint.get('best_generation',-float('inf'))
        best_generation_epoch=checkpoint.get('best_generation_epoch',0)
        early_reference=checkpoint.get('early_reference',best_generation)
        early_reference_epoch=checkpoint.get('early_reference_epoch',best_generation_epoch)
        pending_generation_epoch=checkpoint.get('pending_generation_epoch')
        restore_rng(checkpoint['rng']);del checkpoint
    saved_config.write_text(yaml.safe_dump(config,sort_keys=False))
    source_files=sorted([*Path('src/ocoot').glob('*.py'),*Path('scripts').glob('*.py'),*Path('tests').glob('*.py'),
                         Path('environment.lock.txt'),Path('environment.yml'),Path('pyproject.toml'),Path('paper/implementation_notes.md')])
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (run/f'source_manifest_step_{step}.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(run/f'source_snapshot_step_{step}.tar.gz','w:gz') as archive:
        for p in source_files:archive.add(p,arcname=str(p))
    forward=torch.compile(model,fullgraph=True,mode=tr.get('compile_mode','default')) if tr['compile'] else model
    terminate=[False]
    def on_signal(signum, frame):terminate[0]=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    val_indices=np.sort(np.random.default_rng(923).choice(data.val,min(len(data.val),tr['val_cells']),replace=False))
    np.save(run/'validation_probe_indices.npy',val_indices)
    def save():
        state={'config':config,'model':model.state_dict(),'optimizer':opt.state_dict(),
               'ema':ema.state_dict(),'step':step,'best':best,'best_generation':best_generation,
               'best_generation_epoch':best_generation_epoch,'rng':rng_state()}
        state.update(early_reference=early_reference,early_reference_epoch=early_reference_epoch)
        state['pending_generation_epoch']=pending_generation_epoch
        # old last is retained until the new file is completely fsynced.
        atomic_save(state,run/'last.pt')
        (run/'checkpoint_state.json').write_text(json.dumps({'step':step,'epoch':step/steps_per_epoch,'time':timestamp()},indent=2))
    readme_event(f'启动/续训 `{run}`；训练细胞 {len(indices):,}；有效 batch={effective}、microbatch={micro}；计划 {tr["epochs"]} epoch；噪声标准差={tr["noise_std"]}。完整配置 `{saved_config}`。')
    log_event(run/'events.jsonl',{'event':'start','step':step,'training_cells':len(indices),'steps_per_epoch':steps_per_epoch,'parameters':sum(p.numel() for p in model.parameters()),'gpu':torch.cuda.get_device_name()})
    last_log=time.monotonic();losses=[];log_cells=0
    def epoch_generations(epoch_number):
        nonlocal best_generation,best_generation_epoch,early_reference,early_reference_epoch
        nonlocal pending_generation_epoch,last_log,losses,log_cells
        evaluation_rng=rng_state()
        try:
            if not tr.get('train_subset') and epoch_number in tr.get('diagnostic_generation_epochs',[]):
                from .evaluate import generate,official_metrics
                diagnostic_out=run/f'diagnostic_raw_epoch_{epoch_number:03d}'
                diagnostic_idx=data.evaluation_indices(cap=tr.get('diagnostic_generation_cap',8))
                with torch.random.fork_rng(devices=[0]):
                    pred=generate(forward,data,diagnostic_idx,diagnostic_out,noise_std=tr['noise_std'],steps=50,
                                  batch_size=tr['val_generation_batch'],**{k:v for k,v in config['sampling'].items() if k!='steps'})
                    diagnostic=official_metrics(data,diagnostic_idx,pred,diagnostic_out/'metrics')
                    atomic_save({'config':config,'model':model.state_dict(),'step':step,'weights':'raw_diagnostic',
                                 'criterion':'early_condition_learning_diagnostic','metrics':diagnostic},diagnostic_out/'model.pt')
                model.train()
                log_event(run/'events.jsonl',{'event':'diagnostic_generation','epoch':epoch_number,'step':step,'weights':'raw_diagnostic',**diagnostic})
                readme_event(f'`{run}` 第 {epoch_number} 轮原始权重诊断（非 EMA 正式结果、非测试）：PDCorr={diagnostic["PDCorr"]:.4f}，PDS_L2={diagnostic["PDS_L2"]:.4f}，MSE={diagnostic["MSE"]:.5f}。仅用于尽早发现条件学习问题，不参与正式最佳模型选择。')
                last_log=time.monotonic();losses=[];log_cells=0
            if tr.get('val_generation_epochs') and (epoch_number)%tr['val_generation_epochs']==0:
                from .evaluate import generate,official_metrics
                val_out=run/f'validation_epoch_{epoch_number:03d}'
                eval_idx=data.evaluation_indices(cap=tr['val_generation_cap'])
                # Short pilot runs cannot meaningfully assess a 0.9996 EMA's
                # multi-thousand-step horizon; raw weights are diagnostic only.
                use_ema=tr.get('generation_weights','ema')=='ema'
                if not use_ema and not tr.get('train_subset'):
                    raise ValueError('Raw generation is allowed only for diagnostic subset pilots')
                with ema_parameters(model,ema) if use_ema else contextlib.nullcontext():
                    pred=generate(forward,data,eval_idx,val_out,noise_std=tr['noise_std'],steps=tr['val_generation_steps'],batch_size=tr['val_generation_batch'],**{k:v for k,v in config['sampling'].items() if k!='steps'})
                    metrics=official_metrics(data,eval_idx,pred,val_out/'metrics')
                    atomic_save({'config':config,'model':model.state_dict(),'step':step,
                                 'weights':'ema_0.9996' if use_ema else 'raw_pilot_only',
                                 'criterion':'periodic_validation','metrics':metrics},val_out/'model.pt')
                    score=metrics['PDCorr']+sum(metrics[k] for k in ['PDS_L1','PDS_L2','PDS_cos'])/3
                    if score>best_generation:
                        best_generation=score;best_generation_epoch=epoch_number
                        atomic_save({'config':config,'model':model.state_dict(),'step':step,'criterion':'validation_PDCorr_plus_mean_PDS','weights':'ema_0.9996' if use_ema else 'raw_pilot_only','metrics':metrics},run/'best_generation.pt')
                    if score>early_reference+tr.get('early_stop_min_delta',0.002):
                        early_reference=score;early_reference_epoch=epoch_number
                model.train();log_event(run/'events.jsonl',{'event':'validation_generation','epoch':epoch_number,'step':step,**metrics})
                weight_label='EMA=0.9996' if use_ema else '原始权重诊断'
                readme_event(f'`{run}` 第 {epoch_number} 轮验证生成（{weight_label}，每扰动最多 {tr["val_generation_cap"]} 细胞，50 步）：PDCorr={metrics["PDCorr"]:.4f}，MSE={metrics["MSE"]:.5f}，PDS_L2={metrics["PDS_L2"]:.4f}。这是验证结果，非测试结果。')
                last_log=time.monotonic();losses=[];log_cells=0
        finally:
            restore_rng(evaluation_rng)
        pending_generation_epoch=None
        save()

    def converged(epoch_number):
        return (tr.get('early_stop_patience_epochs') and epoch_number>=tr.get('min_epochs',20)
                and early_reference_epoch and epoch_number-early_reference_epoch>=tr['early_stop_patience_epochs'])

    if pending_generation_epoch is not None:
        readme_event(f'`{run}` 恢复第 {pending_generation_epoch} 轮尚未完成的生成评估；已完成的生成批次直接复用。')
        epoch_generations(pending_generation_epoch)
    while step<total_steps:
        if step%steps_per_epoch==0 and converged(step//steps_per_epoch):
            readme_event(f'`{run}` 恢复时已满足验证收敛规则，保存最终 EMA 后结束。')
            break
        epoch=step//steps_per_epoch
        perm=np.random.default_rng(config['seed']+10007*epoch).permutation(indices)
        while step//steps_per_epoch==epoch and step<total_steps:
            within=step%steps_per_epoch
            batch_indices=perm[within*effective:(within+1)*effective]
            lr=tr['lr']*((step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup))))
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True);batch_loss=torch.zeros((),device='cuda')
            for a in range(0,effective,micro):
                x,p,c,b=data.batch(batch_indices[a:a+micro])
                z,t,v=flow_batch(x,tr['noise_std'],tr['p_mean'],tr['p_std'])
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=(forward(z,t,p,c,b).float()-v).square().mean()
                (loss/(effective//micro)).backward();batch_loss+=loss.detach()/(effective//micro)
            if not torch.isfinite(batch_loss):raise FloatingPointError(f'Nonfinite loss at step {step}; previous checkpoint retained')
            grad_norm=torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None])))
            if not torch.isfinite(grad_norm):raise FloatingPointError(f'Nonfinite gradient at step {step}; previous checkpoint retained')
            # No clipping by default: paper does not specify it. Log anomalies explicitly.
            if tr.get('max_grad_norm'):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tr['max_grad_norm'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; previous checkpoint retained')
            opt.step();ema.update(model);step+=1
            losses.append(batch_loss.item());log_cells+=effective
            if step%tr['log_steps']==0 or step==1:
                elapsed=time.monotonic()-last_log
                event={'event':'train','epoch':step/steps_per_epoch,'step':step,'loss':float(np.mean(losses)), 'lr':lr,
                       'cells_per_sec':log_cells/elapsed,'grad_norm':grad_norm.item(),'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9,'time':timestamp()}
                log_event(run/'events.jsonl',event);update_live_status(run,event)
                losses=[];log_cells=0;last_log=time.monotonic()
            epoch_end=step%steps_per_epoch==0
            if epoch_end or step%tr['val_interval_steps']==0:
                raw=validate(model,forward,data,val_indices,tr)
                with ema_parameters(model,ema):ema_val=validate(model,forward,data,val_indices,tr)
                log_event(run/'events.jsonl',{'event':'validation','step':step,'epoch':step/steps_per_epoch,'raw':raw,'ema_0.9996':ema_val})
                if ema_val['endpoint_mse']<best:
                    best=ema_val['endpoint_mse']
                    with ema_parameters(model,ema):atomic_save({'config':config,'model':model.state_dict(),'step':step,'criterion':'ema_endpoint_mse'},run/'best_denoise.pt')
                last_log=time.monotonic();losses=[];log_cells=0
            diagnostic_due=(not tr.get('train_subset') and epoch+1 in tr.get('diagnostic_generation_epochs',[]))
            formal_due=(tr.get('val_generation_epochs') and (epoch+1)%tr['val_generation_epochs']==0)
            if epoch_end and (diagnostic_due or formal_due):
                pending_generation_epoch=epoch+1
                save()  # persist pending work before sampling, including the final epoch
                epoch_generations(pending_generation_epoch)
            if step%tr['checkpoint_steps']==0 or epoch_end or terminate[0]:save()
            if epoch_end:
                readme_event(f'`{run}` 完成第 {epoch+1}/{tr["epochs"]} 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE={ema_val["endpoint_mse"]:.5f}。',
                             important=(epoch+1)%5==0 or epoch+1==tr['epochs'])
            if terminate[0]:
                readme_event(f'`{run}` 收到停止信号，完成当前优化步 {step} 后安全保存。');return
        # Explicit validation-based early stopping; never stops during warmup.
        if converged(epoch+1):
            readme_event(f'`{run}` 验证生成指标连续 {tr["early_stop_patience_epochs"]} 轮无改善，按预设验证收敛规则停止。');break
    save()
    with ema_parameters(model,ema):atomic_save({'config':config,'model':model.state_dict(),'step':step,'criterion':'final_ema_0.9996'},run/'final_ema.pt')
    finished_tmp=run/'finished.json.tmp'
    finished_tmp.write_text(json.dumps({'time':timestamp(),'step':step,'epoch':step/steps_per_epoch,'planned_epochs':tr['epochs'],
                                                'stopped_early':step<total_steps,'early_reference_epoch':early_reference_epoch,
                                                'best_validation_generation':best_generation if np.isfinite(best_generation) else None},indent=2))
    finished_tmp.replace(run/'finished.json')
    readme_event(f'`{run}` 训练结束（{step/steps_per_epoch:.2f} 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。')

