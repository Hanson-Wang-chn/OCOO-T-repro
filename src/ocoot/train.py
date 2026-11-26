"""Single-GPU or DDP trainer with a fixed global batch and restartable state."""
import contextlib
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import time
import tarfile
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml
from .data import BenchmarkData
from .distributed import TrainingRanks, microbatch_shard
from .flow import flow_batch
from .postprocess import primary_score
from .runtime import DualEMA, atomic_save, exclusive_run, log_event, readme_event, timestamp, update_readme_live


@contextlib.contextmanager
# 临时切换到 EMA 权重执行验证或保存，再恢复原始训练参数。
def ema_parameters(model, ema, ranks=None):
    saved={n:p.detach().clone() for n,p in model.named_parameters()}
    try:
        if ranks is None or ranks.primary:
            ema.copy_to(model)
        if ranks is not None and ranks.size > 1:
            for param in model.parameters():
                dist.broadcast(param.detach(),src=0)
        yield
    finally:
        with torch.no_grad():
            for n,p in model.named_parameters():p.copy_(saved[n])


# 将关键训练指标格式化为 README 实时状态。
def update_live_status(run_dir, event):
    update_readme_live(f'实时训练：`{run_dir}`，epoch **{event["epoch"]:.3f}**，优化步 **{event["step"]}**，'
                       f'loss **{event["loss"]:.5f}**，约 **{event["cells_per_sec"]:.1f} cells/s**，更新于 {event["time"]}。')


@torch.no_grad()
# 在固定验证 RNG 和模型 eval 模式下计算速度误差与端点误差。
def validate(model, forward, data, indices, config):
    model.eval();stats=[]
    # A fixed RNG isolates validation from changes in training RNG.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.random.default_generator.manual_seed(9821);torch.cuda.manual_seed(9821)
        for a in range(0,len(indices),config['micro_batch']):
            x,p,c,b=data.batch(indices[a:a+config['micro_batch']])
            z,t,v=flow_batch(x,config['noise_std'],config['p_mean'],config['p_std'])
            with torch.autocast('cuda',dtype=torch.bfloat16):vhat=forward(z,t,p,c,b).float()
            stats.append((len(x),float((vhat-v).square().mean()),float(((z+(1-t[:,None])*vhat)-x).square().mean())))
    model.train()
    return {key:sum(n*s[i] for n,*s in stats)/sum(x[0] for x in stats) for i,key in enumerate(['velocity_mse','endpoint_mse'])}


# public 训练入口：校验参数、初始化进程环境并进入排他训练锁。
def train(config, resume=True, *, local_micro_batch=None, benchmark_steps=None, preserve_steps=(),
          resume_from=None, max_ema_endpoint_relative_gap=None):
    if resume_from is not None and not resume:
        raise ValueError('Resume from a checkpoint and fresh training are mutually exclusive')
    if max_ema_endpoint_relative_gap is not None and not (
            math.isfinite(max_ema_endpoint_relative_gap) and max_ema_endpoint_relative_gap > 0):
        raise ValueError('EMA endpoint error gap must be positive')
    ranks = TrainingRanks()
    with ranks.initialize(), contextlib.ExitStack() as stack:
        if ranks.primary and benchmark_steps is None:
            stack.enter_context(exclusive_run(config['run_dir'], 'training'))
        if ranks.size > 1 and config['training']['compile']:
            # PyTorch 2.8's DDPOptimizer partitions this BF16 model and
            # materially changes its gradients. The inner model can stay
            # compiled while DDP's native reducer performs the all-reduce.
            stack.enter_context(torch._dynamo.config.patch(optimize_ddp=False))
        ranks.barrier()
        return _train_locked(config, resume, ranks, local_micro_batch, benchmark_steps, preserve_steps,
                             resume_from, max_ema_endpoint_relative_gap)


# 内部完整训练流程：恢复、优化、验证、周期生成、收敛和最终保存。
def _train_locked(config, resume, ranks, local_micro_batch, benchmark_steps, preserve_steps,
                  resume_from, max_ema_endpoint_relative_gap):
    tr=config['training'];run=Path(config['run_dir'])
    generation_mse_weight = tr.get('generation_mse_weight', 0.0)
    if not math.isfinite(generation_mse_weight) or generation_mse_weight < 0:
        raise ValueError('generation_mse_weight must be nonnegative and finite')
# benchmark 模式只执行有限优化步并测量吞吐，不产出正式训练结果。
    benchmarking = benchmark_steps is not None
    primary = ranks.primary
    effective = tr['effective_batch']
    micro = tr['micro_batch'] if local_micro_batch is None else local_micro_batch
# 按 rank 分配连续微批次，保证全局 batch 仍完整且每卡累积一致。
    shard = microbatch_shard(effective, micro, ranks.size, ranks.rank)
    accumulation_by_rank = [len(microbatch_shard(effective, micro, ranks.size, i))
                            for i in range(ranks.size)]
# DDP 会对梯度取平均；按全局 batch 比例缩放每个 rank 的微批损失。
    # DDP averages over ranks. Scale each microbatch by its contribution to
    # the global batch, even if some ranks accumulate one more microbatch.
    micro_loss_weight = ranks.size / (effective // micro)
    local_checkpoint = run/'last.pt'
    if resume_from is not None and (local_checkpoint.exists() or (run/'finished.json').exists()):
        raise FileExistsError('An explicit resume checkpoint requires a new run directory')
    checkpoint_path = Path(resume_from) if resume_from is not None else local_checkpoint
    if benchmarking and (not resume or not checkpoint_path.is_file() or benchmark_steps < 1):
        raise ValueError('Benchmarking needs an existing checkpoint and a positive timed step count')
    if not resume and ((run/'last.pt').exists() or (run/'finished.json').exists()):
        raise FileExistsError('Fresh training requires a new run_dir; existing experiment artifacts are retained')
    if resume and not benchmarking and (run/'finished.json').exists():
        if yaml.safe_load((run/'config.yaml').read_text())!=config:
            raise ValueError('Completed run config differs; use a new run directory')
        return
    if primary and not benchmarking:
        (run/'training.pid').write_text(str(os.getpid()))
    torch.set_num_threads(max(1, 8 // ranks.size))
    torch.backends.cuda.matmul.allow_tf32=True
    ranks.seed(config['seed'])
    data=BenchmarkData(config['data_dir'])
    indices=data.train.copy()
    if tr.get('train_subset'):
        indices=np.sort(np.random.default_rng(config['seed']).choice(indices,tr['train_subset'],replace=False))
    if primary and not benchmarking:
        np.save(run/'train_indices.npy',indices)
# 只有显式恢复或本地 checkpoint 存在时才加载训练状态。
    restoring = resume and (resume_from is not None or checkpoint_path.exists())
# 构造模型并按恢复状态决定是否初始化训练均值空条件。
    model = data.build_model(config['model'])
    if not restoring and model.cfg.null_condition_init == 'training_mean':
        model.initialize_null_conditions(*(torch.from_numpy(data.meta[k][indices]) for k in ('pert', 'cell', 'batch')))
    model = model.to(ranks.device).train()
# AdamW 与双 EMA 是该训练实现的固定优化与评估权重管理方式。
    opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],betas=tuple(tr['betas']),weight_decay=tr['weight_decay'],fused=True)
    ema=DualEMA(model,tuple(tr['ema_decays'])) if primary else None
    steps_per_epoch=len(indices)//effective
    assert steps_per_epoch>0
# 根据有效 batch 和总轮数预计算总步数与 warmup 步数。
    total_steps=steps_per_epoch*tr['epochs'];warmup=steps_per_epoch*tr['warmup_epochs']
    step=0;best=float('inf');best_generation=-float('inf');best_generation_epoch=0
    early_reference=-float('inf');early_reference_epoch=0
    pending_generation_epoch=None
    saved_config=run/'config.yaml'
# 恢复前严格校验配置，并加载模型、优化器、EMA、进度与随机状态。
    if restoring:
        checkpoint=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=False)
        expected_config = checkpoint['config']
        if resume_from is not None:
            expected_config = {**expected_config, 'run_dir': config['run_dir']}
        if expected_config!=config:raise ValueError('Resume config differs from the checkpoint')
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        if primary:
            ema.load_state_dict(checkpoint['ema'],ranks.device)
        step=checkpoint['step'];best=checkpoint['best'];best_generation=checkpoint.get('best_generation',-float('inf'))
        best_generation_epoch=checkpoint.get('best_generation_epoch',0)
        early_reference=checkpoint.get('early_reference',best_generation)
        early_reference_epoch=checkpoint.get('early_reference_epoch',best_generation_epoch)
        pending_generation_epoch=checkpoint.get('pending_generation_epoch')
        ranks.restore_rng(checkpoint,config['seed']);del checkpoint
    if primary and not benchmarking:
        saved_config.write_text(yaml.safe_dump(config,sort_keys=False))
        source_files=sorted([*Path('src/ocoot').glob('*.py'),*Path('scripts').glob('*.py'),*Path('tests').glob('*.py'),
                             Path('environment.lock.txt'),Path('environment.yml'),Path('pyproject.toml'),Path('paper/implementation_notes.md')])
        manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        (run/f'source_manifest_step_{step}.json').write_text(json.dumps(manifest,indent=2))
        with tarfile.open(run/f'source_snapshot_step_{step}.tar.gz','w:gz') as archive:
            for p in source_files:archive.add(p,arcname=str(p))
# 可选编译模型；DDP 模式只编译内部模型以避免优化器兼容问题。
    forward=torch.compile(model,fullgraph=True,mode=tr.get('compile_mode','default')) if tr['compile'] else model
    if ranks.size > 1:
        training_forward = DistributedDataParallel(
            forward, device_ids=[ranks.local_rank], broadcast_buffers=False,
            gradient_as_bucket_view=True, find_unused_parameters=model.cfg.context_start >= model.cfg.depth)
        defer_sync = training_forward.no_sync
    else:
        training_forward = forward
        defer_sync = contextlib.nullcontext
# 捕获终止信号；当前优化步结束后保存并退出，而非立即中断。
    terminate=[False]
    def on_signal(signum, frame):terminate[0]=True
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    val_indices=np.sort(np.random.default_rng(923).choice(data.val,min(len(data.val),tr['val_cells']),replace=False))
    if primary and not benchmarking:
        np.save(run/'validation_probe_indices.npy',val_indices)
# 收集所有 rank 的随机状态，由主 rank 原子写入完整 checkpoint。
    def save():
        rank_rng = ranks.gather_rng()
        if primary:
            save_primary(rank_rng)
        ranks.barrier()

# 主 rank 保存模型、优化器、双 EMA、进度和随机状态。
    def save_primary(rank_rng):
        state={'config':config,'model':model.state_dict(),'optimizer':opt.state_dict(),
               'ema':ema.state_dict(),'step':step,'best':best,'best_generation':best_generation,
               'best_generation_epoch':best_generation_epoch,'rng':rank_rng[0]}
        if ranks.size > 1:
            state['rng_by_rank'] = rank_rng
        state.update(early_reference=early_reference,early_reference_epoch=early_reference_epoch)
        state['pending_generation_epoch']=pending_generation_epoch
# 旧 last.pt 保留到新文件完全刷盘并原子替换后。
        # old last is retained until the new file is completely fsynced.
        atomic_save(state,run/'last.pt')
        if step in preserve_steps:
            snapshot = run/'snapshots'/f'step_{step}'/'full.pt'
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not snapshot.exists():
                os.link(run/'last.pt',snapshot)
        summary = run/'checkpoint_state.json'
        temporary = summary.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'step':step,'epoch':step/steps_per_epoch,'time':timestamp()},indent=2))
        temporary.replace(summary)

    start_step=step
    execution={'world_size':ranks.size,'local_micro_batch':micro,'accumulation':len(shard),
               'accumulation_by_rank':accumulation_by_rank,
               'global_batch':effective,'gpu':torch.cuda.get_device_name(ranks.device),
               'dynamo_ddp_optimizer':False if ranks.size>1 and tr['compile'] else None}
    if resume_from is not None:
        execution['resume_checkpoint'] = str(checkpoint_path.resolve())
    if max_ema_endpoint_relative_gap is not None:
        execution['max_ema_endpoint_relative_gap'] = max_ema_endpoint_relative_gap
    if not benchmarking:
        pid = os.getpid()
        fields = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        workers = ranks.gather({'rank':ranks.rank,'pid':pid,'start_tick':int(fields[19])})
        if primary:
            summary = run/'training_execution.json'
            temporary = summary.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'time':timestamp(),'start_step':step,'workers':workers,
                                             'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                                             **execution},indent=2))
            temporary.replace(summary)
    if primary:
        event={'event':'benchmark_start' if benchmarking else 'start','step':step,
               'training_cells':len(indices),'steps_per_epoch':steps_per_epoch,
               'parameters':sum(p.numel() for p in model.parameters()),**execution}
        if benchmarking:
            print(json.dumps(event),flush=True)
        else:
            readme_event(f'启动/续训 `{run}`，从 step {step} 接续，{ranks.size} 卡、每卡 microbatch={micro}、'
                         f'各卡累积 {accumulation_by_rank} 次、全局 batch={effective}，完整配置 `{saved_config}`。')
            log_event(run/'events.jsonl',event)
    last_log=time.monotonic();losses=[];log_cells=0
    generation_ranks = ranks if ranks.size > 1 and tr.get('distributed_generation', False) else None

# 周期采样前先持久化待处理轮次，采样失败也能从同一轮恢复。
    def epoch_generations(epoch_number):
        nonlocal pending_generation_epoch
        if primary or generation_ranks is not None:
            generate_epoch(epoch_number)
        pending_generation_epoch=None
        save()

# 调用评估模块生成该轮 EMA 状态的验证或诊断候选。
    def generate_epoch(epoch_number):
        nonlocal best_generation,best_generation_epoch,early_reference,early_reference_epoch
        nonlocal last_log,losses,log_cells
        evaluation_rng=ranks.local_rng()
        try:
            if not tr.get('train_subset') and epoch_number in tr.get('diagnostic_generation_epochs',[]):
                from .evaluate import generate,official_metrics
                diagnostic_out=run/f'diagnostic_raw_epoch_{epoch_number:03d}'
                diagnostic_idx=data.evaluation_indices(cap=tr.get('diagnostic_generation_cap',8))
                with torch.random.fork_rng(devices=[ranks.local_rank]):
                    pred=generate(forward,data,diagnostic_idx,diagnostic_out,noise_std=tr['noise_std'],steps=50,
                                  batch_size=tr['val_generation_batch'],ranks=generation_ranks,
                                  **{k:v for k,v in config['sampling'].items() if k!='steps'})
                    if primary:
                        diagnostic=official_metrics(data,diagnostic_idx,pred,diagnostic_out/'metrics')
                        atomic_save({'config':config,'model':model.state_dict(),'step':step,'weights':'raw_diagnostic',
                                     'criterion':'early_condition_learning_diagnostic','metrics':diagnostic},diagnostic_out/'model.pt')
                model.train()
                if primary:
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
                with ema_parameters(model,ema,generation_ranks) if use_ema else contextlib.nullcontext():
                    pred=generate(forward,data,eval_idx,val_out,noise_std=tr['noise_std'],steps=tr['val_generation_steps'],
                                  batch_size=tr['val_generation_batch'],ranks=generation_ranks,
                                  **{k:v for k,v in config['sampling'].items() if k!='steps'})
                    if primary:
                        metrics=official_metrics(data,eval_idx,pred,val_out/'metrics')
                        weights='ema_0.9996' if use_ema else 'raw_pilot_only'
                        artifact={'config':config,'model':model.state_dict(),'step':step,'weights':weights,'metrics':metrics}
                        atomic_save({**artifact,'criterion':'periodic_validation'},val_out/'model.pt')
                        score = primary_score(metrics) - generation_mse_weight * metrics['MSE']
                        if generation_mse_weight:
                            artifact.update(generation_mse_weight=generation_mse_weight, selection_score=score)
                        if score>best_generation:
                            best_generation=score;best_generation_epoch=epoch_number
                            criterion = 'validation_PDCorr_plus_mean_PDS'
                            if generation_mse_weight:
                                criterion += '_minus_weighted_MSE'
                            atomic_save({**artifact,'criterion':criterion},run/'best_generation.pt')
                        if score>early_reference+tr.get('early_stop_min_delta',0.002):
                            early_reference=score;early_reference_epoch=epoch_number
                model.train()
                if primary:
                    log_event(run/'events.jsonl',{'event':'validation_generation','epoch':epoch_number,'step':step,
                                                  'selection_score':score,**metrics})
                    weight_label='EMA=0.9996' if use_ema else '原始权重诊断'
                    readme_event(f'`{run}` 第 {epoch_number} 轮验证生成（{weight_label}，每扰动最多 {tr["val_generation_cap"]} 细胞，50 步）：PDCorr={metrics["PDCorr"]:.4f}，MSE={metrics["MSE"]:.5f}，PDS_L2={metrics["PDS_L2"]:.4f}。这是验证结果，非测试结果。')
                last_log=time.monotonic();losses=[];log_cells=0
        finally:
            ranks.set_local_rng(evaluation_rng)

# 基于验证主分数的耐心轮数执行显式早停，不在 warmup 阶段触发。
    def converged(epoch_number):
        value = (tr.get('early_stop_patience_epochs') and epoch_number>=tr.get('min_epochs',20)
                 and early_reference_epoch and epoch_number-early_reference_epoch>=tr['early_stop_patience_epochs'])
        return ranks.from_primary(bool(value))

    if not benchmarking and pending_generation_epoch is not None:
        if primary:
            readme_event(f'`{run}` 恢复第 {pending_generation_epoch} 轮尚未完成的生成评估；已完成的生成批次直接复用。')
        epoch_generations(pending_generation_epoch)
    while step<total_steps:
        if not benchmarking and step%steps_per_epoch==0 and converged(step//steps_per_epoch):
            if primary:
                readme_event(f'`{run}` 恢复时已满足验证收敛规则，保存最终 EMA 后结束。')
            break
        epoch=step//steps_per_epoch
        perm=np.random.default_rng(config['seed']+10007*epoch).permutation(indices)
        while step//steps_per_epoch==epoch and step<total_steps:
            within=step%steps_per_epoch
            lr=tr['lr']*((step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup))))
            for group in opt.param_groups:group['lr']=lr
# 新累积段开始前清空梯度，并按 defer_sync 控制 DDP 同步时机。
            opt.zero_grad(set_to_none=True);batch_loss=torch.zeros((),device=ranks.device)
            for batch in shard:
                first = within*effective + batch*micro
                with defer_sync() if batch != shard.stop-1 else contextlib.nullcontext():
                    x,p,c,b=data.batch(perm[first:first+micro],device=ranks.device)
                    z,t,v=flow_batch(x,tr['noise_std'],tr['p_mean'],tr['p_std']) # 构造一批训练数据
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=(training_forward(z,t,p,c,b).float()-v).square().mean()*micro_loss_weight # 核心前向计算
                    loss.backward() # 自动反向计算
                batch_loss+=loss.detach()
            stopping=terminate[0]
            if ranks.size>1:
                status=torch.stack((batch_loss,batch_loss.new_tensor(stopping)))
                dist.all_reduce(status)
                batch_loss=status[0]/ranks.size
                stopping=bool(status[1].item())
            if not torch.isfinite(batch_loss):raise FloatingPointError(f'Nonfinite loss at step {step}; previous checkpoint retained')
            grad_norm=torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None])))
            if not torch.isfinite(grad_norm):raise FloatingPointError(f'Nonfinite gradient at step {step}; previous checkpoint retained')
            # No clipping by default: paper does not specify it. Log anomalies explicitly.
            if tr.get('max_grad_norm'):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tr['max_grad_norm'])
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient; previous checkpoint retained')
# 累积完成后更新参数，再同步两套 EMA 权重副本。
            opt.step()
            if primary:
                ema.update(model)
            step+=1
            if benchmarking:
                torch.cuda.synchronize(ranks.device)
                progress=step-start_step
                if progress==2:
                    ranks.barrier()
                    benchmark_start=time.monotonic()
                if progress==2+benchmark_steps:
                    ranks.barrier()
                    elapsed=time.monotonic()-benchmark_start
                    if primary:
                        print(json.dumps({'event':'benchmark_result','start_step':start_step,
                                          'first_timed_step':start_step+3,'end_step':step,
                                          'seconds':elapsed,'cells_per_sec':effective*benchmark_steps/elapsed,
                                          'loss':batch_loss.item(),'grad_norm':grad_norm.item(),
                                          'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9,**execution}),flush=True)
                    return
                if stopping:
                    return
                continue
            if primary:
                losses.append(batch_loss.item());log_cells+=effective
            if primary and (step%tr['log_steps']==0 or step==1):
                elapsed=time.monotonic()-last_log
                event={'event':'train','epoch':step/steps_per_epoch,'step':step,'loss':float(np.mean(losses)), 'lr':lr,
                       'cells_per_sec':log_cells/elapsed,'grad_norm':grad_norm.item(),'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9,'time':timestamp(),**execution}
                log_event(run/'events.jsonl',event);update_live_status(run,event)
                losses=[];log_cells=0;last_log=time.monotonic()
# 每个完整 epoch 或固定验证间隔触发验证。
            epoch_end=step%steps_per_epoch==0
            if epoch_end or step%tr['val_interval_steps']==0:
                unhealthy = False
                if primary:
                    raw=validate(model,forward,data,val_indices,tr)
                    with ema_parameters(model,ema):ema_val=validate(model,forward,data,val_indices,tr)
                    log_event(run/'events.jsonl',{'event':'validation','step':step,'epoch':step/steps_per_epoch,'raw':raw,'ema_0.9996':ema_val})
                    if max_ema_endpoint_relative_gap is not None:
                        unhealthy = (ema_val['endpoint_mse'] >
                                     raw['endpoint_mse'] * (1 + max_ema_endpoint_relative_gap))
                    if ema_val['endpoint_mse']<best:
                        best=ema_val['endpoint_mse']
                        with ema_parameters(model,ema):atomic_save({'config':config,'model':model.state_dict(),'step':step,'criterion':'ema_endpoint_mse'},run/'best_denoise.pt')
                    last_log=time.monotonic();losses=[];log_cells=0
                if max_ema_endpoint_relative_gap is not None:
                    unhealthy = ranks.from_primary(unhealthy)
                else:
                    ranks.barrier()
# EMA 指标显著劣于原始模型时保存 checkpoint 并触发健康停止。
                if unhealthy:
                    save()
                    if primary:
                        log_event(run/'events.jsonl', {'event':'ema_health_stop', 'step':step,
                                  'raw_endpoint_mse':raw['endpoint_mse'],
                                  'ema_endpoint_mse':ema_val['endpoint_mse'],
                                  'max_relative_gap':max_ema_endpoint_relative_gap})
                        readme_event(f'`{run}` 的 EMA 去噪误差触发健康阈值，已在 step {step} 保存完整断点并暂停。')
                    return
            diagnostic_due=(not tr.get('train_subset') and epoch+1 in tr.get('diagnostic_generation_epochs',[]))
            formal_due=(tr.get('val_generation_epochs') and (epoch+1)%tr['val_generation_epochs']==0)
# 到达诊断或正式采样轮次时先保存 pending_generation_epoch，再采样。
            if epoch_end and (diagnostic_due or formal_due):
                pending_generation_epoch=epoch+1
                save()  # persist pending work before sampling, including the final epoch
                if not stopping:
                    epoch_generations(pending_generation_epoch)
            elif step%tr['checkpoint_steps']==0 or epoch_end or stopping or step in preserve_steps:
                save()
            if primary and epoch_end:
                readme_event(f'`{run}` 完成第 {epoch+1}/{tr["epochs"]} 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE={ema_val["endpoint_mse"]:.5f}。',
                             important=(epoch+1)%5==0 or epoch+1==tr['epochs'])
            if stopping:
                if primary:
                    readme_event(f'`{run}` 收到停止信号，完成当前优化步 {step} 后安全保存。')
                return
# 按验证主分数的连续无改善轮数判断收敛；warmup 阶段不会触发。
        # Explicit validation-based early stopping; never stops during warmup.
        if not benchmarking and converged(epoch+1):
            if primary:
                readme_event(f'`{run}` 验证生成指标连续 {tr["early_stop_patience_epochs"]} 轮无改善，按预设验证收敛规则停止。')
            break
    if benchmarking:
        raise ValueError('Too few optimizer steps remaining for the requested benchmark')
    save()
    if not primary:
        return
    with ema_parameters(model,ema):atomic_save({'config':config,'model':model.state_dict(),'step':step,'criterion':'final_ema_0.9996'},run/'final_ema.pt')
    finished_tmp=run/'finished.json.tmp'
    finished_tmp.write_text(json.dumps({'time':timestamp(),'step':step,'epoch':step/steps_per_epoch,'planned_epochs':tr['epochs'],
                                                'stopped_early':step<total_steps,'early_reference_epoch':early_reference_epoch,
                                                'best_validation_generation':best_generation if np.isfinite(best_generation) else None},indent=2))
    finished_tmp.replace(run/'finished.json')
    readme_event(f'`{run}` 训练结束（{step/steps_per_epoch:.2f} 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。')

