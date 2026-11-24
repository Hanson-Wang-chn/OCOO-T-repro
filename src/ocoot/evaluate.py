"""Official Cell Eval wrappers, with explicit target-gene exclusion and DE settings."""
import json
from contextlib import nullcontext
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from .flow import sample
from .runtime import exclusive_run,log_event,update_readme_live,timestamp


METRICS={'pearson_delta':'PDCorr','mse':'MSE','mae':'MAE',
         'discrimination_score_l1':'PDS_L1','discrimination_score_l2':'PDS_L2','discrimination_score_cosine':'PDS_cos',
         'overlap_at_N':'DEOver','precision_at_N':'DEPrec','de_direction_match':'DirAgr'}


def official_metrics(data, indices, pred, outdir, de=False):
    from cell_eval import MetricsEvaluator
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    lines=np.unique(data.meta['cell'][indices]);assert len(lines)==1, 'Evaluate per cell line'
    ctr=data.control[data.meta['cell'][data.control]==lines[0]]
    # A cap limits generated cells only. Retain all held-out real cells for the
    # selected perturbations, so model selection uses stable reference means.
    split_ids=np.unique(data.meta['split'][indices])
    real_indices=indices
    if len(split_ids)==1 and split_ids[0] in [1,2]:
        pool=data.val if split_ids[0]==1 else data.test
        real_indices=pool[(data.meta['cell'][pool]==lines[0]) & np.isin(data.meta['pert'][pool],data.meta['pert'][indices])]
    np.save(outdir/'real_indices.npy',real_indices)
    (outdir/'population_counts.json').write_text(json.dumps({'real_perturbed_cells':len(real_indices),'generated_cells':len(indices),'shared_real_controls':len(ctr)},indent=2))
    # Same true controls in both real and prediction, as required by Appendix B.
    real=np.concatenate([np.asarray(data.x[real_indices]),np.asarray(data.x[ctr])])
    prediction=np.concatenate([pred,np.asarray(data.x[ctr])])
    def observation(idx):
        pids=np.concatenate([data.meta['pert'][idx],data.meta['pert'][ctr]])
        return pd.DataFrame({'gene':[data.vocab['all_genes'][p] for p in pids]},index=[str(i) for i in range(len(pids))])
    var=pd.DataFrame(index=data.vocab['genes'])
    a=ad.AnnData(real,obs=observation(real_indices),var=var);b=ad.AnnData(prediction,obs=observation(indices),var=var.copy())
    # Only paper metrics: avoids quadratic cell-distance metrics irrelevant here.
    from cell_eval.metrics import metrics_registry
    configs={k:{} for k in METRICS if de or k not in ['overlap_at_N','precision_at_N','de_direction_match']}
    if de:
        evaluator=MetricsEvaluator(b,a,control_pert='non-targeting',pert_col='gene',num_threads=8,outdir=str(outdir),skip_de=False)
        per,agg=evaluator.compute(profile='full',metric_configs=configs,
                                 skip_metrics=[k for k in metrics_registry.list_metrics() if k not in configs],break_on_error=True)
    else:
        # Array metrics are mathematically defined for raw real-valued flow
        # outputs too. Call the same official metric functions directly to avoid
        # auto-preprocessing or rejecting negative values during diagnostics.
        from cell_eval import PerturbationAnndataPair,MetricPipeline
        pair=PerturbationAnndataPair(a,b,pert_col='gene',control_pert='non-targeting')
        pipe=MetricPipeline(profile=None,break_on_error=True);pipe.add_metrics(list(configs))
        pipe.compute_anndata_metrics(pair);per=pipe.get_results();agg=pipe.get_agg_results()
        per.write_csv(outdir/'results.csv');agg.write_csv(outdir/'agg_results.csv')
    # Aggregate columns are toolkit-version-specific; compute from per-pert columns
    # with toolkit NaN semantics and preserve original CSVs for audit.
    frame=per.to_pandas()
    summary={display:float(frame[k].mean()) for k,display in METRICS.items() if k in frame}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


@torch.no_grad()
def generate(model, data, indices, outdir, noise_std=1., batch_size=32, steps=50,
             guidance=None,cfg_min=.1,cfg_max=.9,clamp=False,seed=4242,resume=True,*,ranks=None):
    outdir=Path(outdir);outdir.mkdir(exist_ok=True,parents=True)
    model.eval()
    options={'noise_std':noise_std,'batch_size':batch_size,'steps':steps,'guidance':guidance,'cfg_min':cfg_min,'cfg_max':cfg_max,'clamp':clamp,'seed':seed}
    if len(indices) == 0 or batch_size < 1:
        raise ValueError('Generation requires cells and a positive batch size')
    if ranks is not None and ranks.size > 1:
        with exclusive_run(outdir, 'generation') if ranks.primary else nullcontext():
            return _generate_distributed(model, data, indices, outdir, options, resume, ranks)
    start = _generation_start(outdir, indices, options, resume)
    progress = lambda completed: _report_generation(outdir, completed, len(indices), steps)
    result, raw_result = _generate_local(model, data, indices, outdir, options, start, progress)
    _save_sample_statistics(outdir, result, raw_result)
    return result


def _generation_start(outdir, indices, options, resume):
    marker=outdir/'generation_state.json'
    if marker.exists() and resume:
        state=json.loads(marker.read_text())
        if state['options']!=options: raise ValueError('Cannot resume sampling with different options')
        np.testing.assert_array_equal(np.load(outdir/'indices.npy'),indices)
        start=state['completed']
        if state['total'] != len(indices) or not 0 <= start <= len(indices):
            raise ValueError('Inconsistent generation cache size')
        if start != len(indices) and start % options['batch_size']:
            raise ValueError('Incomplete generation cache ends inside a batch')
        return start
    return 0


def _prediction_files(outdir, indices, n_genes, clamp, start):
    shape = (len(indices), n_genes)
    if not start:
        np.save(outdir/'indices.npy',indices)
    mode = 'r+' if start else 'w+'
    result = np.lib.format.open_memmap(outdir/'prediction.npy',mode=mode,dtype=np.float32,shape=shape)
    raw_result = None
    if clamp:
        raw_file=outdir/'prediction_raw.npy'
        if start and not raw_file.exists():raise ValueError('Raw sampling cache is missing; use a new output directory')
        raw_result=np.lib.format.open_memmap(raw_file,mode=mode,dtype=np.float32,shape=shape)
    for array in (result, raw_result):
        if array is not None and (array.dtype != np.float32 or array.shape != shape):
            raise ValueError('Inconsistent prediction array schema')
    return result, raw_result


def _mark_generated(outdir, options, completed, total):
    tmp=outdir/'generation_state.tmp'
    tmp.write_text(json.dumps({'completed':completed,'total':total,'options':options}))
    tmp.replace(outdir/'generation_state.json')


def _report_generation(outdir, completed, total, steps):
    update_readme_live(f'实时生成评估：`{outdir}`，已生成 **{completed:,}/{total:,}** 个细胞（Heun {steps} 步）；更新于 {timestamp()}。')


def _generate_local(model, data, indices, outdir, options, start, progress=None):
    batch_size = options['batch_size']
    clamp = options['clamp']
    seed = options['seed']
    result, raw_result = _prediction_files(outdir, indices, data.x.shape[1], clamp, start)
    flow_options = {k: options[k] for k in ('noise_std', 'steps', 'guidance', 'cfg_min', 'cfg_max')}
    for a in range(start,len(indices),batch_size):
        idx=indices[a:a+batch_size]
        p,c,b=[torch.from_numpy(data.meta[k][idx]).cuda() for k in ['pert','cell','batch']]
        x=sample(model,p,c,b,data.x.shape[1],seed=seed+a,clamp=False,**flow_options).cpu().numpy()
        if not np.isfinite(x).all():raise FloatingPointError('Nonfinite generated samples')
        completed = a + len(idx)
        if raw_result is not None:
            raw_result[a:completed]=x
            raw_result.flush()
        result[a:completed]=np.maximum(x,0) if clamp else x
        result.flush()
        _mark_generated(outdir, options, completed, len(indices))
        if a//batch_size%10==0:log_event(outdir/'generation.jsonl',{'completed':completed,'total':len(indices)})
        if progress is not None and (a//batch_size%5==0 or completed==len(indices)):
            progress(completed)
    return result, raw_result


def _generate_distributed(model, data, indices, outdir, options, resume, ranks):
    start = ranks.from_primary(_generation_start(outdir, indices, options, resume) if ranks.primary else None)
    if start < len(indices):
        batch_size = options['batch_size']
        batches = (len(indices) - start + batch_size - 1) // batch_size
        base, extra = divmod(batches, ranks.size)
        bounds = [min(len(indices), start + (rank * base + min(rank, extra)) * batch_size)
                  for rank in range(ranks.size + 1)]
        directory = outdir / f'parts_w{ranks.size}_from{start}'

        def report_parts(_):
            completed = start + sum(json.loads(path.read_text())['completed']
                                    for path in directory.glob('rank_*/generation_state.json'))
            _report_generation(outdir, completed, len(indices), options['steps'])

        first, last = bounds[ranks.rank:ranks.rank+2]
        if first < last:
            part = directory / f'rank_{ranks.rank}'
            part.mkdir(parents=True, exist_ok=True)
            part_options = dict(options, seed=options['seed']+first)
            offset = _generation_start(part, indices[first:last], part_options, resume)
            _generate_local(model, data, indices[first:last], part, part_options, offset,
                            report_parts if ranks.primary else None)
        ranks.barrier()
        if ranks.primary:
            result, raw_result = _prediction_files(outdir, indices, data.x.shape[1], options['clamp'], start)
            for rank, (first, last) in enumerate(zip(bounds, bounds[1:])):
                if first == last:
                    continue
                part = directory / f'rank_{rank}'
                part_options = dict(options, seed=options['seed']+first)
                if _generation_start(part, indices[first:last], part_options, True) != last-first:
                    raise RuntimeError(f'Incomplete generation part: {part}')
                part_result, part_raw = _prediction_files(
                    part, indices[first:last], data.x.shape[1], options['clamp'], last-first)
                result[first:last] = part_result
                if raw_result is not None:
                    raw_result[first:last] = part_raw
            result.flush()
            if raw_result is not None:
                raw_result.flush()
            _mark_generated(outdir, options, len(indices), len(indices))
            update_readme_live(f'{ranks.size} 卡生成评估：`{outdir}`，已合并 **{len(indices):,}** 个细胞；更新于 {timestamp()}。')
    elif ranks.primary:
        result, raw_result = _prediction_files(outdir, indices, data.x.shape[1], options['clamp'], start)
    if ranks.primary:
        _save_sample_statistics(outdir, result, raw_result)
        return result
    return None


def _save_sample_statistics(outdir, result, raw_result):
    def moments(array):
        total=0.;squares=0.;negative=0;zeros=0;n=0;lo=float('inf');hi=-float('inf')
        for a in range(0,len(array),4096):
            x=np.asarray(array[a:a+4096]);total+=x.sum(dtype=np.float64);squares+=np.square(x).sum(dtype=np.float64)
            negative+=int((x<0).sum());zeros+=int((x==0).sum());n+=x.size;lo=min(lo,float(x.min()));hi=max(hi,float(x.max()))
        return {'mean':float(total/n),'std':float(np.sqrt(max(0.,squares/n-(total/n)**2))),
                'min':lo,'max':hi,'negative_fraction':negative/n,'zero_fraction':zeros/n}
    (outdir/'sample_statistics.json').write_text(json.dumps({'raw':moments(raw_result if raw_result is not None else result),
                                                           'evaluated':moments(result),'postprocess':'clamp_min_0' if raw_result is not None else 'none'},indent=2))

