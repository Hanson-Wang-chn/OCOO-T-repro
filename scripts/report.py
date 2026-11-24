"""Compare full-test results to Table 3; bootstrap over perturbations only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ocoot.evaluate import METRICS
from ocoot.postprocess import zero_floor
from ocoot.runtime import file_sha256

PAPER={'PDCorr':.437,'DEOver':.232,'DEPrec':.240,'DirAgr':.766,
       'PDS_L1':.820,'PDS_L2':.853,'PDS_cos':.833,'MSE':.00790,'MAE':.061}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation-dir',required=True);p.add_argument('--outdir',default='results');a=p.parse_args()
    ev=Path(a.evaluation_dir);out=Path(a.outdir);out.mkdir(exist_ok=True,parents=True)
    config=json.loads((ev/'evaluation_config.json').read_text())
    floor=zero_floor(config.get('zero_floor',0.))
    if config['split']!='test' or config['cap'] is not None or not config['de']:
        raise ValueError('Final report requires all test cells and all nine paper metrics')
    per=pd.read_csv(ev/'metrics/results.csv');population=json.loads((ev/'metrics/population_counts.json').read_text())
    assert population['real_perturbed_cells']==26878 and population['generated_cells']==26878
    assert len(per)==380
    baseline_metrics=None
    failed_validation_checks=[]
    protocol_path=ev/'test_protocol.json'
    if protocol_path.exists():
        checks=json.loads(protocol_path.read_text()).get('validation_checks',{})
        failed_validation_checks=[key for key,passed in checks.items() if not passed]
    if floor>0:
        completed=json.loads((ev/'postprocess_test_completed.json').read_text())
        if (completed['protocol']['zero_floor']!=floor or
                completed['protocol']['selection_sha256']!=config['postprocess_selection_sha256'] or
                completed['selection_changed_after_test']):
            raise ValueError('Report requires the postprocessing setting frozen before test generation')
        for path,digest in completed['output_sha256'].items():
            if file_sha256(path)!=digest:raise ValueError('Completed test artifact changed: '+path)
        baseline=Path(config['baseline_evaluation_dir'])
        baseline_config=json.loads((baseline/'evaluation_config.json').read_text())
        if any(config.get(k)!=v for k,v in baseline_config.items()):
            raise ValueError('Baseline and selected test use different generation settings')
        baseline_metrics=json.loads((baseline/'metrics/summary.json').read_text())
        if set(baseline_metrics)!=set(METRICS.values()):raise ValueError('Baseline needs all nine metrics')
    rng=np.random.default_rng(42);rows=[]
    for key,display in METRICS.items():
        values=per[key].dropna().to_numpy()
        means=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
        lo,hi=np.quantile(means,[.025,.975])
        rows.append({'metric':display,'paper':PAPER[display],'reproduction':float(values.mean()),
                     'CI_low':float(lo),'CI_high':float(hi),'valid_perturbations':len(values)})
        if baseline_metrics is not None:rows[-1]['baseline_clamp0']=baseline_metrics[display]
    frame=pd.DataFrame(rows);frame.to_csv(out/'benchmark_comparison.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(11,8),layout='constrained')
    for ax,row in zip(axes.flat,rows):
        v=row['reproduction'];lo=row['CI_low'];hi=row['CI_high']
        ax.bar([0,1],[row['paper'],v],color=['#858d99','#3478a6'],width=.55)
        ax.errorbar([1],[v],yerr=[[v-lo],[hi-v]],fmt='none',color='black',capsize=3)
        ax.set_xticks([0,1],['Paper',f'Floor {floor:g}' if floor>0 else 'Reproduction']);ax.set_title(row['metric']);ax.set_ylim(bottom=min(0.,lo))
    title='Replogle–Nadig: 2,000 HVGs; 380 test perturbations'
    if floor>0:title+=f'; validation-selected zero floor {floor:g}'
    fig.suptitle(title)
    fig.savefig(out/'benchmark_comparison.png',dpi=180);fig.savefig(out/'benchmark_comparison.pdf');plt.close(fig)
    text='# Replogle–Nadig 复现结果\n\n'
    text+='正式测试：26,878 个扰动细胞、380 个扰动、2,000 HVG；真实对照与预测端对照一致。\n\n'
    if failed_validation_checks:
        text+='完整验证低于项目检查线的指标为 '+ '、'.join(failed_validation_checks) +'。测试配置按登记计划在完整验证后冻结。\n\n'
    if floor>0:
        text+=f'本次采用完整验证锁定的近零阈值 **{floor:g}**：将低于该值的生成表达置0。原文未公开这一处理细节，属于明确记录的复现选择。下面同时保留相同raw样本的原始clamp0对照；未根据测试结果选择阈值。\n\n'
    text+='| 指标 | 原文 Table 3 | 本次 | 95% 区间 |\n| --- | ---: | ---: | ---: |\n'
    for r in rows:text+=f'| {r["metric"]} | {r["paper"]:.5g} | {r["reproduction"]:.5g} | [{r["CI_low"]:.5g}, {r["CI_high"]:.5g}] |\n'
    if baseline_metrics is not None:
        text+='\n同一生成样本的后处理对照：\n\n| 指标 | 原始clamp0 | 验证选定处理 |\n| --- | ---: | ---: |\n'
        for r in rows:text+=f'| {r["metric"]} | {r["baseline_clamp0"]:.5g} | {r["reproduction"]:.5g} |\n'
    text+='\n区间来自测试扰动的 2,000 次 bootstrap，反映条件间差异，不代表多次独立训练的误差。\n\n'
    text+='检查点与采样设置：\n\n```json\n'+json.dumps(config,indent=2,ensure_ascii=False)+'\n```\n'
    text+='\n原文未公开的实现选择与所有尝试见 `paper/implementation_notes.md`；本表由实际 Cell Eval 结果生成，不自动判定复现成功。\n'
    (out/'REPORT.md').write_text(text)


if __name__=='__main__':main()

