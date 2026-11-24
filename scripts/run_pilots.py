"""Sequential pilots, so a long first run cannot leave the GPU idle afterward."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int)
    p.add_argument('--trials',default='sigma_1.0,sigma_0.5,sigma_1.0_t1000,sigma_0.5_t1000,sigma_1.0_cellinit_1.0,sigma_0.5_cellinit_1.0',
                   help='Comma-separated trial names, without the pilot_ prefix')
    p.add_argument('--comparison-output',default='runs/pilot_comparison.json')
    p.add_argument('--de',action='store_true',help='Add the three DE metrics on nonnegative cached predictions')
    p.add_argument('--evaluate-only',action='store_true',help='Refuse unfinished trials instead of starting training')
    p.add_argument('--pid-file',default='logs/pilots.pid')
    a=p.parse_args()
    trials=[trial.strip() for trial in a.trials.split(',')]
    if not trials or any(not trial or '/' in trial or '\\' in trial or trial in ['.','..'] for trial in trials):
        p.error('trials must be nonempty local trial names')
    for trial in trials:
        if not Path(f'configs/pilot_{trial}.yaml').is_file():p.error(f'Missing config for {trial}')
    Path('logs').mkdir(exist_ok=True)
    pid_file=Path(a.pid_file);pid_file.parent.mkdir(parents=True,exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    if a.wait_pid:
        while True:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:break
            time.sleep(15)
    for trial in trials:
        run=Path(f'runs/pilot_{trial}')
        if (run/'finished.json').exists():continue
        if a.evaluate_only:raise RuntimeError(f'Evaluation-only mode requires a completed trial: {trial}')
        with open(f'logs/pilot_{trial}.log','a') as log:
            subprocess.run([sys.executable,'scripts/train.py','--config',f'configs/pilot_{trial}.yaml'],stdout=log,stderr=log,check=True)
    results={}
    for trial in trials:
        path=Path(f'runs/pilot_{trial}/validation_epoch_012')
        results[trial]=json.loads((path/'metrics/summary.json').read_text())
        # Report nonnegative postprocessing from the same generated cells.
        import numpy as np
        from ocoot.data import BenchmarkData
        from ocoot.evaluate import official_metrics
        data=BenchmarkData();idx=np.load(path/'indices.npy');pred=np.load(path/'prediction.npy',mmap_mode='r')
        metric_directory=path/('metrics_clamped_de' if a.de else 'metrics_clamped')
        results[trial]['clamped']=official_metrics(data,idx,np.maximum(pred,0),metric_directory,de=a.de)
        results[trial]['negative_fraction']=float((pred<0).mean())
    output=Path(a.comparison_output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(results,indent=2));temporary.replace(output)
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()

