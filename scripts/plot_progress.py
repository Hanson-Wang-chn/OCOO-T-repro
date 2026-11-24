"""Standalone plots from logs; safe while training is writing new JSONL rows."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('run',nargs='?',default='runs/pilot_sigma_1.0');a=p.parse_args()
    run=Path(a.run);events=[]
    for line in (run/'events.jsonl').read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    train=[e for e in events if e['event']=='train'];val=[e for e in events if e['event']=='validation']
    fig,ax=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    ax[0].plot([e['epoch'] for e in train],[e['loss'] for e in train],lw=1,label='train velocity MSE')
    ax[0].set(xlabel='Epoch',ylabel='Velocity MSE');ax[0].legend()
    for key,label in [('raw','raw'),('ema_0.9996','EMA 0.9996')]:
        ax[1].plot([e['epoch'] for e in val],[e[key]['endpoint_mse'] for e in val],marker='.',label=label)
    ax[1].set(xlabel='Epoch',ylabel='Fixed validation endpoint MSE');ax[1].legend()
    fig.suptitle(run.name);fig.savefig(run/'progress.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()

