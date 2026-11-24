import argparse
import yaml
from ocoot.train import train

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--fresh',action='store_true');a=p.parse_args()
    train(yaml.safe_load(open(a.config)),resume=not a.fresh)

