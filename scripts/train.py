import argparse
import yaml
from ocoot.train import train

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--resume-from', help='Start a new run directory from a complete checkpoint with the same training config')
    parser.add_argument('--local-micro-batch', type=int, help='Override the execution microbatch per GPU')
    parser.add_argument('--max-ema-endpoint-relative-gap', type=float,
                        help='Stop and save if EMA denoising MSE exceeds raw denoising MSE by this fraction')
    parser.add_argument('--benchmark-steps', type=int, help='Time optimizer steps after two warmup steps without writing run files')
    parser.add_argument('--preserve-step', type=int, action='append', default=[],
                        help='Pin the first full checkpoint at this step before any evaluation')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    train(config, resume=not args.fresh, local_micro_batch=args.local_micro_batch,
          benchmark_steps=args.benchmark_steps, preserve_steps=args.preserve_step,
          resume_from=args.resume_from, max_ema_endpoint_relative_gap=args.max_ema_endpoint_relative_gap)

