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
    parser.add_argument('--ema-health-min-step', type=int, default=None,
                        help='First step eligible for the EMA denoising health check (default: config or 12000)')
    parser.add_argument('--finish-schedule', action='store_true',
                        help='Resume an early-stopped run through its original cosine budget')
    parser.add_argument('--benchmark-steps', type=int, help='Time optimizer steps after two warmup steps without writing run files')
    parser.add_argument('--preserve-step', type=int, action='append', default=[],
                        help='Pin the first full checkpoint at this step before any evaluation')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    max_gap = args.max_ema_endpoint_relative_gap
    if max_gap is None:
        max_gap = config['training'].get('max_ema_endpoint_relative_gap')
    min_step = args.ema_health_min_step
    if min_step is None:
        min_step = config['training'].get('ema_health_min_step', 12000)
    train(config, resume=not args.fresh, local_micro_batch=args.local_micro_batch,
          benchmark_steps=args.benchmark_steps, preserve_steps=args.preserve_step,
          resume_from=args.resume_from, max_ema_endpoint_relative_gap=max_gap,
          ema_health_min_step=min_step, finish_schedule=args.finish_schedule)

