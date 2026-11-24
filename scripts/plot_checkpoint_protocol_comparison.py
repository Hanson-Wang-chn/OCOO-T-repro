"""Plot unchanged paired checkpoint results under two separate protocols."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--periodic', required=True)
    parser.add_argument('--guided', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--reference-epoch', type=int, required=True)
    parser.add_argument('--candidate-epoch', type=int, required=True)
    args = parser.parse_args()
    paths = [Path(args.periodic), Path(args.guided)]
    results = [json.loads(path.read_text()) for path in paths]
    assert all(result['perturbations'] == 60 for result in results)
    assert results[0]['epochs'] == [args.reference_epoch, args.candidate_epoch]
    assert results[1]['factor'] == 'checkpoint'
    assert results[0]['steps'] == [results[1]['reference_options']['step'],
                                   results[1]['candidate_options']['step']]
    order = [row['metric'] for row in results[0]['rows']]
    assert order == [row['metric'] for row in results[1]['rows']]
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12,
                         'xtick.labelsize': 8, 'ytick.labelsize': 9}):
        fig, axes = plt.subplots(3, 3, figsize=(12, 8.8))
        fig.subplots_adjust(left=.09, right=.98, bottom=.14, top=.84, wspace=.44, hspace=.59)
        for i, (metric, ax) in enumerate(zip(order, axes.flat)):
            for y, (result, color) in enumerate(zip(results, ['#28648a', '#a85d20'])):
                row = result['rows'][i]
                ax.hlines(y, *row['ci95'], color=color, linewidth=2)
                ax.plot(row['difference'], y, 'o', color=color, markersize=5)
            ax.axvline(0, color='.55', linestyle='--', linewidth=1)
            ax.set_yticks([0, 1], ['ordinary', 'fixed CFG'])
            ax.set_ylim(-.55, 1.55)
            ax.set_title(metric + (' (lower is better)' if metric in ('MSE', 'MAE') else ''))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_useOffset(False)
            formatter.set_powerlimits((-3, 3))
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(axis='x', color='.9', linewidth=.5)
            ax.set_axisbelow(True)
        fig.suptitle(f'Epoch {args.candidate_epoch} minus epoch {args.reference_epoch}', y=.975, fontsize=16)
        fig.text(.5, .932, 'Paired 95% bootstrap intervals; 60 perturbations; identical noise within each protocol',
                 ha='center', fontsize=10)
        score_labels = []
        for label, result in zip(['ordinary cap32 / B32', 'w2 [0.1,0.9] cap16 / B16'], results):
            change = result['score_difference']
            score_labels.append(f"{label}: primary change {change['value']:+.4f} "
                                f"[{change['ci95'][0]:+.4f}, {change['ci95'][1]:+.4f}]")
        fig.text(.5, .895, '\n'.join(score_labels), ha='center', va='center', fontsize=10)
        fig.text(.5, .07, 'Higher is better except MSE and MAE. Absolute scores across protocols are not compared.',
                 ha='center', fontsize=10)
        fig.text(.5, .035, 'Intervals do not measure independent training-seed or generation-noise uncertainty.',
                 ha='center', fontsize=9, color='.3')
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'paired_protocols.{suffix}', dpi=180)
        plt.close(fig)
    metadata = {'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'recomputed_metrics': False, 'source_results_changed': False,
                'artifacts': {f'paired_protocols.{suffix}': hashlib.sha256(
                    (output / f'paired_protocols.{suffix}').read_bytes()).hexdigest() for suffix in ('png', 'pdf')}}
    (output / 'figure_provenance.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
