"""Export a readable figure from the completed, unchanged paired analysis."""
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
    parser.add_argument('--trial-dir', required=True)
    args = parser.parse_args()
    root = Path(args.trial_dir)
    source = root / 'paired_comparison.json'
    result = json.loads(source.read_text())
    assert result['generated_cells_each'] == 960 and result['perturbations'] == 60
    assert result['continuation_updates'] == 1118 and not result['test_access']
    cases = result['comparisons']
    order = [row['metric'] for row in cases['floor_0']['rows']]
    assert order == [row['metric'] for row in cases['floor_0p05']['rows']]
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 12, 'xtick.labelsize': 8,
                         'ytick.labelsize': 9, 'axes.unicode_minus': True}):
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))
        fig.subplots_adjust(left=.095, right=.98, bottom=.14, top=.88, wspace=.43, hspace=.57)
        for index, (key, ax) in enumerate(zip(order, axes.flat)):
            for y, (name, color) in enumerate((('floor_0', '#28648a'), ('floor_0p05', '#a85d20'))):
                row = cases[name]['rows'][index]
                assert row['metric'] == key
                low, high = row['ci95']
                ax.hlines(y, low, high, color=color, linewidth=2)
                ax.plot(row['difference'], y, 'o', color=color, markersize=5)
            ax.axvline(0, color='.55', linestyle='--', linewidth=1)
            ax.set_yticks([0, 1], ['cutoff 0', 'cutoff 0.05'])
            ax.set_ylim(-.55, 1.55)
            ax.set_title(key + (' (lower is better)' if key in ('MSE', 'MAE') else ''))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            formatter.set_useOffset(False)
            ax.xaxis.set_major_formatter(formatter)
            ax.grid(axis='x', color='.9', linewidth=.5)
            ax.set_axisbelow(True)
        fig.suptitle('Batch retained versus original combined dropout', y=.97, fontsize=15)
        fig.text(.5, .925, 'One mature epoch; 960 validation cells; 60 perturbations; paired 95% bootstrap intervals',
                 ha='center', fontsize=10)
        fig.text(.5, .065, 'Difference = batch retained minus original; higher is better except MSE and MAE.',
                 ha='center', fontsize=10)
        fig.text(.5, .03, 'Same inference indices and noise; intervals do not measure independent training seeds.',
                 ha='center', fontsize=9, color='.3')
        for suffix in ('png', 'pdf'):
            fig.savefig(root / f'paired_comparison_readable.{suffix}', dpi=180)
        plt.close(fig)
    metadata = {'analysis_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_results_changed': False, 'recomputed_metrics': False,
                'reason': 'Display only: shared caption and shorter ticks avoid clipping in the original registered figure.',
                'artifacts': {f'paired_comparison_readable.{s}': hashlib.sha256(
                    (root / f'paired_comparison_readable.{s}').read_bytes()).hexdigest() for s in ('png', 'pdf')}}
    (root / 'readable_figure_provenance.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
