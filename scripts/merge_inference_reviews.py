"""Build a read-only analysis view of identical-protocol validation reviews."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from ocoot.runtime import exclusive_run, file_sha256, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dirs', nargs='+', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    inputs = [Path(name).resolve() for name in args.review_dirs]
    out = Path(args.outdir).resolve()
    if out in inputs:
        raise ValueError('Analysis view must be separate from its source reviews')
    protocols = []
    cases = {}
    locations = {}
    source_hashes = {}
    for folder in inputs:
        state = json.loads((folder / 'review_state.json').read_text())
        if state['stage'] != 'complete':
            raise ValueError(f'Incomplete source review: {folder}')
        config = json.loads((folder / 'review_config.json').read_text())
        if config['split'] != 'validation':
            raise ValueError('Only validation reviews can be combined')
        protocols.append(config)
        source_hashes[str(folder)] = {name: file_sha256(folder / name)
                                     for name in ('review_config.json', 'comparison.json')}
        current = json.loads((folder / 'comparison.json').read_text())
        if len(current) != len(config['guidances']):
            raise ValueError(f'Missing completed cases: {folder}')
        for tag, record in current.items():
            if '/' in tag or '\\' in tag or tag in ('.', '..'):
                raise ValueError('Case tag is not local')
            path = folder / tag
            if not (path / 'review_completed.json').exists():
                raise ValueError(f'Missing per-case completion marker: {path}')
            if tag in cases:
                if cases[tag] != record:
                    raise ValueError(f'Conflicting repeated case: {tag}')
                for name in ('indices.npy', 'prediction.npy', 'prediction_raw.npy',
                             'evaluation_config.json', 'metrics/results.csv'):
                    if file_sha256(path / name) != file_sha256(locations[tag] / name):
                        raise ValueError(f'Repeated case has different source bytes: {tag}/{name}')
            else:
                cases[tag], locations[tag] = record, path.resolve()
    normalized = [copy.deepcopy(config) for config in protocols]
    for config in normalized:
        config.pop('guidances')
    if any(config != normalized[0] for config in normalized[1:]):
        raise ValueError('Reviews differ beyond the list of guidance strengths')
    order = sorted(cases, key=lambda tag: (-1 if cases[tag]['guidance'] is None else cases[tag]['guidance']))
    combined = {tag: cases[tag] for tag in order}
    config = copy.deepcopy(protocols[0])
    config.update(guidances=[cases[tag]['guidance'] for tag in order],
                  creation_method='read-only cache union; no generated values changed',
                  source_reviews=source_hashes)
    fingerprint = {'source_reviews': source_hashes, 'script_sha256': file_sha256(__file__)}
    manifest = out / 'merge_manifest.json'
    if out.exists() and not manifest.exists() and any(out.iterdir()):
        raise FileExistsError('Output contains artifacts from a different operation')
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out, 'merge'):
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Analysis view is pinned to different source files')
        atomic_json(manifest, {'time': timestamp(), 'stage': 'linking', 'fingerprint': fingerprint})
        for tag in order:
            link = out / tag
            if link.is_symlink():
                if link.resolve() != locations[tag]:
                    raise ValueError('Existing case link points to different data')
            elif link.exists():
                raise FileExistsError('Do not replace original case directories')
            else:
                link.symlink_to(locations[tag], target_is_directory=True)
        atomic_json(out / 'review_config.json', config)
        atomic_json(out / 'comparison.json', combined)
        subprocess.run([sys.executable, 'scripts/analyze_inference_review.py', '--review-dir', str(out)], check=True)
        atomic_json(manifest, {'time': timestamp(), 'stage': 'completed', 'fingerprint': fingerprint,
                              'cases': order, 'new_generated_cells': 0,
                              'source_mutations': False, 'test_access': False})
        print(json.dumps({'outdir': str(out), 'cases': order, 'new_generated_cells': 0}))


if __name__ == '__main__':
    main()
