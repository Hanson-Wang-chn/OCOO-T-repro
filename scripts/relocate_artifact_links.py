"""Relocate project artifact symlinks; preserve all recorded scientific JSON."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--artifacts-root', required=True)
    parser.add_argument('--old-artifacts-root', default='/home/dataset-assist-0/whs/ocoo-t-artifacts')
    parser.add_argument('--old-project-root', default='/home/whs/ocoo-t')
    parser.add_argument('--apply', action='store_true', help='Default is a read-only dry run')
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    artifacts = Path(args.artifacts_root).resolve()
    if not artifacts.is_dir():
        raise FileNotFoundError('Copy the complete artifacts directory first')
    mappings = [(args.old_artifacts_root.rstrip('/') + '/', artifacts),
                (args.old_project_root.rstrip('/') + '/', project)]
    changes, missing = [], []
    for base, directories, files in os.walk(project, followlinks=False):
        directories[:] = [d for d in directories if d not in ('.git', '__pycache__', '.pytest_cache')]
        for name in [*directories, *files]:
            path = Path(base) / name
            if not path.is_symlink():
                continue
            original = os.readlink(path)
            for prefix, destination in mappings:
                if original.startswith(prefix):
                    target = destination / original[len(prefix):]
                    if not target.exists():
                        missing.append(str(target))
                    relative = os.path.relpath(target, path.parent)
                    changes.append({'path': str(path.relative_to(project)),
                                    'old': original, 'new': relative})
                    break
    # Preflight every destination before changing even one link.
    if missing:
        raise FileNotFoundError('Nothing changed. Missing archive targets: ' + json.dumps(missing))
    if args.apply:
        for row in changes:
            path = project / row['path']
            temporary = path.with_name(path.name + '.migration-link.tmp')
            if temporary.exists() or temporary.is_symlink():
                raise FileExistsError('Retain existing temporary file: ' + str(temporary))
            temporary.symlink_to(row['new'])
            os.replace(temporary, path)
        report = project / 'logs' / ('migration_links_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
        report.parent.mkdir(exist_ok=True, parents=True)
        report.write_text(json.dumps({'time': dt.datetime.now().astimezone().isoformat(),
                                     'project': str(project), 'artifacts': str(artifacts),
                                     'scientific_JSON_modified': False, 'changes': changes},
                                    ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'dry_run': not args.apply, 'links_to_relocate': len(changes),
                      'missing_targets': missing, 'scientific_JSON_modified': False}))


if __name__ == '__main__':
    main()
