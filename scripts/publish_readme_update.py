"""Publish a registered README edit at the next 15-minute update window."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan_text = plan_path.read_text()
    plan = json.loads(plan_text)
    status = plan_path.with_suffix('.published.json')
    while not status.exists():
        with Path('logs/readme.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            readme = Path('README.md')
            previous_mtime = readme.stat().st_mtime
            remaining = previous_mtime + 900 - time.time()
            if remaining <= 0:
                # Apply to the current document, retaining changes/events that
                # the trainer has already published while this job waited.
                original = readme.read_text()
                text = original
                for old, new in plan['replacements']:
                    if text.count(old) != 1:
                        raise ValueError('README changed since the registered edit; review before applying')
                    text = text.replace(old, new, 1)
                pending_path = Path('logs/readme_pending.json')
                pending = json.loads(pending_path.read_text()) if pending_path.exists() else {'events': []}
                if pending.get('live') is not None:
                    start, end = '<!-- LIVE_TRAINING_START -->', '<!-- LIVE_TRAINING_END -->'
                    block = f'{start}\n\n{pending["live"]}\n\n{end}'
                    text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
                for event in pending['events']:
                    if event.strip() not in text:
                        text += event
                backup = Path(plan['backup'])
                if backup.exists():
                    raise FileExistsError('Backup already exists; inspect previous publication')
                backup.write_text(original)
                temporary = readme.with_suffix('.md.tmp')
                temporary.write_text(text)
                temporary.replace(readme)
                if pending_path.exists():
                    pending_path.unlink()
                result = {'time': dt.datetime.now().astimezone().isoformat(timespec='seconds'),
                          'previous_publication_unix': previous_mtime,
                          'publication_interval_seconds': readme.stat().st_mtime - previous_mtime,
                          'plan_sha256': hashlib.sha256(plan_text.encode()).hexdigest(),
                          'readme_sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'backup': str(backup)}
                status.write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
                return
        time.sleep(min(max(remaining, .05), 30))


if __name__ == '__main__':
    main()
