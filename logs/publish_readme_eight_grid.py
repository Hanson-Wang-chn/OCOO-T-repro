"""One-time overview refresh, using the same 15-minute publication lock."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import time


readme = Path('README.md')
draft = Path('logs/README_eight_grid_pending.md')
while True:
    with Path('logs/readme.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        remaining = 900 - (time.time() - readme.stat().st_mtime)
        if remaining <= 0:
            before = readme.read_text()
            pending_path = Path('logs/readme_pending.json')
            pending = json.loads(pending_path.read_text()) if pending_path.exists() else {}
            start = '<!-- LIVE_TRAINING_START -->'
            end = '<!-- LIVE_TRAINING_END -->'
            previous_live = before.split(start, 1)[1].split(end, 1)[0].strip() if start in before and end in before else ''
            live = pending.get('live') or previous_live or '主训练持续运行，使用只读状态脚本查看最新断点。'
            text = draft.read_text().replace('{{LIVE_STATUS}}', live)
            for event in pending.get('events', []):
                if event.strip() not in text:
                    text += event
            backup = Path('logs/README_before_eight_grid_20261118.md')
            if not backup.exists():
                backup.write_text(before)
            temporary = readme.with_suffix('.md.tmp')
            temporary.write_text(text)
            temporary.replace(readme)
            if pending_path.exists():
                pending_path.unlink()
            record = {'time': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                      'publication_interval_seconds': 900,
                      'previous_readme_backup': str(backup),
                      'draft_sha256': hashlib.sha256(draft.read_bytes()).hexdigest(),
                      'readme_sha256': hashlib.sha256(readme.read_bytes()).hexdigest()}
            Path('logs/readme_eight_grid_publish.json').write_text(json.dumps(record, indent=2))
            print(json.dumps(record, indent=2), flush=True)
            break
    time.sleep(min(30, max(.1, remaining)))
