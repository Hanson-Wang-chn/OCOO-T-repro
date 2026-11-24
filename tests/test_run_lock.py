import subprocess
import sys
from ocoot.runtime import exclusive_run


def test_directory_lock_prevents_duplicate_writers(tmp_path):
    code="""import sys
from ocoot.runtime import exclusive_run
try:
    with exclusive_run(sys.argv[1],'training'): print('acquired')
except RuntimeError as e:
    print(e)
    raise SystemExit(5)
"""
    with exclusive_run(tmp_path,'training'):
        child=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,timeout=30)
        assert child.returncode==5 and 'already running' in child.stdout
    child=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,timeout=30)
    assert child.returncode==0 and 'acquired' in child.stdout


def test_process_crash_releases_run_lock(tmp_path):
    code="""import sys,signal
from ocoot.runtime import exclusive_run
with exclusive_run(sys.argv[1],'training'):
    print('ready',flush=True)
    signal.pause()
"""
    child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path)],stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        child.kill();child.wait(timeout=10)
        with exclusive_run(tmp_path,'training'):pass
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=10)
