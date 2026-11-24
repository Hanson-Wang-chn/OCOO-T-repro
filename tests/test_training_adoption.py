"""Replacing the workflow parent must not launch a duplicate active trainer."""
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time

import pytest


@pytest.mark.parametrize('finished',[True,False])
def test_existing_trainer_is_waited_for_until_its_lock_releases(tmp_path,finished):
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_experiment.py'))
    code="""import fcntl,pathlib,sys
run=pathlib.Path(sys.argv[1])
with (run/'training.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 print('ready',flush=True)
 sys.stdin.readline()
 if sys.argv[2]=='finish':(run/'finished.json').write_text('{}')
"""
    child=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),'finish' if finished else 'interrupt'],
                           stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    result=[]
    try:
        assert child.stdout.readline().strip()=='ready'
        waiter=threading.Thread(target=lambda:result.append(namespace['wait_for_existing_training'](tmp_path,.01)))
        waiter.start();time.sleep(.04)
        assert waiter.is_alive() and not result
        child.stdin.write('release\n');child.stdin.flush();child.wait(timeout=5)
        waiter.join(timeout=5)
        assert result==[True] and not waiter.is_alive()
        assert (tmp_path/'finished.json').exists()==finished
        assert not namespace['training_lock_held'](tmp_path)
        assert namespace['wait_for_existing_training'](tmp_path,.01) is False
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
        child.stdin.close();child.stdout.close()
