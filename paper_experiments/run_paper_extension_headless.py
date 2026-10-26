"""Run the frozen experiment under pythonw, with file-backed output streams."""

from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'result/paper_experiments_20261017/supplementary_scenarios_v2_48each'


def main():
    os.chdir(ROOT)
    prior_stdout,prior_stderr=sys.stdout,sys.stderr
    with (STUDY/'run.log').open('a',encoding='utf-8',buffering=1) as output, \
         (STUDY/'run.err').open('a',encoding='utf-8',buffering=1) as errors:
        sys.stdout,sys.stderr = output,errors
        console = None
        if os.name=='nt':
            import ctypes
            console = int(ctypes.windll.kernel32.GetConsoleWindow())
            if console:
                raise RuntimeError('Launch with pythonw.exe; no console window is permitted')
        record=dict(pid=os.getpid(),parent_pid=os.getppid(),console_window=console,
                    started_at=datetime.now().isoformat(timespec='seconds'),state='running')
        path=STUDY/'launcher_status.json'
        path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        original_run=subprocess.run

        def run_child(*args,**kwargs):
            if not kwargs.get('capture_output'):
                kwargs.setdefault('stdout',output)
                kwargs.setdefault('stderr',errors)
            if os.name=='nt':
                kwargs['creationflags']=kwargs.get('creationflags',0)|subprocess.CREATE_NO_WINDOW
            return original_run(*args,**kwargs)

        subprocess.run=run_child
        script=Path(__file__).with_name('extend_paper_manuscript_supplement.py')
        sys.argv=[str(script),*sys.argv[1:]]
        try:
            runpy.run_path(str(script),run_name='__main__')
            record['state']='completed'
        except BaseException:
            record['state']='failed'
            traceback.print_exc(file=errors)
            raise
        finally:
            subprocess.run=original_run
            record['finished_at']=datetime.now().isoformat(timespec='seconds')
            path.write_text(json.dumps(record,indent=2),encoding='utf-8')
            sys.stdout,sys.stderr=prior_stdout,prior_stderr


if __name__=='__main__':
    main()
