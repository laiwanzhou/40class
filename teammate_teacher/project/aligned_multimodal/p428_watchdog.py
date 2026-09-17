"""Own-child process wall-time guard for P428 runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out-dir",required=True)
    p.add_argument("--pilot-dir"); p.add_argument("--run",action="store_true")
    a=p.parse_args(); out=Path(a.out_dir)
    if out.exists(): raise FileExistsError(out)
    # Sibling log/report allows the child to retain its exclusive mkdir contract.
    log_path=Path(str(out)+".process.log"); report_path=Path(str(out)+".process.json")
    if log_path.exists() or report_path.exists(): raise FileExistsError(log_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-u","-m","aligned_multimodal.p428_rebuild_skeleton_bank",
             "--run" if a.run else "--pilot","--out-dir",str(out)]
    if a.pilot_dir: command.extend(["--pilot-dir",a.pilot_dir])
    started=time.monotonic(); limit=10860 if a.run else 960
    with log_path.open("x",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            code=process.wait(timeout=limit); status="complete" if code==0 else "failed"
        except subprocess.TimeoutExpired:
            process.kill(); code=process.wait(timeout=10); status="watchdog_timeout"
    report={"status":status,"exit_code":code,"pid":process.pid,"seconds":time.monotonic()-started,
            "wall_limit_seconds":limit,"command":command}
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
    if code != 0: raise SystemExit(1)


if __name__=="__main__": main()
