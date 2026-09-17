"""Owned-child CPU P431 watchdog."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--out-dir",required=True);p.add_argument("--pilot-dir");p.add_argument("--run",action="store_true");a=p.parse_args()
    out=Path(a.out_dir);log=Path(str(out)+".process.log");report=Path(str(out)+".process.json")
    if out.exists() or log.exists() or report.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,"-u","-m","aligned_multimodal.p431_irthermal_run","--run" if a.run else "--pilot","--out-dir",str(out)]
    env=os.environ.copy()
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[key]="1"
    if a.pilot_dir:cmd.extend(["--pilot-dir",a.pilot_dir])
    start=time.monotonic()
    with log.open("x",encoding="utf-8") as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        try:code=child.wait(timeout=1230 if a.run else 330);status="complete" if code==0 else "failed"
        except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=10);status="watchdog_timeout"
    r={"status":status,"exit_code":code,"pid":child.pid,"seconds":time.monotonic()-start};report.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    if code:raise SystemExit(1)


if __name__=="__main__":main()
