"""Parent watchdog for the bounded P430 pilot; owns only the child process."""
import argparse, json, subprocess, sys, time
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--out-dir", required=True); a=p.parse_args(argv)
    out=Path(a.out_dir); log=Path(str(out)+".process.log"); report=Path(str(out)+".process.json")
    if out.exists() or log.exists() or report.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    command=[sys.executable,"-u","-m","aligned_multimodal.p430_rebuild_token_bank","--out-dir",str(out)]
    started=time.monotonic()
    with log.open("x",encoding="utf-8") as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        try: code=child.wait(timeout=960); status="complete" if code==0 else "failed"
        except subprocess.TimeoutExpired:
            child.kill(); code=child.wait(timeout=10); status="watchdog_timeout"
    payload={"status":status,"exit_code":code,"pid":child.pid,"seconds":time.monotonic()-started,
             "wall_limit_seconds":960,"command":command}
    report.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload),flush=True)
    if code != 0: raise SystemExit(1)
if __name__ == "__main__": main()
