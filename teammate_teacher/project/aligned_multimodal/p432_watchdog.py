"""Parent watchdog for the bounded P432 LaViLa pilot."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run P432 LaViLa pilot under a hard wall-clock watchdog.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir); log = Path(str(out) + ".process.log"); report = Path(str(out) + ".process.json")
    if out.exists() or log.exists() or report.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "aligned_multimodal.p432_rebuild_lavila_bank",
               "--out-dir", str(out)]
    started = time.monotonic()
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,env=env)
        try:
            code = child.wait(timeout=960); status = "complete" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            child.kill(); code = child.wait(timeout=10); status = "watchdog_timeout"
    payload = {"status": status, "exit_code": code, "pid": child.pid,
               "seconds": time.monotonic() - started, "wall_limit_seconds": 960,
               "command": command}
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)
    if code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
