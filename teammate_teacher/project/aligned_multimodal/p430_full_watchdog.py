"""Parent watchdog for the P430 full rebuild; owns only the child process."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run P430 full rebuild under a hard wall-clock watchdog.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); pilot = Path(a.pilot_dir)
    log = Path(str(out) + ".process.log"); report = Path(str(out) + ".process.json")
    if out.exists() or log.exists() or report.exists():
        raise FileExistsError(out)
    if not pilot.exists():
        raise FileNotFoundError(pilot)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "aligned_multimodal.p430_rebuild_full_token_bank",
               "--out-dir", str(out), "--pilot-dir", str(pilot)]
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=1860); status = "complete" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            child.kill(); code = child.wait(timeout=10); status = "watchdog_timeout"
    payload = {"status": status, "exit_code": code, "pid": child.pid,
               "seconds": time.monotonic() - started, "wall_limit_seconds": 1860,
               "command": command}
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)
    if code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
