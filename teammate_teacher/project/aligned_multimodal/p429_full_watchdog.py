"""Parent-only watchdog for the P429 full thermal rebuild."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def wait_owned(child,active_path,*,clock=time.monotonic,sleep=time.sleep,total_limit=43320,context_limit=7200):
    started=clock();active=None;active_since=None
    while child.poll() is None:
        now=clock();observed=active
        try:observed=json.loads(Path(active_path).read_text(encoding="utf-8"))["context"]
        except (OSError,ValueError,KeyError,TypeError):pass
        if observed!=active:
            active=observed;active_since=now if active is not None else None
        reason=None
        if now-started>=total_limit:reason="watchdog_timeout"
        elif active_since is not None and now-active_since>=context_limit:reason="context_watchdog_timeout"
        if reason is not None:
            child.kill();return child.wait(timeout=10),reason
        sleep(1)
    code=child.wait(timeout=10)
    return code,"complete" if code==0 else "failed"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run P429 full thermal rebuild with a hard wall limit")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-dir", required=True)
    a = ap.parse_args(argv)
    out, pilot = Path(a.out_dir), Path(a.pilot_dir)
    log, report = Path(str(out) + ".process.log"), Path(str(out) + ".process.json")
    if out.exists() or log.exists() or report.exists():
        raise FileExistsError(out)
    if not pilot.exists():
        raise FileNotFoundError(pilot)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "aligned_multimodal.p429_rebuild_full_thermal_bank",
               "--out-dir", str(out), "--pilot-dir", str(pilot)]
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code,status=wait_owned(child,out/"active_context.json")
        except BaseException:
            if child.poll() is None:
                child.kill();child.wait(timeout=10)
            raise
    payload = {"status": status, "exit_code": code, "pid": child.pid,
               "seconds": time.monotonic() - started, "wall_limit_seconds": 43320,
               "context_wall_limit_seconds":7200,
               "command": command}
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)
    if code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
