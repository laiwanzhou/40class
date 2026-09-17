"""Bounded runtime diagnostics using synthetic labels; never a scoring experiment."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import faulthandler
import json
from pathlib import Path
import subprocess
import sys
import time


def child(device):
    faulthandler.dump_traceback_later(20, repeat=True)
    import numpy as np
    import torch
    from .p428_skeleton_training import train_trajectory
    torch.set_num_threads(4)
    class Synthetic:
        labels = None
        def __init__(self, n):
            self.x = torch.randn(n, 12, 17, 4)
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return {"skeleton": self.x[i]}
    print(json.dumps({"event": "imports_complete", "device": device}), flush=True)
    started = time.monotonic()
    if device == "cached":
        from .p90_teacher_common import load_protocol
        from .p428_skeleton_data import P428SkeletonDataset
        protocol=load_protocol()
        cache=Path(__file__).resolve().parent/"cache/aligned_192x144"
        ids=protocol.sample_ids.astype(str)
        train=P428SkeletonDataset(cache,ids,np.arange(64),augment=True)
        pred=P428SkeletonDataset(cache,ids,np.arange(64,67))
        print(json.dumps({"event":"cache_ready"}),flush=True)
        actual_device="cuda"
    else:
        train,pred=Synthetic(64),Synthetic(3); actual_device=device
    p, state, d = train_trajectory(train, pred, np.arange(64)%40,
        epochs=1, device=actual_device, deadline=started+40)
    print(json.dumps({"event": "fit_complete", "seconds": time.monotonic()-started,
        "shape": list(p.shape), "parameters": sum(v.numel() for v in state.values()), "diagnostics": d}), flush=True)
    faulthandler.cancel_dump_traceback_later()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--child", choices=["cpu","cuda","cached"])
    parser.add_argument("--out-dir"); args=parser.parse_args()
    if args.child:
        child(args.child); return
    if not args.out_dir: parser.error("--out-dir required")
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    results=[]
    for device in ("cpu", "cuda", "cached"):
        started=time.monotonic()
        with (out/f"{device}.log").open("w", encoding="utf-8") as log:
            proc=subprocess.Popen([sys.executable,"-u","-m","aligned_multimodal.p428_runtime_probe","--child",device],
                stdout=log, stderr=subprocess.STDOUT)
            try:
                code=proc.wait(timeout=60); status="complete" if code==0 else "failed"
            except subprocess.TimeoutExpired:
                proc.kill(); code=proc.wait(timeout=10); status="watchdog_timeout"
        result={"device":device,"status":status,"exit_code":code,"seconds":time.monotonic()-started}
        results.append(result); print(json.dumps(result),flush=True)
    (out/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__": main()
