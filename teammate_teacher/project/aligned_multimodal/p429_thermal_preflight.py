"""Bounded source-only thermal cost preflight, not a usable expert bank."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


def child(out):
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    import numpy as np
    import torch
    from .p90_teacher_common import load_protocol
    from .p416_nested_frozen_family_router import EXCLUDED_USERS
    from .p429_thermal_data import load_path_map, ThermalDataset
    from .p429_thermal_training import train_trajectory, _WEIGHTS, _WEIGHTS_SHA256
    from .p419_vjepa_repeat_group_bridge import sha
    from thermal_baseline.thermal_oof_data import IMAGE_EXTENSIONS
    root=Path(__file__).resolve().parent.parent
    manifest=root/"thermal_baseline/data/subject_folds/fold_0.csv"
    protocol=load_protocol()
    source_users=set(protocol.users[(protocol.fold_id!=0) & ~np.isin(protocol.users,list(EXCLUDED_USERS))].astype(str))
    rows=list(csv.DictReader(manifest.open(encoding="utf-8-sig",newline="")))
    rows=[r for r in rows if r["user_id"] in source_users]
    ids=np.asarray([r["sample_id"] for r in rows]); labels=np.asarray([int(r["class_id"]) for r in rows])
    paths=load_path_map(manifest, ids)
    chosen=np.linspace(0,len(ids)-1,128).round().astype(int)
    if len(set(chosen))!=128: raise ValueError("insufficient unique source rows")
    train_ix,pred_ix=chosen[::2],chosen[1::2]
    out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),root/"docs/research/STABLE_093_P429_THERMAL_REBUILD.md",manifest,
        Path(_WEIGHTS),*[root/"aligned_multimodal"/n for n in ["p429_thermal_data.py","p429_thermal_training.py","p90_teacher_common.py","p416_nested_frozen_family_router.py"]],
        root/"thermal_baseline/thermal_oof_data.py",root/"thermal_baseline/thermal_tsm_model.py"]
    payload={"files_sha256":{str(p):sha(p) for p in files},"source_users":sorted(source_users),
        "train_ids":ids[train_ix].tolist(),"prediction_ids":ids[pred_ix].tolist(),
        "train_labels":labels[train_ix].tolist(),"prediction_labels_received":False,"purpose":"cost only; not expert bank"}
    raw={}
    for i in chosen:
        for p in paths[ids[i]].iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS: raw[str(p)]=sha(p)
    payload["raw_frame_sha256"]=raw
    (out/"registry.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    snapshot=out/"source_snapshot";snapshot.mkdir()
    for p in files:
        if p.suffix in {".py",".md"}: (snapshot/p.name).write_bytes(p.read_bytes())
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_per_process_memory_fraction(min(1.,8*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    train=ThermalDataset(paths,ids,train_ix,augment=True)
    pred=ThermalDataset(paths,ids,pred_ix)
    logits,state,d=train_trajectory(train,pred,labels[train_ix],epochs=1,deadline=started+150)
    torch.save(state,out/"diagnostic_checkpoint.pt")
    np.savez_compressed(out/"diagnostic_logits.npz",sample_ids=ids[pred_ix],logits=logits)
    summary={"seconds":time.monotonic()-started,"peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
        "diagnostics":d,"train_rows":64,"prediction_rows":64,"raw_files_hashed":len(raw),
        "weights_sha256":_WEIGHTS_SHA256,"torch_version":torch.__version__,"gpu":torch.cuda.get_device_name(),
        "python_version":sys.version,"cuda_version":torch.version.cuda,"cudnn_version":torch.backends.cudnn.version(),
        "allocator_limit_gib":8,"seed":20260720,"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "checkpoint_sha256":sha(out/"diagnostic_checkpoint.pt"),"logits_sha256":sha(out/"diagnostic_logits.npz"),
        "accuracy_evaluated":False,"test_rows_loaded":0,"expert_bank_ready":False,"target_achieved":False}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)
    faulthandler.cancel_dump_traceback_later()


def main():
    p=argparse.ArgumentParser();p.add_argument("--out-dir",required=True);p.add_argument("--child",action="store_true")
    a=p.parse_args();out=Path(a.out_dir)
    if a.child: child(out);return
    log=Path(str(out)+".process.log");report=Path(str(out)+".process.json")
    if out.exists() or log.exists() or report.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    with log.open("x",encoding="utf-8") as stream:
        proc=subprocess.Popen([sys.executable,"-u","-m","aligned_multimodal.p429_thermal_preflight","--child","--out-dir",str(out)],stdout=stream,stderr=subprocess.STDOUT)
        try:
            code=proc.wait(timeout=180);status="complete" if code==0 else "failed"
        except subprocess.TimeoutExpired:
            proc.kill();code=proc.wait(timeout=10);status="watchdog_timeout"
    result={"status":status,"exit_code":code,"seconds":time.monotonic()-started,"pid":proc.pid}
    report.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result),flush=True)
    if code: raise SystemExit(1)


if __name__=="__main__":main()
