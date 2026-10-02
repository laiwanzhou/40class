"""Acquire/verify public initializers. Does not prepare data or train models."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.no_vote_protocol import load_protocol, read_config
from src.experiments.no_vote_types import write_json
from src.experiments.no_vote_weights import (MC3_URL, MC3_SHA256, VIDEOMAE_REPO, VIDEOMAE_REVISION,
    VIDEOMAE_SHA256, VIDEOMAE_FILE_HASHES, YOLO_URL, YOLO_SHA256, resolve_model_directory, validate_attention_biases,
    verify_weights_manifest, weight_origin)
from src.experiments.teammate_source import (load_teammate_symbol, sha256_file,
                                           verify_teammate_source)


def progress(**items):
    print(json.dumps(items, ensure_ascii=False), flush=True)


def download_file(url: str, destination: Path, *, expected_hash: str | None = None):
    import requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".downloading")
    try:
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            done, reported = 0, 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk); done += len(chunk)
                    if done-reported >= 16*1024*1024:
                        progress(event="download", file=destination.name, bytes=done, total=total)
                        reported = done
        digest = sha256_file(temporary)
        if expected_hash and not digest.startswith(expected_hash):
            raise ValueError(f"download checksum mismatch: {destination.name}")
        temporary.replace(destination)
        progress(event="download_complete", file=destination.name, bytes=done, sha256=digest)
    finally:
        temporary.unlink(missing_ok=True)


def validate_models(weights: dict) -> dict:
    import torch
    import transformers
    import torchvision
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    from torchvision.models.video import mc3_18
    from ultralytics import YOLO
    result = {}
    folder = resolve_model_directory(weights["videomae"])
    model = VideoMAEForVideoClassification.from_pretrained(
        str(folder), local_files_only=True, use_safetensors=True)
    model.eval().requires_grad_(False)
    VideoMAEImageProcessor.from_pretrained(str(folder), local_files_only=True)
    if model.config.hidden_size != 1024 or model.config.num_labels != 400 or model.config.num_frames != 16:
        raise ValueError("VideoMAE initializer architecture changed")
    result["videomae"] = {"status":"verified", "classes":400,"hidden_size":1024,"frames":16,
        "parameters":sum(p.numel() for p in model.parameters()),
        "attention_biases":validate_attention_biases(model, folder/"model.safetensors"),
        "transformers":transformers.__version__,"torch":torch.__version__}
    del model
    gc.collect()
    model = mc3_18(weights=None)
    state = torch.load(weights["mc3"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    result["mc3"] = {"status":"verified","classes":model.fc.out_features,
                     "parameters":sum(p.numel() for p in model.parameters()),
                     "torchvision":torchvision.__version__,"torch":torch.__version__}
    if model.fc.out_features != 400:
        raise ValueError("MC3 initializer class count changed")
    del model, state
    gc.collect()
    yolo = YOLO(str(weights["yolo"]), verbose=False)
    if yolo.task != "pose" or list(yolo.model.kpt_shape) != [17,3]:
        raise ValueError("YOLO initializer pose architecture changed")
    result["yolo"] = {"status":"verified","task":"pose","keypoint_shape":[17,3],
                      "parameters":sum(p.numel() for p in yolo.model.parameters())}
    del yolo
    gc.collect()
    return result


def acquire(config: Path, *, verify_only: bool = False, source_root: Path | None = None) -> dict:
    protocol = load_protocol(config, verify_assets=False, source_root=source_root)
    if protocol.recipe["execution_kind"] != "formal":
        raise ValueError("acquisition is for formal public weights; fixture receipts are test-only")
    raw, base = read_config(config)
    source = raw["source"]
    manifest = protocol.source_root.parent/"source_manifest.json" if source_root else (
        Path(source["manifest"]) if Path(source["manifest"]).is_absolute() else base/source["manifest"])
    report = verify_teammate_source(protocol.source_root, manifest,
        expected_count=source["expected_files"], expected_sha256=source["expected_sha256"])
    restore = load_teammate_symbol(report,"build_p46_videomae_cache","restore_legacy_attention_biases")
    if not callable(restore):
        raise ValueError("verified attention-bias source symbol missing")
    specs = protocol.recipe["weight_specs"]
    expected = {"videomae":{"provider":"huggingface","repo_id":VIDEOMAE_REPO,"revision":VIDEOMAE_REVISION},
                "mc3":{"provider":"url","url":MC3_URL},"yolo":{"provider":"url","url":YOLO_URL}}
    if any(weight_origin(specs[n]) != expected[n] for n in expected):
        raise ValueError("public initializer origin pin changed")
    receipt_path = protocol.run_root/"protocol/weights_manifest.json"
    if verify_only:
        verify_weights_manifest(receipt_path, specs)
    else:
        try:
            verify_weights_manifest(receipt_path, specs)
            progress(event="reuse_verified_receipt")
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            folder = protocol.weights["videomae"]
            folder.mkdir(parents=True,exist_ok=True)
            model_path = folder/"model.safetensors"
            reuse_value = raw["weights"]["videomae"].get("reuse_path")
            reuse = ((base/str(reuse_value)).resolve()/"model.safetensors") if reuse_value else None
            if model_path.is_file() and sha256_file(model_path)==VIDEOMAE_SHA256:
                progress(event="reuse_local_weight", file=model_path.name)
            elif reuse and reuse.is_file() and sha256_file(reuse)==VIDEOMAE_SHA256:
                temporary = model_path.with_name(model_path.name+".building")
                shutil.copyfile(reuse,temporary)
                temporary.replace(model_path)
                progress(event="reuse_public_checkpoint", file=model_path.name, sha256=VIDEOMAE_SHA256)
            else:
                download_file(f"https://huggingface.co/{VIDEOMAE_REPO}/resolve/{VIDEOMAE_REVISION}/model.safetensors",
                              model_path, expected_hash=VIDEOMAE_SHA256)
            for name in ("config.json","preprocessor_config.json"):
                download_file(f"https://huggingface.co/{VIDEOMAE_REPO}/resolve/{VIDEOMAE_REVISION}/{name}",folder/name,
                              expected_hash=VIDEOMAE_FILE_HASHES[name])
            mc3 = protocol.weights["mc3"]
            if not mc3.is_file() or sha256_file(mc3) != MC3_SHA256:
                download_file(MC3_URL,mc3,expected_hash=MC3_SHA256)
            # With no existing verified receipt, obtain the pose weight from its
            # public pinned release rather than attributing an unknown local file.
            download_file(YOLO_URL,protocol.weights["yolo"],expected_hash=YOLO_SHA256)
    progress(event="initialization_validation_start", device="cpu", training=False)
    validation = validate_models(dict(protocol.weights))
    receipt = {"schema_version":1,"fixture":False,"weights":{}}
    for name, local in protocol.weights.items():
        paths = [local/f for f in ("config.json","preprocessor_config.json","model.safetensors")] if name=="videomae" else [local]
        receipt["weights"][name] = {"origin":weight_origin(specs[name]),"local_path":str(local),
            "files":[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in paths],
            "validation":validation[name]}
    write_json(receipt_path,receipt)
    verify_weights_manifest(receipt_path,specs)
    write_json(protocol.run_root/"protocol/verified_source.json",{
        "source_root":str(report.root),"manifest_path":str(report.manifest_path),
        "manifest_sha256":report.manifest_sha256,"file_count":report.file_count,
        "requested_symbols":["build_p46_videomae_cache.restore_legacy_attention_biases"]})
    sealed = load_protocol(config,source_root=source_root)
    identity = sealed.identity()
    write_json(protocol.run_root/"protocol/resolved_protocol.json",sealed)
    completion = {"task":1,"status":"complete","protocol_sha256":identity,
        "source_files_verified":report.file_count,"weights_manifest_sha256":sha256_file(receipt_path),
        "model_initialization":validation,"training_executed":False,
        "participant_data_read":False,"task2_started":False}
    write_json(protocol.run_root/"protocol/task1_complete.json",completion)
    progress(event="task1_complete",protocol_sha256=identity, source_files_verified=report.file_count)
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--source-root",type=Path)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args(argv)
    acquire(args.config,verify_only=args.verify_only,source_root=args.source_root)


if __name__=="__main__":
    main()
