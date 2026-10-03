"""Public-weight resolution, receipts and version-aware initialization checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .teammate_source import sha256_file


VIDEOMAE_REPO = "MCG-NJU/videomae-large-finetuned-kinetics"
VIDEOMAE_REVISION = "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"
VIDEOMAE_SHA256 = "92d33bedfe4f171705af418b5ddf059a2beb8ef4d1532621c2ae0a6693dbb8bc"
MC3_URL = "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
MC3_SHA256 = "a90a0ba35ca1242d15b77511ff28bfb29cc596988b5ea36081042f8e2f54212b"
YOLO_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
VIDEOMAE_FILE_HASHES = {
    "model.safetensors": VIDEOMAE_SHA256,
    "config.json": "6f2007409b792e4843725a76fcf70ba215c2a53df575c74a5dc8821b7b249ca0",
    "preprocessor_config.json": "c3aa722f22a7ff0d234d407862025fe47672f74f46f92c91f757f7a7354107c2",
}


def verify_public_digest(role: str, filename: str, digest: str) -> None:
    expected = (VIDEOMAE_FILE_HASHES.get(filename) if role == "videomae" else
                {"mc3": MC3_SHA256, "yolo": YOLO_SHA256}.get(role))
    if expected is None or digest != expected:
        raise ValueError(f"public initializer hash mismatch: {role}/{filename}")


def weight_origin(spec: dict) -> dict:
    provider = spec.get("provider")
    if provider == "huggingface":
        return {"provider":provider, "repo_id":spec.get("repo_id"), "revision":spec.get("revision")}
    if provider == "url":
        return {"provider":provider, "url":spec.get("url")}
    raise ValueError("unsupported public-weight provider")


def resolve_model_directory(model: str | Path, *, revision: str | None = None,
                            snapshot_loader=None) -> Path:
    path = Path(model)
    is_local = isinstance(model, Path) or path.is_dir() or path.is_absolute() or str(model).startswith(("./", "../", ".\\", "..\\")) or bool(re.match(r"^[A-Za-z]:",str(model)))
    if is_local:
        if not path.is_dir():
            raise FileNotFoundError(f"local model directory does not exist: {path}")
        resolved = path.resolve()
    else:
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Hub models require an explicit immutable revision")
        if snapshot_loader is None:
            from huggingface_hub import snapshot_download
            snapshot_loader = snapshot_download
        resolved = Path(snapshot_loader(repo_id=str(model), revision=revision,
                                       allow_patterns=["config.json","preprocessor_config.json","model.safetensors"])).resolve()
    if any(not (resolved/name).is_file() for name in ("config.json","preprocessor_config.json","model.safetensors")):
        raise FileNotFoundError("local model directory is missing config/processor/weights")
    return resolved


def verify_weights_manifest(path: Path, specs: dict, *, allow_fixture: bool = False,
                            verify_contents: bool = True) -> str:
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema_version") != 1 or set(body.get("weights", {})) != {"videomae","mc3","yolo"}:
        raise ValueError("invalid weights manifest")
    fixture = body.get("fixture", False)
    if fixture and not allow_fixture:
        raise ValueError("formal protocol cannot use fixture weights")
    seen = set()
    for name in ("videomae","mc3","yolo"):
        spec, entry = specs[name], body["weights"][name]
        if entry.get("origin") != weight_origin(spec):
            raise ValueError(f"weight origin mismatch: {name}")
        local = Path(spec["local_path"]).resolve()
        if Path(entry.get("local_path", "")).resolve() != local:
            raise ValueError(f"weight location mismatch: {name}")
        expected_status = "fixture" if fixture else "verified"
        if entry.get("validation", {}).get("status") != expected_status:
            raise ValueError(f"weight initialization validation incomplete: {name}")
        files = entry.get("files", [])
        if not isinstance(files, list) or not files:
            raise ValueError(f"missing weight file records: {name}")
        names = set()
        for item in files:
            file = Path(item["path"]).resolve()
            inside = file.is_relative_to(local) if name == "videomae" else file == local
            if not inside or str(file) in seen:
                raise ValueError("weight receipt contains duplicate or unexpected files")
            seen.add(str(file)); names.add(file.name)
            if not file.is_file() or file.stat().st_size != item.get("bytes") or (verify_contents and sha256_file(file) != item.get("sha256")):
                raise ValueError(f"weight file mismatch: {file.name}")
            if not fixture:
                verify_public_digest(name, file.name, item["sha256"])
        expected_names = {"model.safetensors","config.json","preprocessor_config.json"} if name == "videomae" else {local.name}
        if names != expected_names:
            raise ValueError(f"weight file set mismatch: {name}")
    if not fixture:
        expected = {"videomae":{"provider":"huggingface","repo_id":VIDEOMAE_REPO,"revision":VIDEOMAE_REVISION},
                    "mc3":{"provider":"url","url":MC3_URL}, "yolo":{"provider":"url","url":YOLO_URL}}
        if any(weight_origin(specs[n]) != expected[n] for n in expected):
            raise ValueError("formal public initializer pin changed")
    return sha256_file(path)


def validate_attention_biases(model, checkpoint: Path) -> dict:
    import torch
    from safetensors import safe_open
    restored = verified = 0
    maximum = 0.0
    formats = set()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as weights, torch.no_grad():
        for i, layer in enumerate(model.videomae.encoder.layer):
            attention = layer.attention.attention
            prefix = f"videomae.encoder.layer.{i}.attention.attention."
            q, v = weights.get_tensor(prefix+"q_bias"), weights.get_tensor(prefix+"v_bias")
            if hasattr(attention,"q_bias") and attention.q_bias is not None:
                formats.add("native_split")
                actual_q, actual_v = attention.q_bias, attention.v_bias
            else:
                formats.add("query_key_value")
                if any(getattr(attention,name).bias is None for name in ("query","key","value")):
                    raise ValueError("unsupported VideoMAE attention bias layout")
                attention.query.bias.copy_(q)
                attention.value.bias.copy_(v)
                attention.key.bias.zero_()
                actual_q, actual_v = attention.query.bias, attention.value.bias
                restored += 2
            for actual, expected in ((actual_q,q),(actual_v,v)):
                if actual.shape != expected.shape:
                    raise ValueError("VideoMAE attention bias shape mismatch")
                maximum = max(maximum,float((actual.detach().cpu().float()-expected.float()).abs().max()))
                verified += 1
            if attention.key.bias is not None:
                maximum = max(maximum,float(attention.key.bias.detach().cpu().abs().max()))
    if verified != 2*len(model.videomae.encoder.layer) or maximum != 0:
        raise ValueError("VideoMAE attention bias verification mismatch")
    return {"formats":sorted(formats), "verified_tensors":verified,
            "restored_tensors":restored, "maximum_difference":maximum,
            "key_bias_policy":"absent or exact zero"}
