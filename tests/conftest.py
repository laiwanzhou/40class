from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml


@pytest.fixture
def no_vote_fixture(tmp_path: Path) -> tuple[Path, dict]:
    """Isolated on-disk inputs, never competition data or network weights."""
    project = tmp_path / "teammate_teacher" / "project"
    module = project / "aligned_multimodal" / "unit_ops.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 7\nif __name__ == '__main__':\n    raise RuntimeError('main executed')\n")
    helper = module.parent / "helper.py"
    helper.write_text("VALUE = 8\n")
    manifest = project.parent / "source_manifest.json"
    entries = []
    for path in (module, helper):
        raw = path.read_bytes()
        entries.append({"path": path.relative_to(project).as_posix(), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    manifest.write_text(json.dumps({"file_count": 2, "files": entries, "training_executed": False}))
    run = tmp_path / "run"
    weights = {}
    receipt = {"schema_version": 1, "fixture": True, "weights": {}}
    origins = {
        "videomae": {"provider": "huggingface", "repo_id": "MCG-NJU/videomae-large-finetuned-kinetics",
                     "revision": "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"},
        "mc3": {"provider": "url", "url": "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth"},
        "yolo": {"provider": "url", "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"},
    }
    for name in origins:
        folder = run / "weights" / name
        folder.mkdir(parents=True)
        path = folder / ("model.safetensors" if name == "videomae" else name + ".pth")
        path.write_bytes((name + " fixture").encode())
        paths = [path]
        if name == "videomae":
            for filename in ("config.json", "preprocessor_config.json"):
                extra = folder / filename
                extra.write_text("{}")
                paths.append(extra)
        local = folder if name == "videomae" else path
        weights[name] = {**origins[name], "local_path": str(local)}
        receipt["weights"][name] = {
            "origin": origins[name], "local_path": str(local),
            "files": [{"path": str(p), "bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            "validation": {"status": "fixture"},
        }
    protocol = run / "protocol"
    protocol.mkdir()
    (protocol / "weights_manifest.json").write_text(json.dumps(receipt))
    populations = {
        "train12": {"users": ["fit1"], "expected_rows": 40},
        "development2": {"users": ["dev1"], "expected_rows": 40},
        "refit14": {"users": ["dev1", "fit1"], "expected_rows": 80},
        "final4": {"users": ["target1"], "expected_rows": 40},
    }
    payload = {
        "schema_version": 1, "execution_kind": "fixture", "project_root": ".",
        "run_id": "unit", "run_root": str(run), "partitions": populations,
        "source": {"root": str(project), "manifest": str(manifest), "expected_files": 2,
                   "expected_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "weights": weights,
        "public_manifests": {k: str(protocol / (k + ".csv")) for k in populations},
        "supervised_labels": {k: str(protocol / (k + "_labels.csv")) for k in populations if k != "final4"},
        "recipe": {"seed": 20260811, "class_ids": list(range(40)),
                   "visual_student": {"epochs": {"max": 18}},
                   "unordered": {"fit1", "dev1", "target1"}},
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    yield config, payload
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin and Path(origin).resolve().is_relative_to(project):
            sys.modules.pop(name, None)


@pytest.fixture
def no_vote_rows():
    import numpy as np
    return {"sample_ids": ("id-a", "id-b"), "user_ids": ("fit1", "fit1"),
            "class_ids": tuple(range(40)), "probabilities": np.full((2, 40), .025),
            "labels": np.array([0, 1], dtype=np.int64)}


@pytest.fixture
def no_vote_fake_dag():
    """Literal ancestry inputs for the later registry tests; no registry code."""
    return {
        "public": {"stage": "initializers", "kind": "public_weights", "fit_users": [], "parents": []},
        "select": {"stage": "A1", "kind": "supervised_model", "fit_users": ["fit1"], "parents": ["public"]},
        "refit": {"stage": "A1", "kind": "supervised_model", "fit_users": ["dev1", "fit1"], "parents": ["public"]},
    }
