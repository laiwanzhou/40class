from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


def test_local_model_directory_never_becomes_hub_repo_id(no_vote_fixture):
    from src.experiments.no_vote_weights import resolve_model_directory
    _, payload = no_vote_fixture
    folder = Path(payload["weights"]["videomae"]["local_path"])
    def forbidden_network(**kwargs): raise AssertionError("Hub called for a local directory")
    assert resolve_model_directory(str(folder), snapshot_loader=forbidden_network) == folder.resolve()


def test_missing_local_model_does_not_fall_back_to_hub(tmp_path):
    from src.experiments.no_vote_weights import resolve_model_directory
    with pytest.raises(FileNotFoundError): resolve_model_directory(tmp_path / "missing")


def test_hub_directory_requires_explicit_revision(no_vote_fixture):
    from src.experiments.no_vote_weights import resolve_model_directory
    _, payload = no_vote_fixture
    with pytest.raises(ValueError, match="revision"): resolve_model_directory("owner/model")
    folder = Path(payload["weights"]["videomae"]["local_path"])
    def pinned_snapshot(**kwargs):
        assert kwargs["repo_id"] == "owner/model"
        assert kwargs["revision"] == "a" * 40
        return str(folder)
    assert resolve_model_directory("owner/model", revision="a"*40, snapshot_loader=pinned_snapshot) == folder.resolve()


def test_modified_weight_receipt_cannot_hide_file_drift(no_vote_fixture):
    from src.experiments.no_vote_weights import verify_weights_manifest
    _, payload = no_vote_fixture
    path = Path(payload["run_root"]) / "protocol/weights_manifest.json"
    verify_weights_manifest(path, payload["weights"], allow_fixture=True)
    Path(payload["weights"]["mc3"]["local_path"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="mismatch"): verify_weights_manifest(path, payload["weights"], allow_fixture=True)


def test_formal_weights_reject_fixture_receipt(no_vote_fixture):
    from src.experiments.no_vote_weights import verify_weights_manifest
    _, payload = no_vote_fixture
    path = Path(payload["run_root"]) / "protocol/weights_manifest.json"
    with pytest.raises(ValueError, match="fixture"): verify_weights_manifest(path, payload["weights"])


def test_weights_require_correct_revision_and_completed_validation(no_vote_fixture):
    from src.experiments.no_vote_weights import verify_weights_manifest
    _, payload = no_vote_fixture
    path = Path(payload["run_root"]) / "protocol/weights_manifest.json"
    body = json.loads(path.read_text())
    body["weights"]["videomae"]["origin"]["revision"] = "b"*40
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="origin"): verify_weights_manifest(path, payload["weights"], allow_fixture=True)


def test_public_binary_hash_requires_all_64_digits():
    from src.experiments.no_vote_weights import verify_public_digest
    # The official Torch filename contains only a prefix. The acquisition
    # receipt must not accept another file just because that prefix matches.
    with pytest.raises(ValueError, match="hash"):
        verify_public_digest("mc3", "mc3_18-a90a0ba3.pth", "a90a0ba3" + "0" * 56)
    verify_public_digest("mc3", "mc3_18-a90a0ba3.pth", "a90a0ba35ca1242d15b77511ff28bfb29cc596988b5ea36081042f8e2f54212b")


def test_public_pose_and_processor_are_content_pinned():
    from src.experiments.no_vote_weights import verify_public_digest
    with pytest.raises(ValueError): verify_public_digest("yolo", "yolo11n-pose.pt", "0" * 64)
    with pytest.raises(ValueError): verify_public_digest("videomae", "config.json", "0" * 64)


@pytest.mark.parametrize("native", [True,False])
def test_attention_bias_validation_handles_both_transformers_formats(tmp_path, native):
    from src.experiments.no_vote_weights import validate_attention_biases
    q = torch.tensor([1.,2.,3.,4.])
    v = torch.tensor([4.,3.,2.,1.])
    attention = SimpleNamespace(query=torch.nn.Linear(4,4,bias=not native),
                                key=torch.nn.Linear(4,4,bias=not native),
                                value=torch.nn.Linear(4,4,bias=not native))
    if native:
        attention.q_bias = torch.nn.Parameter(q.clone())
        attention.v_bias = torch.nn.Parameter(v.clone())
    layer = SimpleNamespace(attention=SimpleNamespace(attention=attention))
    model = SimpleNamespace(videomae=SimpleNamespace(encoder=SimpleNamespace(layer=[layer])))
    path = tmp_path / "model.safetensors"
    save_file({"videomae.encoder.layer.0.attention.attention.q_bias":q,
               "videomae.encoder.layer.0.attention.attention.v_bias":v},str(path))
    report = validate_attention_biases(model,path)
    assert report["verified_tensors"] == 2
    assert report["maximum_difference"] == 0
    if native: assert torch.equal(attention.q_bias,q)
    else:
        assert torch.equal(attention.query.bias,q)
        assert torch.count_nonzero(attention.key.bias)==0
