from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


def _write(config, payload):
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_protocol_hash_across_processes(no_vote_fixture):
    config, _ = no_vote_fixture
    code = "from src.experiments.no_vote_protocol import load_protocol; import sys; print(load_protocol(sys.argv[1]).identity())"
    hashes = []
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(config)],
                                env=env, capture_output=True, text=True, check=True)
        hashes.append(result.stdout.strip())
    assert len(set(hashes)) == 1
    assert len(hashes[0]) == 64


@pytest.mark.parametrize("change", ["seed", "budget", "users", "weight", "source"])
def test_identity_changes_with_each_training_dependency(no_vote_fixture, change):
    from src.experiments.no_vote_protocol import load_protocol
    config, payload = no_vote_fixture
    before = load_protocol(config)
    if change == "seed":
        payload["recipe"]["seed"] += 1
    elif change == "budget":
        payload["recipe"]["visual_student"]["epochs"]["max"] = 17
    elif change == "users":
        payload["partitions"]["final4"]["users"] = ["target2"]
    elif change == "weight":
        import hashlib
        receipt = Path(payload["run_root"]) / "protocol/weights_manifest.json"
        body = json.loads(receipt.read_text())
        entry = body["weights"]["mc3"]["files"][0]
        path = Path(entry["path"])
        path.write_bytes(b"changed fixture")
        entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        receipt.write_text(json.dumps(body))
    else:
        import hashlib
        manifest = Path(payload["source"]["manifest"])
        body = json.loads(manifest.read_text())
        entry = body["files"][0]
        path = Path(payload["source"]["root"]) / entry["path"]
        path.write_text("VALUE = 10\n")
        entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(body))
        payload["source"]["expected_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _write(config, payload)
    assert load_protocol(config).identity() != before.identity()


@pytest.mark.parametrize("case", ["overlap", "refit", "duplicate", "boolean_count", "final_labels", "nested_final_labels", "nan"])
def test_invalid_protocol_inputs_are_rejected(no_vote_fixture, case):
    from src.experiments.no_vote_protocol import load_protocol
    config, payload = no_vote_fixture
    if case == "overlap": payload["partitions"]["final4"]["users"] = ["dev1"]
    elif case == "refit": payload["partitions"]["refit14"]["users"] = ["fit1"]
    elif case == "duplicate": payload["partitions"]["train12"]["users"] = ["fit1", "fit1"]
    elif case == "boolean_count": payload["partitions"]["train12"]["expected_rows"] = True
    elif case == "final_labels": payload["supervised_labels"]["final4"] = "private/final_labels.csv"
    elif case == "nested_final_labels": payload["recipe"]["hidden"] = {"final_label_path": "private/labels.csv"}
    else: payload["recipe"]["temperature"] = float("nan")
    _write(config, payload)
    with pytest.raises(ValueError): load_protocol(config)


def test_protocol_is_immutable_after_loading(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    config, _ = no_vote_fixture
    protocol = load_protocol(config)
    with pytest.raises(TypeError): protocol.recipe["seed"] = 1
    with pytest.raises(TypeError): protocol.recipe["visual_student"]["epochs"]["max"] = 1


def test_draft_cannot_be_used_as_verified_protocol(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    config, _ = no_vote_fixture
    protocol = load_protocol(config, verify_assets=False)
    with pytest.raises(ValueError, match="verified"): protocol.identity()


def test_formal_config_preserves_eighteen_user_ownership():
    from src.experiments.no_vote_protocol import load_protocol
    config = Path(__file__).resolve().parents[1] / "configs/experiments/teammate_single_teacher_fixed_split.yaml"
    protocol = load_protocol(config, verify_assets=False)
    groups = protocol.partitions
    assert set(groups["train12"].users) == {"user1","user2","user3","user5","user8","user9","user16","user18","user19","user20","user21","user22"}
    assert set(groups["development2"].users) == {"user6", "user7"}
    assert set(groups["final4"].users) == {"user4", "user17", "user23", "user24"}
    assert set(groups["refit14"].users) == set(groups["train12"].users) | set(groups["development2"].users)
    assert [groups[x].expected_rows for x in ("train12","development2","refit14","final4")] == [2039,388,2427,609]
    assert "final4" not in protocol.supervised_labels


def test_prediction_rejects_wrong_rows_classes_and_zero_probabilities(no_vote_rows, tmp_path):
    from src.experiments.no_vote_types import ArtifactRef, Prediction, RowIndex
    rows = RowIndex(no_vote_rows["sample_ids"], no_vote_rows["user_ids"], no_vote_rows["class_ids"])
    ref = ArtifactRef(tmp_path / "record.json", "a" * 64)
    with pytest.raises(ValueError):
        Prediction(rows, np.zeros((2,40)), np.zeros((2,40)), np.ones(2, dtype=bool), ref)
    with pytest.raises(ValueError): RowIndex(("same","same"), ("a","b"), tuple(range(40)))
    with pytest.raises(ValueError): RowIndex(("a",), ("a",), tuple(reversed(range(40))))
    prediction = Prediction(rows, np.zeros((2,40)), no_vote_rows["probabilities"], np.array([True,False]), ref)
    assert prediction.valid.tolist() == [True,False]
