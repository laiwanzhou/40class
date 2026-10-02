from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _verify(payload):
    from src.experiments.teammate_source import verify_teammate_source
    source = payload["source"]
    return verify_teammate_source(Path(source["root"]), Path(source["manifest"]),
                                  expected_count=source["expected_files"], expected_sha256=source["expected_sha256"])


def test_source_hash_drift_is_rejected(no_vote_fixture):
    _, payload = no_vote_fixture
    report = _verify(payload)
    assert report.file_count == 2
    path = Path(payload["source"]["root"]) / "aligned_multimodal/unit_ops.py"
    path.write_text("VALUE = 99\n")
    with pytest.raises(ValueError, match="mismatch"): _verify(payload)


def test_import_requires_verified_origin_and_does_not_execute_main(no_vote_fixture):
    from src.experiments.teammate_source import load_teammate_symbol
    _, payload = no_vote_fixture
    report = _verify(payload)
    value = load_teammate_symbol(report, "unit_ops", "VALUE")
    assert value == 7
    assert not list(Path(payload["source"]["root"]).rglob("*.pyc"))


def test_unlisted_module_is_not_imported(no_vote_fixture):
    from src.experiments.teammate_source import load_teammate_symbol
    _, payload = no_vote_fixture
    report = _verify(payload)
    (Path(payload["source"]["root"]) / "aligned_multimodal/not_listed.py").write_text("raise RuntimeError('executed')")
    with pytest.raises(ValueError, match="manifest"): load_teammate_symbol(report, "not_listed", "VALUE")


def test_same_module_name_from_another_directory_is_rejected(no_vote_fixture, monkeypatch, tmp_path):
    from src.experiments.teammate_source import load_teammate_symbol
    _, payload = no_vote_fixture
    fake = ModuleType("unit_ops")
    fake.__file__ = str(tmp_path / "other/unit_ops.py")
    fake.VALUE = 999
    monkeypatch.setitem(sys.modules, "unit_ops", fake)
    with pytest.raises(ImportError, match="origin"):
        load_teammate_symbol(_verify(payload), "unit_ops", "VALUE")


def _replace_source_module(payload, source):
    manifest = Path(payload["source"]["manifest"])
    body = json.loads(manifest.read_text())
    entry = body["files"][0]
    path = Path(payload["source"]["root"])/entry["path"]
    path.write_text(source)
    entry.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest.write_text(json.dumps(body))
    payload["source"]["expected_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_verified_top_module_rejects_foreign_cached_dependency(no_vote_fixture, monkeypatch, tmp_path):
    from src.experiments.teammate_source import load_teammate_symbol
    _, payload = no_vote_fixture
    _replace_source_module(payload, "from helper import VALUE\n")
    foreign = ModuleType("helper")
    foreign.__file__ = str(tmp_path / "other/helper.py")
    foreign.VALUE = 999
    monkeypatch.setitem(sys.modules,"helper",foreign)
    with pytest.raises(ImportError,match="origin"):
        load_teammate_symbol(_verify(payload),"unit_ops","VALUE")


def test_loader_validates_a_real_transitive_dependency(no_vote_fixture):
    from src.experiments.teammate_source import load_teammate_symbol
    _, payload = no_vote_fixture
    _replace_source_module(payload,"from helper import VALUE\n")
    assert load_teammate_symbol(_verify(payload),"unit_ops","VALUE") == 8
    expected = Path(payload["source"]["root"])/"aligned_multimodal/helper.py"
    assert Path(sys.modules["helper"].__file__).resolve() == expected.resolve()


def test_exported_callable_cannot_be_silently_replaced_by_foreign_code(no_vote_fixture, monkeypatch, tmp_path):
    from src.experiments.teammate_source import load_teammate_symbol
    _, payload = no_vote_fixture
    _replace_source_module(payload,"from external_ops import calculate\n")
    foreign = ModuleType("external_ops")
    foreign.__file__ = str(tmp_path/"other/external_ops.py")
    exec(compile("def calculate(): return 999",foreign.__file__,"exec"),foreign.__dict__)
    monkeypatch.setitem(sys.modules,"external_ops",foreign)
    with pytest.raises(ImportError,match="origin"):
        load_teammate_symbol(_verify(payload),"unit_ops","calculate")


def test_manifest_count_duplicates_and_escape_paths_are_rejected(no_vote_fixture):
    from src.experiments.teammate_source import verify_teammate_source
    _, payload = no_vote_fixture
    path = Path(payload["source"]["manifest"])
    original = json.loads(path.read_text())
    root = Path(payload["source"]["root"])
    for mutation in ("count", "duplicate", "escape"):
        body = json.loads(json.dumps(original))
        if mutation == "count": body["file_count"] = 3
        elif mutation == "duplicate": body["files"][1] = body["files"][0]
        else: body["files"][0]["path"] = "../outside.py"
        path.write_text(json.dumps(body))
        with pytest.raises(ValueError): verify_teammate_source(root, path)
