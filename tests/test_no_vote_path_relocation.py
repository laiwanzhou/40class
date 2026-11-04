"""A root rename preserves logical addresses, never content/recipe checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml


def _module():
    from src.experiments import path_relocation
    return path_relocation


def _alias(old, new):
    if os.name == 'nt':
        subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                        "New-Item -ItemType Junction -Path $env:AUDIT_ALIAS -Target $env:AUDIT_TARGET | Out-Null"],
                       env=dict(os.environ, AUDIT_ALIAS=str(old), AUDIT_TARGET=str(new)),
                       check=True, capture_output=True)
    else:
        old.symlink_to(new, target_is_directory=True)


def _prerequisites(p):
    root = p.run_root / 'protocol'
    reference = {'record_path': str(p.run_root / 'artifacts/not-consumed.json'), 'sha256': 'a' * 64}
    outputs = {pop: {'skeleton': reference, 'imu': reference} for pop in p.partitions}
    def put(name, body):
        path = root / name; path.write_text(json.dumps(body))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    ready = put('tiny_motion_ready.json', {'outputs': outputs, 'models': {}, 'protocol_sha256': p.identity()})
    review = put('tiny_review.json', {'verdict': 'GO', 'unresolved_p1_p2': [], 'task8_revision': 'ir_supervision_v2', 'ready': ready})
    complete = put('tiny_complete.json', {'complete': True, 'outputs': outputs, 'protocol_sha256': p.identity()})
    parts = {pop: {'prediction': reference} for pop in p.partitions}
    student = put('task5_artifacts_ready.json', {'partitions': parts})
    put('task5_complete.json', {'complete': True, 'independent_review': {'status': 'GO'},
        'protocol_sha256': p.identity(), 'artifacts_ready_sha256': student['sha256'], 'partitions': parts})
    put('task8_current.json', {'complete': complete, 'independent_review': review, 'verdict': 'GO',
        'task8_revision': 'ir_supervision_v2', 'outputs': outputs, 'models': {}, 'protocol_sha256': p.identity()})


def test_renamed_protocol_and_reference_keep_identity_and_verify_payload(no_vote_fixture, monkeypatch):
    m = _module()
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_types import canonical_hash
    from src.experiments.artifact_record import ArtifactRegistry, VerificationContext
    from src.experiments.simple_fusion import current_inputs
    config, payload = no_vote_fixture
    p = load_protocol(config)
    _prerequisites(p)
    _, proofs_before = current_inputs(p, VerificationContext(p))
    file = p.run_root / 'tiny.json'
    file.write_text('{"value":1}')
    ref = ArtifactRegistry(p).register(stage='tiny', kind='raw_cache', phase='raw', files=[file])
    original_identity, original_ref_hash = p.identity(), canonical_hash(ref)
    original_bytes = ref.record_path.read_bytes()
    old = config.parent
    new = old.with_name(old.name + '-renamed')
    old.rename(new)
    _alias(old, new)
    monkeypatch.setattr(m, 'active_relocation', lambda: m.RootRelocation(old, new))
    moved = load_protocol(new / config.name)
    assert moved.run_root.is_relative_to(new)
    assert moved.identity() == original_identity
    _, proofs_after = current_inputs(moved, VerificationContext(moved))
    assert proofs_after == proofs_before
    assert canonical_hash(ref) == original_ref_hash
    registry = ArtifactRegistry(moved)
    record = registry.verify(ref)
    assert ref.record_path.read_bytes() == original_bytes
    assert registry.verify_file(record, new / 'run/tiny.json').read_text() == '{"value":1}'
    (new / 'run/tiny.json').write_text('{"value":2}')
    with pytest.raises(ValueError, match='hash'):
        registry.verify_file(record, new / 'run/tiny.json')
    payload['recipe']['seed'] += 1
    (new / config.name).write_text(yaml.safe_dump(payload))
    assert load_protocol(new / config.name).identity() != original_identity


def test_mapping_requires_same_directory_and_keeps_unrelated_paths(tmp_path):
    m = _module()
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    with pytest.raises(ValueError, match='same|alias'):
        m.RootRelocation(old, new)
    old.rmdir(); _alias(old, new)
    relocation = m.RootRelocation(old, new)
    assert relocation.logical(new / 'a') == old / 'a'
    unrelated = tmp_path / 'new-other/a'
    assert relocation.logical(unrelated) == unrelated.resolve()


def test_migration_manifest_rejects_anchor_drift_and_parent_escape(tmp_path):
    m = _module()
    old, new = tmp_path / 'old', tmp_path / 'new'
    new.mkdir(); _alias(old, new)
    file = new / 'config.json'; file.write_bytes(b'unchanged')
    manifest = new / 'migration.json'
    body = {'schema_version': 1, 'historical_root': str(old), 'runtime_root': str(new),
            'anchors': {'config.json': hashlib.sha256(file.read_bytes()).hexdigest()}}
    manifest.write_text(json.dumps(body))
    assert m.load_relocation(manifest).logical(file) == old / 'config.json'
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='anchor'):
        m.load_relocation(manifest)
    body['anchors'] = {'../outside.json': 'a' * 64}
    manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='anchor|relative'):
        m.load_relocation(manifest)
