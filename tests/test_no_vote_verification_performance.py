"""Bounded verification keeps role checks without reopening raw ancestry."""
from pathlib import Path

import pytest


def test_cached_model_and_new_registration_do_not_hash_raw_ancestors(no_vote_fixture, tmp_path, monkeypatch):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments import artifact_record as ar
    registry = ar.ArtifactRegistry(load_protocol(no_vote_fixture[0]))
    raw = tmp_path / 'raw.png'; raw.write_bytes(b'raw input')
    cache = registry.root / 'cache.json'; cache.write_text('{}')
    parent = registry.register(stage='p28', kind='raw_cache', phase='raw', files=[cache], raw_inputs=[raw])
    roi = registry.register(stage='p29', kind='raw_cache', phase='raw', files=[cache], parents=[parent], raw_inputs=[raw])
    model_file = registry.root / 'head.json'; model_file.write_text('{}')
    model = registry.register(stage='A1', kind='supervised_model', phase='select', files=[model_file],
        parents=[roi], fit_users=registry.protocol.partitions['train12'].users)
    original = ar.sha256_file
    reads = []
    def guarded(path):
        reads.append(Path(path))
        if Path(path) in {raw, cache}: raise AssertionError('reopened raw ancestry')
        return original(path)
    monkeypatch.setattr(ar, 'sha256_file', guarded)
    registry.verify(model, 'A1', 'select')
    child_file = registry.root / 'child.json'; child_file.write_text('{}')
    child = registry.register(stage='A2', kind='supervised_model', phase='select', files=[child_file],
        parents=[model], fit_users=registry.protocol.partitions['train12'].users)
    assert registry.verify(child).phase == 'select'
    assert raw not in reads and cache not in reads


def test_protocol_load_reads_receipts_not_all_initializer_and_source_contents(no_vote_fixture, monkeypatch):
    from src.experiments import no_vote_weights as weights, teammate_source as source
    from src.experiments.no_vote_protocol import load_protocol
    config, payload = no_vote_fixture
    module = Path(payload['source']['root']) / 'aligned_multimodal/unit_ops.py'
    weight_files = {Path(entry['local_path']) for name, entry in payload['weights'].items() if name != 'videomae'}
    def guard(original):
        def checked(path):
            if Path(path) == module or Path(path) in weight_files:
                raise AssertionError('protocol scanned unused asset content')
            return original(path)
        return checked
    monkeypatch.setattr(source, 'sha256_file', guard(source.sha256_file))
    monkeypatch.setattr(weights, 'sha256_file', guard(weights.sha256_file))
    assert len(load_protocol(config).identity()) == 64


def test_symbol_loading_hashes_requested_module_not_unused_source(no_vote_fixture, monkeypatch):
    from src.experiments import teammate_source as source
    _, payload = no_vote_fixture
    root = Path(payload['source']['root']); manifest = Path(payload['source']['manifest'])
    report = source.verify_teammate_source(root, manifest)
    original = source.sha256_file
    unused = root / 'aligned_multimodal/helper.py'
    def guarded(path):
        if Path(path) == unused: raise AssertionError('unused source was scanned')
        return original(path)
    monkeypatch.setattr(source, 'sha256_file', guarded)
    assert source.load_teammate_symbol(report, 'unit_ops', 'VALUE') == 7


def test_direct_input_mutation_is_still_rejected(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import ArtifactRegistry
    registry = ArtifactRegistry(load_protocol(no_vote_fixture[0]))
    path = registry.root / 'model.json'; path.write_text('{}')
    ref = registry.register(stage='A1', kind='supervised_model', phase='select', files=[path],
        fit_users=registry.protocol.partitions['train12'].users)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='hash'): registry.verify(ref)
