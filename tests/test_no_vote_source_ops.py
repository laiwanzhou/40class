"""Task1 contracts: source-derived recipes, root lineage and immutable descriptors."""
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.experiments.artifact_record import ArtifactRegistry
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import write_json


SOURCE_ROOT = Path(__file__).resolve().parents[2] / '_teacher_branch_upload/teammate_teacher/project'


def module():
    assert importlib.util.find_spec('src.experiments.no_vote_source_ops') is not None, 'Task1 source_ops is not implemented'
    return importlib.import_module('src.experiments.no_vote_source_ops')


def v3_protocol(fixture):
    config, _ = fixture
    p = load_protocol(config)
    from src.experiments.teammate_source import SOURCE_MANIFEST_SHA256
    return replace(p, source_root=SOURCE_ROOT.resolve(), recipe={**p.recipe, 'source_contract_version': 3,
        'asset_bindings':{**p.recipe['asset_bindings'],'source_manifest_sha256':SOURCE_MANIFEST_SHA256}})


def test_reuse_public_weight_receipt_without_large_content_reads(no_vote_fixture, monkeypatch):
    import src.experiments.no_vote_weights as weights
    config,body=no_vote_fixture
    old=load_protocol(config)
    destination=old.run_root.parent/'new-run'
    specs={name:dict(spec,local_path=str(destination/'weights'/name/(Path(spec['local_path']).name)
        if name!='videomae' else destination/'weights'/name)) for name,spec in body['weights'].items()}
    real_hash=weights.sha256_file
    def only_receipts(path):
        assert Path(path).name=='weights_manifest.json', 'old large weights must not be hashed again'
        return real_hash(path)
    monkeypatch.setattr(weights,'sha256_file',only_receipts)
    assert hasattr(weights,'reuse_verified_weights'), 'Task1 public receipt reuse is not implemented'
    receipt=weights.reuse_verified_weights(old.run_root/'protocol/weights_manifest.json',destination,specs,allow_fixture=True)
    assert receipt.exists()
    import os
    for name,spec in specs.items():
        old_local=Path(body['weights'][name]['local_path']);new_local=Path(spec['local_path'])
        if name=='videomae':old_local/= 'model.safetensors';new_local/= 'model.safetensors'
        assert os.path.samefile(old_local,new_local)
    assert weights.verify_weights_manifest(receipt,specs,allow_fixture=True,verify_contents=False)


def test_v3_config_binds_descriptors_and_blocks_supervised_generation(no_vote_fixture):
    op=module();config,body=no_vote_fixture
    compiled=op.compile_source_ops(SOURCE_ROOT);roster=op.compile_teacher_roster(compiled,('videomae_large',))
    root=Path(body['run_root']);a=root/'protocol/source_ops_manifest.json';b=root/'protocol/teacher_roster.json'
    write_json(a,compiled);write_json(b,roster)
    from hashlib import sha256
    body['recipe']['source_contract_version']=3
    body['recipe'].update(op.runtime_recipe_values(compiled))
    from src.experiments.teammate_source import SOURCE_MANIFEST_SHA256, SOURCE_FILE_COUNT
    body['source']={'root':str(SOURCE_ROOT),'manifest':str(SOURCE_ROOT.parent/'source_manifest.json'),
        'expected_files':SOURCE_FILE_COUNT,'expected_sha256':SOURCE_MANIFEST_SHA256}
    body['recipe']['source_ops_bindings']={'manifest_path':str(a),'manifest_sha256':sha256(a.read_bytes()).hexdigest(),
        'roster_path':str(b),'roster_sha256':sha256(b.read_bytes()).hexdigest()}
    import yaml
    config.write_text(yaml.safe_dump(body),encoding='utf-8')
    p=load_protocol(config)
    assert p.recipe['source_ops_bindings']['formal_generation_ready'] is False
    body['recipe']['visual_student']['head_learning_rate']=.5
    config.write_text(yaml.safe_dump(body),encoding='utf-8')
    with pytest.raises(ValueError,match='source recipe'):
        load_protocol(config)
    body['recipe'].update(op.runtime_recipe_values(compiled))
    body['recipe']['retained_large_roots']=['videomae_large','vjepa2']
    config.write_text(yaml.safe_dump(body),encoding='utf-8')
    with pytest.raises(ValueError,match='source recipe'):
        load_protocol(config)
    body['recipe'].update(op.runtime_recipe_values(compiled))
    config.write_text(yaml.safe_dump(body),encoding='utf-8')
    f=root/'fake-model.json';f.write_text('{}')
    with pytest.raises(ValueError,match='source.*not ready'):
        ArtifactRegistry(p).register(stage='A1',kind='supervised_model',phase='select',files=[f],fit_users=p.partitions['train12'].users)
    a.write_text('{}')
    with pytest.raises(ValueError,match='source.*binding|descriptor.*hash'):
        load_protocol(config)


def test_final_recipe_overrides_defaults():
    compiled = module().compile_source_ops(SOURCE_ROOT)
    r = compiled['recipe']
    assert r['visual_student']['fixed_epochs'] == 16
    assert r['motion_pretrain']['fixed_epochs'] == 24
    assert (r['fusion']['stage_a_epochs'], r['fusion']['stage_b_epochs']) == (4, 20)
    assert r['adaptation']['final40']['visual_head_learning_rate'] == 5e-5
    assert r['adaptation']['final40']['scope'] == 'heads_motion_encoder'
    assert r['adaptation']['base12']['scope'] == 'heads'
    assert r['mechanism']['alpha'] == 3000
    assert r['mechanism']['class_weight_power'] == .75
    assert r['mechanism']['head_temperature'] == 1
    assert r['mechanism']['backbone_root'] == 'videomae_large'
    assert r['mechanism']['ridge_solver']=='lsqr'
    assert r['mechanism']['ridge_tol']==1e-5
    assert r['mechanism']['ridge_max_iter']==5000
    assert r['visual_student']['fusion_mode']=='gated'
    assert r['visual_student']['temporal_modeling'] is True
    assert r['visual_student']['spatial_region_modeling'] is False
    assert r['visual_student']['workers']==0
    assert r['fusion']['fusion_position']=='clip'
    assert r['source_bank']['rf_temperature'] == 3
    assert r['source_bank']['base_weight'] == .95
    assert r['source_bank']['rf_weight'] == .05
    assert r['postprocessing']['chosen_probability'] == .9805
    assert r['postprocessing']['other_probability'] == .0005
    assert r['visual_teacher']['families'] == ['early', 'late', 'window_mean', 'early_late', 'temporal_delta', 'kinetics']
    assert r['visual_teacher']['alphas'] == [300.,1000.,3000.,10000.]
    assert r['visual_teacher']['log_temperature_bounds'] == [-2.302585,2.302585]
    assert r['fullwindow_knn']['neighbors'] == 3
    assert r['fullwindow_knn']['neighbor_weight'] == .4


def test_source_values_have_file_symbol_line_and_digest():
    compiled = module().compile_source_ops(SOURCE_ROOT)
    assert compiled['schema_version'] == 3
    assert compiled['training_executed'] is False
    assert compiled['source_manifest_sha256']
    for profile, fields in compiled['evidence'].items():
        assert fields, profile
        for entry in fields:
            assert entry['path'] and entry['symbol'] and entry['line'] > 0
            assert len(entry['sha256']) == 64
            assert entry['origin'] in {'actual_override', 'source_default', 'source_constant', 'source_expression'}
    assert {'geometry','pixels','motion_processing','p12','global_mixture','repeat_group','sequence_gate','safe_base'} <= set(compiled['operators'])


def test_teacher_root_roster_and_exact_original_slot_order():
    compiled = module().compile_source_ops(SOURCE_ROOT)
    roster = module().compile_teacher_roster(compiled, ('videomae_large',))
    assert roster['retained_large_roots'] == ['videomae_large']
    assert len(roster['slots']) == 30
    names = [x['name'] for x in roster['slots']]
    assert names[0] == 'p89_safe_probability'
    assert names[-1] == 'p306_union_repeat_physical'
    assert sum(x['category'] == 'visual_head' for x in roster['slots']) == 6
    assert sum(x['category'] == 'mechanism' for x in roster['slots']) == 10
    assert next(x for x in roster['slots'] if x['name'] == 'p12_skeleton')['status'] != 'excluded'
    assert next(x for x in roster['slots'] if x['name'] == 'p12_thermal')['status'] != 'excluded'
    p128=next(x for x in roster['slots'] if x['name']=='p128_hierarchical_multimodal')
    assert p128['status']=='excluded'
    assert {'videomaev2','internvideo2'}<=set(p128['removed_large_roots'])
    for x in roster['slots']:
        assert x['evidence'] and x['status'] in {'retained', 'excluded', 'unresolved'}
        if x['status'] == 'excluded':
            assert x['removed_large_roots']


def test_roster_rejects_adding_another_large_root():
    compiled = module().compile_source_ops(SOURCE_ROOT)
    with pytest.raises(ValueError, match='one.*videomae_large'):
        module().compile_teacher_roster(compiled, ('videomae_large', 'vjepa2'))


def test_missing_final_configuration_is_a_gate_not_a_guessed_default():
    compiled = module().compile_source_ops(SOURCE_ROOT)
    roster = module().compile_teacher_roster(compiled, ('videomae_large',))
    assert roster['formal_generation_ready'] is False
    assert {'p12_sd_final_configuration', 'p12_thermal_final_configuration', 'p89_safe_base_producer_closure'} <= {x['code'] for x in roster['blocking_requirements']}
    assert all(x['reason'] and x['required_evidence'] for x in roster['blocking_requirements'])
    assert 'unresolved_teacher_lineage' in {x['code'] for x in roster['blocking_requirements']}


def test_source_compiler_does_not_import_execute_or_read_predictions(monkeypatch):
    op = module()
    def forbidden(*args, **kwargs):
        raise AssertionError('source import or array read forbidden')
    monkeypatch.setattr(importlib, 'import_module', forbidden)
    import numpy as np
    monkeypatch.setattr(np, 'load', forbidden)
    compiled = op.compile_source_ops(SOURCE_ROOT)
    assert compiled['training_executed'] is False
    assert len(compiled['consumed_files']) < 80
    assert all(Path(x).suffix in {'.py', '.md', '.json'} for x in compiled['consumed_files'])


def test_v3_descriptor_registration_and_v2_identity_unchanged(no_vote_fixture):
    op = module()
    v2 = load_protocol(no_vote_fixture[0])
    before = v2.identity()
    p = v3_protocol(no_vote_fixture)
    source = op.export_source_ops(SOURCE_ROOT, p.run_root/'protocol/source_ops_manifest.json', protocol=p)
    roster = op.resolve_teacher_roster(source, ('videomae_large',), protocol=p)
    reg = ArtifactRegistry(p)
    assert reg.verify(source, 'source_ops', 'public').kind == 'descriptor'
    assert reg.verify(roster, 'teacher_roster', 'public').parents == (source,)
    assert reg.read(source).fit_users == () and reg.read(source).rows is None
    assert load_protocol(no_vote_fixture[0]).identity() == before


def test_source_recipe_projection_matches_each_runtime_consumer():
    op=module();compiled=op.compile_source_ops(SOURCE_ROOT)
    values=op.runtime_recipe_values(compiled)
    assert values['visual_student']['head_learning_rate']==compiled['recipe']['visual_student']['head_learning_rate']
    assert values['fusion']['motion_aux_mask']=='source_batch_reduction'
    assert values['fusion']['visual_head_learning_rate']==compiled['recipe']['fusion']['visual_learning_rate']
    assert values['motion_pretrain']['contrastive_temperature']==compiled['recipe']['motion_pretrain']['temperature']
    assert values['motion']['external_transform']=='identity_source_descriptor'
    assert values['source_recipe']==compiled['recipe']


@pytest.mark.parametrize('stage,phase,fit', [('unknown_descriptor','public',()), ('source_ops','select',()), ('source_ops','public',('fit1',))])
def test_descriptor_cannot_bypass_roles(no_vote_fixture,stage,phase,fit):
    p = v3_protocol(no_vote_fixture)
    f = p.run_root/'public.json'
    f.write_text('{}')
    with pytest.raises(ValueError):
        ArtifactRegistry(p).register(stage=stage,kind='descriptor',phase=phase,files=[f],fit_users=fit)


def test_v2_rejects_v3_descriptor(no_vote_fixture):
    p = load_protocol(no_vote_fixture[0])
    f = p.run_root/'public.json';f.write_text('{}')
    with pytest.raises(ValueError, match='kind|descriptor'):
        ArtifactRegistry(p).register(stage='source_ops',kind='descriptor',phase='public',files=[f])


def test_descriptor_payload_and_roster_parent_are_not_optional(no_vote_fixture):
    op=module();p=v3_protocol(no_vote_fixture);r=ArtifactRegistry(p)
    f=p.run_root/'empty-descriptor.json';f.write_text('{}')
    with pytest.raises(ValueError,match='descriptor'):
        r.register(stage='source_ops',kind='descriptor',phase='public',files=[f],config={'source_contract_version':3})
    body=op.compile_source_ops(SOURCE_ROOT)
    source=op.export_source_ops(SOURCE_ROOT,p.run_root/'protocol/source_ops_manifest.json',protocol=p)
    roster=op.compile_teacher_roster(body,('videomae_large',))
    target=p.run_root/'protocol/teacher_roster.json';write_json(target,roster)
    with pytest.raises(ValueError,match='descriptor.*parent'):
        r.register(stage='teacher_roster',kind='descriptor',phase='public',files=[target],config={'source_contract_version':3})
    result=r.register(stage='teacher_roster',kind='descriptor',phase='public',files=[target],parents=[source],config={'source_contract_version':3})
    assert r.verify(result).parents==(source,)


def test_export_detects_source_content_drift(no_vote_fixture, tmp_path):
    op = module()
    p = v3_protocol(no_vote_fixture)
    ref = op.export_source_ops(SOURCE_ROOT,p.run_root/'protocol/source_ops_manifest.json',protocol=p)
    body = json.loads(next(Path(x) for x in ArtifactRegistry(p).read(ref).files).read_text())
    consumed = dict(body['consumed_files'])
    consumed[next(iter(consumed))] = {'bytes':1,'sha256':'0'*64}
    body['consumed_files'] = consumed
    with pytest.raises(ValueError, match='source.*mismatch'):
        op.verify_consumed_source(body,SOURCE_ROOT)
