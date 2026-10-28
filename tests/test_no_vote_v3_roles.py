"""Task2 v3 roles do not weaken phase, row, roster or asset ownership."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.artifact_record import ArtifactRegistry
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import write_json
from test_artifact_record import prediction_rows


def registry(fixture):
    old=load_protocol(fixture[0])
    p=replace(old,recipe={**old.recipe,'source_contract_version':3,
        'source_ops_bindings':{'formal_generation_ready':True}})
    return ArtifactRegistry(p)


def model(r,stage,phase='select'):
    file=r.root/(stage.replace('/','_')+phase+'.bin');file.write_bytes(b'new model')
    return r.register(stage=stage,kind='supervised_model',phase=phase,files=[file],
        fit_users=r.protocol.partitions['train12' if phase=='select' else 'refit14'].users)


@pytest.mark.parametrize('stage',['source_expert/imu_rf','source_expert/p12_thermal','visual_head/early_late','visual_mechanism'])
def test_v3_expert_prediction_roles(no_vote_fixture,stage):
    r=registry(no_vote_fixture);rows=prediction_rows(r,'development2');own=model(r,stage)
    path=r.root/'prediction.json';path.write_text('{}')
    ref=r.register(stage=stage,kind='predictions',phase='predict',files=[path],parents=[own],
        rows=rows,config={'partition':'development2'})
    assert r.verify(ref).stage==stage
    other=model(r,'A2')
    with pytest.raises(ValueError,match='own|ancestor'):
        r.register(stage=stage,kind='predictions',phase='predict',files=[path],parents=[other],
            rows=rows,config={'partition':'development2'})


@pytest.mark.parametrize('stage',['mystery','source_expert/missing','visual_head/wrong'])
def test_v3_unknown_stage_rejected(no_vote_fixture,stage):
    r=registry(no_vote_fixture);rows=prediction_rows(r,'development2');own=model(r,'A2')
    path=r.root/'prediction.json';path.write_text('{}')
    with pytest.raises(ValueError,match='unknown'):
        r.register(stage=stage,kind='predictions',phase='predict',files=[path],parents=[own],rows=rows,config={'partition':'development2'})


def test_teacher_only_asset_requires_explicit_recipe_binding(no_vote_fixture):
    r=registry(no_vote_fixture);file=r.root/'thermal.pt';file.write_bytes(b'public imagenet')
    with pytest.raises(ValueError,match='asset|binding'):
        r.register(stage='source_initializer/thermal',kind='public_weights',phase='public',files=[file])


def test_teacher_bank_needs_roster_and_ordered_same_partition_predictions(no_vote_fixture):
    r=registry(no_vote_fixture);rows=prediction_rows(r,'development2');own=model(r,'source_expert/imu_rf')
    path=r.root/'pred.json';path.write_text('{}')
    pred=r.register(stage='source_expert/imu_rf',kind='predictions',phase='predict',files=[path],parents=[own],
        rows=rows,config={'partition':'development2'})
    file=r.root/'bank.npz';np.savez(file,sample_ids=rows.sample_ids,class_ids=np.arange(40),
        probabilities=np.full((40,1,40),.025),availability=np.ones((40,1),bool),fallback=np.zeros((40,1),bool))
    with pytest.raises(ValueError,match='roster'):
        r.register(stage='teacher_bank',kind='teacher_bank',phase='predict',files=[file],parents=[pred],rows=rows,
            config={'partition':'development2','experts':['source_expert/imu_rf']})


def test_v3_no_old_protocol_identity_changes(no_vote_fixture):
    p=load_protocol(no_vote_fixture[0]);before=p.identity();registry(no_vote_fixture)
    assert load_protocol(no_vote_fixture[0]).identity()==before


def test_valid_bank_and_source_transform_contract(no_vote_fixture):
    r=registry(no_vote_fixture)
    source_path=r.root/'protocol/source.json';roster_path=r.root/'protocol/roster.json'
    source={'schema_version':3,'training_executed':False,'oof_enabled':False,
        'source_manifest_sha256':r.protocol.recipe['asset_bindings']['source_manifest_sha256'],
        'recipe':{'defined':True},'operators':{'crop':{}},'consumed_files':['source.py']}
    write_json(source_path,source)
    import hashlib
    source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
    write_json(roster_path,{'schema_version':3,'training_executed':False,'oof_enabled':False,
        'source_ops_sha256':source_sha,'retained_large_roots':['videomae_large'],
        'slots':[{'name':f'expert{i}','status':'excluded'} for i in range(30)],'formal_generation_ready':True})
    p=replace(r.protocol,recipe={**r.protocol.recipe,'source_ops_bindings':{'formal_generation_ready':True,
        'manifest_path':str(source_path),'manifest_sha256':source_sha,'roster_path':str(roster_path),
        'roster_sha256':hashlib.sha256(roster_path.read_bytes()).hexdigest()}})
    r=ArtifactRegistry(p)
    source_ref=r.register(stage='source_ops',kind='descriptor',phase='public',files=[source_path],config={'source_contract_version':3})
    roster=r.register(stage='teacher_roster',kind='descriptor',phase='public',files=[roster_path],parents=[source_ref],config={'source_contract_version':3})
    transform=r.root/'protocol/transform.json';write_json(transform,{'schema_version':3,'cross_sample_fit':False,'operators':['source_resample']})
    fake_path=r.root/'fake-source.json';fake_path.write_text('{}')
    fake_source=r.register(stage='source_ops',kind='raw_cache',phase='raw',files=[fake_path])
    with pytest.raises(ValueError,match='transform'):
        r.register(stage='source_transform',kind='descriptor',phase='public',files=[transform],parents=[fake_source],config={'source_contract_version':3})
    assert r.register(stage='source_transform',kind='descriptor',phase='public',files=[transform],parents=[source_ref],config={'source_contract_version':3})
    rows=prediction_rows(r,'development2');own=model(r,'source_expert/imu_rf')
    path=r.root/'pred.json';path.write_text('{}')
    pred=r.register(stage='source_expert/imu_rf',kind='predictions',phase='predict',files=[path],parents=[own],rows=rows,config={'partition':'development2'})
    file=r.root/'bank.npz';np.savez(file,sample_ids=rows.sample_ids,class_ids=np.arange(40),probabilities=np.full((40,1,40),.025),
        availability=np.ones((40,1),bool),fallback=np.zeros((40,1),bool))
    ref=r.register(stage='teacher_bank',kind='teacher_bank',phase='predict',files=[file],parents=[roster,pred],rows=rows,
        config={'partition':'development2','experts':['source_expert/imu_rf']})
    assert r.verify(ref).kind=='teacher_bank'
