"""Frozen ID-bound outputs, once-only truth access and paired comparisons."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import hashlib

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('src.experiments.no_vote_evaluation'), 'Task13 evaluator missing'
    return importlib.import_module('src.experiments.no_vote_evaluation')


def samples(tmp_path):
    from src.experiments.no_vote_types import RowIndex,Prediction,ArtifactRef
    index=RowIndex(('a','b','c','d'),('u1','u1','u2','u2'),tuple(range(40)))
    def pred(guesses,name):
        prob=np.full((4,40),.0005,np.float32);prob[np.arange(4),guesses]=.9805
        return Prediction(index,np.log(prob),prob,np.array([True,True,True,False]),ArtifactRef(tmp_path/(name+'.json'),hashlib.sha256(name.encode()).hexdigest()))
    return pred([0,0,2,0],'before'),pred([0,1,0,0],'after'),np.array([0,1,2,3]),np.array([True,True,True,False])


def test_transfer_contrasts_same_population(tmp_path):
    a,b,y,mask=samples(tmp_path)
    value=module().transfer_contrast(a,b,y,mask,before_phase='refit',after_phase='refit')
    assert value['all']['rescue']==1 and value['all']['harm']==1 and value['all']['delta_correct']==0
    assert value['all']['disagreement']==2 and value['all']['delta_accuracy_points']==0
    assert value['adapted']['total']==3 and value['excluded_prior']['total']==1
    with pytest.raises(ValueError,match='phase'):module().transfer_contrast(a,b,y,mask,before_phase='select',after_phase='refit')
    wrong=replace(b,index=replace(b.index,sample_ids=('d','c','b','a')))
    with pytest.raises(ValueError,match='ID|population'):module().transfer_contrast(a,wrong,y,mask,before_phase='refit',after_phase='refit')


def test_accuracy_and_agreement_are_distinct(tmp_path):
    a,b,y,mask=samples(tmp_path)
    report=module().classification_metrics(b,y)
    assert report['accuracy']==.5 and report['correct']==2 and report['total']==4
    assert report['macro_f1_fixed40']==pytest.approx(3/80)
    assert report['class_support'][:4]==[1,1,1,1] and report['observed_class_count']==4
    assert np.mean(a.probabilities.argmax(1)==b.probabilities.argmax(1))==.5
    with pytest.raises(ValueError,match='label'):module().classification_metrics(b,np.array([0,1,2,40]))


def test_exact_final_ids_classes_prior(tmp_path):
    a,b,y,mask=samples(tmp_path);prior=np.full(40,.025,np.float32)
    prob=a.probabilities.copy();prob[~mask]=prior;good=replace(a,probabilities=prob,logits=np.log(prob))
    module().validate_final_prediction(good,good.index,mask,prior)
    with pytest.raises(ValueError,match='prior'):module().validate_final_prediction(a,a.index,mask,prior)
    wrong=replace(good,index=replace(good.index,sample_ids=('d','c','b','a')))
    with pytest.raises(ValueError,match='ID'):module().validate_final_prediction(wrong,good.index,mask,prior)


def test_read_labels_exact_ids_reordered_and_canonical_mapping(tmp_path):
    from src.experiments.no_vote_types import RowIndex
    from src.experiments.no_vote_manifest import NAMESPACE
    m=module();p=tmp_path/'truth.csv';p.write_text('sample_id,class_id\nb,1\na,0\n',encoding='utf-8')
    index=RowIndex(('a','b'),('u1','u2'),tuple(range(40)))
    np.testing.assert_array_equal(m.read_final_labels(p,index,label_format='opaque'),[0,1])
    p.write_text('sample_id,class_id\na,0\na,1\n',encoding='utf-8')
    with pytest.raises(ValueError,match='duplicate|ID'):m.read_final_labels(p,index,label_format='opaque')
    originals=['trial1','trial2'];ids=tuple(hashlib.sha256((NAMESPACE+x).encode()).hexdigest() for x in originals)
    index=RowIndex(ids,('u1','u2'),tuple(range(40)))
    p.write_text('sample_id,user_id,class_id\ntrial2,u2,1\nother,unused,39\ntrial1,u1,0\n',encoding='utf-8')
    np.testing.assert_array_equal(m.read_final_labels(p,index,label_format='canonical_manifest'),[0,1])


def test_reveal_persisted_before_truth_open_and_cannot_repeat(tmp_path,monkeypatch):
    m=module();frozen=tmp_path/'frozen.json';truth=tmp_path/'private.csv';output=tmp_path/'evaluation'
    index=samples(tmp_path)[0].index
    state=tmp_path/'run_state.json'
    frozen.write_text(json.dumps({'schema_version':1,'protocol_sha256':'identity','state_path':str(state),'candidate_order':list(m.CANDIDATES)}))
    state.write_text(json.dumps({'state':'frozen','protocol_sha256':'identity','generation':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()}}))
    calls=[]
    monkeypatch.setattr(m,'load_frozen_predictions',lambda path:(json.loads(frozen.read_text()),{},index,np.ones(4,bool),{}))
    def stop_on_read(*a,**kw):
        calls.append(True)
        assert json.loads(state.read_text())['state']=='revealed'
        raise ValueError('synthetic private file failure')
    monkeypatch.setattr(m,'read_final_labels',stop_on_read)
    with pytest.raises(ValueError,match='synthetic'):m.evaluate_frozen(frozen,truth,output)
    assert len(calls)==1
    with pytest.raises(ValueError,match='revealed|once'):m.evaluate_frozen(frozen,truth,output)
    assert len(calls)==1


def test_freeze_requires_v3_source_ops(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import ArtifactRegistry
    m=module();p=load_protocol(no_vote_fixture[0])
    with pytest.raises(ValueError,match='v3|source'):m.freeze_generation(p,ArtifactRegistry(p),{})
    p=replace(p,recipe={**p.recipe,'source_contract_version':3,'candidates':list(m.CANDIDATES)})
    with pytest.raises(ValueError,match='source|roster'):m.freeze_generation(p,ArtifactRegistry(p),{})


def test_metadata_only_context_never_reads_large_payload_or_registers(no_vote_fixture,tmp_path):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    p=load_protocol(no_vote_fixture[0]);c=VerificationContext(p,metadata_only=True)
    r=ArtifactRegistry(p,context=c);payload=p.run_root/'model.pt';payload.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='metadata|payload'):c.hash_file(payload)
    with pytest.raises(ValueError,match='metadata|read-only'):r.register(stage='x',kind='raw_cache',phase='raw',files=[payload])


def test_generation_closed_after_frozen(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.visual_student import phase_lock
    p=load_protocol(no_vote_fixture[0]);state=p.run_root/'run_state.json';state.write_text(json.dumps({'state':'frozen','protocol_sha256':p.identity()}))
    file=p.run_root/'new.json';file.write_text('{}')
    with pytest.raises(ValueError,match='frozen|generation'):ArtifactRegistry(p).register(stage='x',kind='raw_cache',phase='raw',files=[file])
    with pytest.raises(ValueError,match='frozen|generation'):
        with phase_lock(p.run_root/'A9-40/adapt'):pass


def test_metadata_graph_reuse_keeps_ancestors_and_detects_drift(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    p=load_protocol(no_vote_fixture[0]);r=ArtifactRegistry(p)
    file=p.run_root/'raw.json';file.write_text('{}')
    raw=r.register(stage='raw',kind='raw_cache',phase='raw',files=[file])
    file2=p.run_root/'model.json';file2.write_text('{}')
    model=r.register(stage='A1',kind='supervised_model',phase='refit',files=[file2],parents=[raw],fit_users=p.partitions['refit14'].users)
    file3=p.run_root/'other.json';file3.write_text('{}')
    other=r.register(stage='class_prior',kind='statistics',phase='refit',files=[file3],parents=[model],fit_users=p.partitions['refit14'].users)
    bounded=ArtifactRegistry(p,context=VerificationContext(p,metadata_only=True))
    bounded.verify(model);assert bounded.verify(other).complete
    raw.record_path.write_text('{}')
    with pytest.raises(ValueError,match='drift|hash'):bounded.verify(other)


def fake_frozen(tmp_path):
    from src.experiments.no_vote_types import canonical_bytes
    m=module();a,b,y,mask=samples(tmp_path);prior=np.full(40,.025,np.float32)
    payloads={};refs={}
    for i,name in enumerate(m.CANDIDATES):
        pred=a if name in {'A1','A7','A8'} else b
        prob=pred.probabilities.copy();prob[~mask]=prior
        path=tmp_path/(name+'.npz');np.savez(path,sample_ids=pred.index.sample_ids,class_ids=np.arange(40),
            logits=np.log(prob),probabilities=prob,valid=mask)
        payloads[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        record=tmp_path/(name+'-record.json');record.write_text('{}')
        refs[name]={'record_path':str(record),'sha256':hashlib.sha256(record.read_bytes()).hexdigest()}
    frozen=tmp_path/'frozen.json';state=tmp_path/'run_state.json'
    body=dict(schema_version=1,protocol_sha256='fixture',state_path=str(state),candidate_order=list(m.CANDIDATES),
        candidates=refs,payloads=payloads,rows=a.index,target_mask=mask.tolist(),prior=prior.tolist(),
        modalities={'ir':mask.tolist()},primary_contrast='A9-40_minus_A1')
    frozen.write_bytes(canonical_bytes(body));state.write_text(json.dumps(dict(state='frozen',protocol_sha256='fixture',
        generation=dict(path=str(frozen),sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()))))
    truth=tmp_path/'truth.csv';truth.write_text('sample_id,class_id\nd,3\nc,2\nb,1\na,0\n')
    return frozen,truth


def test_fixture_evaluation_reports_all_candidates_and_fixed_contrasts(tmp_path):
    m=module();frozen,truth=fake_frozen(tmp_path)
    output=m.evaluate_frozen(frozen,truth,tmp_path/'evaluation')
    report=json.loads(output.read_text(encoding='utf-8'))
    assert set(report['candidates'])==set(m.CANDIDATES)
    assert report['candidates']['A9-40']['correct']==2
    assert report['contrasts']['A9-40_minus_A1']['all']['rescue']==1
    assert report['contrasts']['A9-40_minus_A1']['all']['harm']==1
    assert json.loads((tmp_path/'run_state.json').read_text())['state']=='revealed'
    with pytest.raises(ValueError,match='once|revealed'):m.evaluate_frozen(frozen,truth,tmp_path/'other')


def test_modified_frozen_payload_rejected_before_truth_access(tmp_path,monkeypatch):
    m=module();frozen,truth=fake_frozen(tmp_path);(tmp_path/'A1.npz').write_bytes(b'changed')
    def forbid(*a,**kw):raise AssertionError('truth must not be read when frozen input drifts')
    monkeypatch.setattr(m,'read_final_labels',forbid)
    with pytest.raises(ValueError,match='drift'):m.evaluate_frozen(frozen,truth,tmp_path/'evaluation')
    assert json.loads((tmp_path/'run_state.json').read_text())['state']=='frozen'


def test_metadata_only_still_hashes_consumed_descriptor_payload(no_vote_fixture):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    from src.experiments.no_vote_types import write_json
    p=load_protocol(no_vote_fixture[0]);p=replace(p,recipe={**p.recipe,'source_contract_version':3})
    r=ArtifactRegistry(p);source=p.run_root/'protocol/source.json'
    write_json(source,dict(schema_version=3,training_executed=False,oof_enabled=False,
        source_manifest_sha256=p.recipe['asset_bindings']['source_manifest_sha256'],recipe={'fixture':True},operators={'fixture':{}},consumed_files=['x.py']))
    origin=r.register(stage='source_ops',kind='descriptor',phase='public',files=[source],config={'source_contract_version':3})
    file=p.run_root/'protocol/transform.json';write_json(file,dict(schema_version=3,cross_sample_fit=False,operators=['original']))
    descriptor=r.register(stage='source_transform',kind='descriptor',phase='public',files=[file],parents=[origin],config={'source_contract_version':3})
    write_json(file,dict(schema_version=3,cross_sample_fit=False,operators=['CHANGED']))
    with pytest.raises(ValueError,match='hash|drift'):
        ArtifactRegistry(p,context=VerificationContext(p,metadata_only=True)).verify(descriptor)


def test_formal_reveal_requires_review_bound_to_exact_generation(tmp_path):
    m=module();frozen,truth=fake_frozen(tmp_path)
    body=json.loads(frozen.read_text());body['execution_kind']='formal';frozen.write_text(json.dumps(body))
    state=tmp_path/'run_state.json';s=json.loads(state.read_text());s['generation']['sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest();state.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='review'):m.evaluate_frozen(frozen,truth,tmp_path/'evaluation')
    assert json.loads(state.read_text())['state']=='frozen'
    review=tmp_path/'protocol/freeze_review.json';review.parent.mkdir(exist_ok=True)
    review.write_text(json.dumps(dict(verdict='GO',generation_sha256='wrong',protocol_sha256='fixture',unresolved_p1_p2=[])))
    with pytest.raises(ValueError,match='review|generation'):m.approve_frozen_generation(frozen,review)
    review.write_text(json.dumps(dict(verdict='GO',generation_sha256=s['generation']['sha256'],protocol_sha256='fixture',unresolved_p1_p2=[])))
    m.approve_frozen_generation(frozen,review)
    assert m.evaluate_frozen(frozen,truth,tmp_path/'evaluation').exists()


def test_frozen_small_metadata_drift_blocks_truth(tmp_path,monkeypatch):
    m=module();frozen,truth=fake_frozen(tmp_path);small=tmp_path/'recipe.json';small.write_text('{}')
    body=json.loads(frozen.read_text());body['consumed_small_files']=[dict(path=str(small),sha256=hashlib.sha256(small.read_bytes()).hexdigest())]
    frozen.write_text(json.dumps(body));state=tmp_path/'run_state.json';s=json.loads(state.read_text());s['generation']['sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest();state.write_text(json.dumps(s))
    small.write_text('{"changed":true}')
    monkeypatch.setattr(m,'read_final_labels',lambda *a,**kw:pytest.fail('truth read before drift rejection'))
    with pytest.raises(ValueError,match='metadata|drift'):m.evaluate_frozen(frozen,truth,tmp_path/'evaluation')
    assert json.loads(state.read_text())['state']=='frozen'


def test_sealed_protocol_identity_is_cached_but_new_recipe_gets_new_identity(no_vote_fixture,monkeypatch):
    from src.experiments.no_vote_protocol import load_protocol
    p=load_protocol(no_vote_fixture[0]);expected=p.identity()
    def forbid(*a,**kw):raise AssertionError('immutable protocol identity must not resolve paths again')
    with monkeypatch.context() as scoped:
        scoped.setattr(Path,'resolve',forbid)
        assert p.identity()==expected
    changed=replace(p,recipe={**p.recipe,'seed':p.recipe['seed']+1})
    assert changed.identity()!=expected
