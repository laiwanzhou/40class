"""Task11 source precedence/sharpening and no fabricated producer closure."""
import importlib
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('src.experiments.source_teacher_bank'), 'Task11 source bank seams are missing'
    return importlib.import_module('src.experiments.source_teacher_bank')


def test_p310_precedence_and_sharpening_keep_missing_prior():
    m=module();prior=np.arange(1,41,dtype=np.float32);prior/=prior.sum()
    targets=m.precedence_targets(np.array([1,2,3]),np.array([4,5,6]),np.array([1,8,9]),np.array([True,True,False]),prior)
    assert targets['precedence_prediction'].tolist()==[4,8,9]
    assert targets['precedence_route'].tolist()==[False,True,True]
    assert targets['target_mask'].tolist()==[True,True,False]
    assert np.isclose(targets['structured_distillation_probability'][0,4],.9805)
    np.testing.assert_array_equal(targets['structured_distillation_probability'][0,np.arange(40)!=4],np.full(39,.0005,np.float32))
    np.testing.assert_array_equal(targets['structured_distillation_probability'][2],prior)
    assert np.isclose(targets['structured_confidence'][0],.9805)
    np.testing.assert_allclose(targets['structured_distillation_probability'].sum(1),1,atol=1e-6)


def test_retained_roster_requires_exact_producers_and_refuses_removed_roots():
    m=module();slots=[{'name':'visual','status':'retained'},{'name':'excluded','status':'excluded'}]
    for producers in [{},{'visual':{'roots':['videomaev2']}},{'visual':{'roots':['videomae_large'],'fresh':False}},
                      {'visual':{'roots':['videomae_large'],'fresh':True},'excluded':{'roots':[],'fresh':True}}]:
        with pytest.raises(ValueError):m.validate_producer_roster(slots,producers)
    m.validate_producer_roster(slots,{'visual':{'roots':['videomae_large'],'fresh':True}})
    with pytest.raises(ValueError):m.validate_producer_roster([{'name':'unknown','status':'unresolved'}],{})


def test_a18_lineage_is_evidenced_removed_root_not_nonvisual(no_vote_fixture):
    m=module()
    from test_no_vote_source_ops import v3_protocol
    from src.experiments.artifact_record import VerificationContext
    p=v3_protocol(no_vote_fixture);before=p.identity();facts=m.a18_lineage(p,VerificationContext(p))
    assert facts['name']=='a18_best_session' and facts['status']=='excluded'
    assert set(facts['roots'])=={'videomaev2','internvideo2'}
    assert facts['actual_consumption_confirmed'] and p.identity()==before


def test_precedence_rejects_missing_classes_and_invalid_masks():
    m=module();prior=np.full(40,.025)
    for new,mask in [(np.array([40]),np.array([True])),(np.array([2]),np.array([1]))]:
        with pytest.raises(ValueError):m.precedence_targets(np.array([1]),np.array([1]),new,mask,prior)


def test_missing_producers_and_candidate_configs_do_not_open_formal_gate(no_vote_fixture,tmp_path):
    m=module()
    from dataclasses import replace
    from test_no_vote_source_ops import v3_protocol,SOURCE_ROOT
    from src.experiments.no_vote_source_ops import compile_source_ops,compile_teacher_roster
    from src.experiments.no_vote_types import write_json
    from src.experiments.artifact_record import VerificationContext
    from hashlib import sha256
    p=v3_protocol(no_vote_fixture);path=p.run_root/'protocol/old_roster.json'
    write_json(path,compile_teacher_roster(compile_source_ops(SOURCE_ROOT),('videomae_large',)))
    original=path.read_bytes();recipe={**p.recipe,'source_ops_bindings':{'roster_path':str(path),
        'roster_sha256':sha256(original).hexdigest(),'formal_generation_ready':False}}
    p=replace(p,recipe=recipe);identity=p.identity()
    body=m.task11_preflight(p,VerificationContext(p))
    assert path.read_bytes()==original and p.identity()==identity and body['parent_roster']['sha256']==sha256(original).hexdigest()
    assert next(x for x in body['slots'] if x['name']=='a18_best_session')['status']=='excluded'
    assert body['known_thermal_runner_output']!=body['required_thermal_historical_output']
    with pytest.raises(ValueError,match='safe_base_intermediate_adaptation_producer'):m.require_task11_formal_ready(body)
    # A pending preflight descriptor cannot be promoted by only flipping flags.
    with pytest.raises(ValueError):m.require_task11_formal_ready({**body,'formal_generation_ready':True,'blocking_requirements':[]})
    with pytest.raises(ValueError):m.require_task11_formal_ready({'formal_generation_ready':True,'blocking_requirements':[]})
