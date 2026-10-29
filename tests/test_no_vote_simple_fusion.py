"""Task9 diagnostics: exact joins, fit-only calibration, isolated ancestry."""
import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import softmax

from src.experiments.no_vote_types import ArtifactRef,Prediction,RowIndex


def module():
    assert importlib.util.find_spec('src.experiments.simple_fusion'), 'Task9 simple fusion is missing'
    return importlib.import_module('src.experiments.simple_fusion')


def prediction(tmp_path, scores, valid, ids=('a','b','c')):
    x=np.zeros((len(ids),40));x[:,:2]=scores
    index=RowIndex(ids,tuple('fit1' for _ in ids),tuple(range(40)))
    return Prediction(index,x,softmax(x,axis=1),np.asarray(valid,bool),ArtifactRef(tmp_path/'unused.json','a'*64))


def test_available_mean_excludes_invalid_and_all_missing_uses_prior(tmp_path):
    m=module();a=prediction(tmp_path,[[4,0],[4,0],[4,0]],[1,1,0])
    b=prediction(tmp_path,[[0,4],[0,4],[0,4]],[1,0,0]);prior=np.arange(1,41)/820
    p,v=m.combine_available([a,b],prior,index=a.index,temperatures=[1,1])
    np.testing.assert_allclose(p[0],(a.probabilities[0]+b.probabilities[0])/2)
    np.testing.assert_allclose(p[1],a.probabilities[1]);np.testing.assert_array_equal(p[2],prior)
    assert v.tolist()==[True,True,False]


def test_join_permutation_and_wrong_user_or_set_fail(tmp_path):
    m=module();a=prediction(tmp_path,[[1,0],[0,2],[3,0]],[1,0,1]);order=[2,0,1]
    b=Prediction(RowIndex(('c','a','b'),a.index.user_ids,a.index.class_ids),a.logits[order],a.probabilities[order],a.valid[order],a.artifact)
    p,v=m.combine_available([b],np.full(40,.025),index=a.index)
    np.testing.assert_allclose(p[[0,2]],a.probabilities[[0,2]]);assert v.tolist()==[True,False,True]
    for ids,users in [(('c','a','z'),a.index.user_ids),(('c','a','b'),('dev1','fit1','fit1'))]:
        bad=replace(b,index=RowIndex(ids,users,a.index.class_ids))
        with pytest.raises(ValueError,match='ID|user'):m.combine_available([bad],np.full(40,.025),index=a.index)
    with pytest.raises(ValueError):RowIndex(('a','a'),('fit1','fit1'),tuple(range(40)))
    with pytest.raises(ValueError):RowIndex(('a',),('fit1',),tuple(reversed(range(40))))


def test_temperature_fit_ignores_invalid_targets_and_stays_bounded(tmp_path):
    m=module();a=prediction(tmp_path,[[1,0],[0,1],[1,0]],[1,1,0]);y=np.array([0,1,0])
    t=m.fit_temperature(a,y);assert .25<=t<=4 and t<1
    assert m.fit_temperature(a,np.array([0,1,39]))==t
    assert m.fit_temperature(replace(a,valid=np.zeros(3,bool)),y)==1
    wrong=m.fit_temperature(a,np.array([1,0,0]));assert wrong>1


def write_labels(path,index,y):
    path.write_text('sample_id,class_id\n'+''.join(f'{s},{int(c)}\n' for s,c in zip(index.sample_ids,y)),encoding='utf-8')


def test_dev_can_select_policy_but_cannot_fit_temperatures(tmp_path):
    m=module();fit=prediction(tmp_path,[[1,0],[0,1],[1,0]],[1,1,1]);dev=prediction(tmp_path,[[2,0],[0,2],[2,0]],[1,1,1])
    dev=replace(dev,index=RowIndex(('d','e','f'),('dev1',)*3,dev.index.class_ids))
    fp=tmp_path/'fit.csv';dp=tmp_path/'dev.csv';write_labels(fp,fit.index,[0,1,0]);write_labels(dp,dev.index,[0,1,0])
    branches={k:fit for k in ['visual','skeleton','imu']};db={k:dev for k in branches}
    a=m.select_calibration(branches,fp,db,dp,prior=np.full(40,.025))
    write_labels(dp,dev.index,[1,0,1]);b=m.select_calibration(branches,fp,db,dp,prior=np.full(40,.025))
    assert a.config['fitted_temperatures']==b.config['fitted_temperatures']
    assert {x['policy'] for x in a.config['candidates']}=={'identity','fitted'}
    assert a.config['policy']=='identity'  # Accuracy/F1/worst-user ties prefer identity.


@pytest.mark.parametrize('stage',['A7','A8'])
def test_diagnostic_ancestry_cannot_enter_main_chain(no_vote_fixture,stage):
    module()
    from test_no_vote_v3_roles import registry,model
    from test_artifact_record import prediction_rows
    r=registry(no_vote_fixture);file=r.root/'diag.bin';file.write_bytes(b'diagnostic')
    diag=r.register(stage='A6-VSI',kind='supervised_model',phase='select',files=[file],
        fit_users=r.protocol.partitions['train12'].users,config={'role':'diagnostic_not_source_main'})
    with pytest.raises(ValueError,match='diagnostic'):
        if stage=='A7':
            r.register(stage='A7',kind='supervised_model',phase='select',files=[file],parents=[diag],fit_users=r.protocol.partitions['train12'].users)
        else:
            rows=prediction_rows(r,'development2');own=model(r,'A7')
            r.register(stage='A8',kind='predictions',phase='predict',files=[file],parents=[own,diag],rows=rows,config={'partition':'development2'})


def test_diagnostics_cannot_enter_teacher_bank(no_vote_fixture):
    module()
    from test_no_vote_v3_roles import registry
    from test_artifact_record import prediction_rows
    r=registry(no_vote_fixture);rows=prediction_rows(r,'development2');file=r.root/'diag.bin';file.write_bytes(b'x')
    # The bank must reject diagnostic experts before a roster or model can disguise their role.
    from types import SimpleNamespace
    from src.experiments.no_vote_roles import validate_bank
    record=SimpleNamespace(config={'partition':'development2','experts':['A6-VSI']},rows=rows,parents=(),files={})
    with pytest.raises(ValueError,match='diagnostic'):validate_bank(r,record)


def test_inference_combine_does_not_open_labels(tmp_path,monkeypatch):
    m=module();a=prediction(tmp_path,[[1,0],[0,1],[1,0]],[1,1,0]);original=Path.open
    def guard(path,*args,**kwargs):
        if path.suffix=='.csv':raise AssertionError('labels opened by inference')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    p,v=m.combine_available([a],np.full(40,.025),index=a.index,temperatures=[2])
    np.testing.assert_allclose(p[0],softmax(a.logits[0]/2));assert not v[-1]


def test_current_inputs_bind_reviewed_new_outputs_and_reject_legacy_or_drift(no_vote_fixture):
    m=module()
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import VerificationContext
    from src.experiments.no_vote_types import write_json
    from src.experiments.teammate_source import sha256_file
    p=load_protocol(no_vote_fixture[0]);root=p.run_root/'protocol';identity=p.identity()
    refs={pop:{k:{'record_path':str(root/f'{pop}_{k}.json'),'sha256':'a'*64} for k in m.BRANCHES} for pop in p.partitions}
    motion={pop:{k:refs[pop][k] for k in ['skeleton','imu']} for pop in p.partitions}
    models={phase:{'record_path':str(root/(phase+'.json')),'sha256':'b'*64} for phase in ['select','refit']}
    def proof(name,body):
        path=root/name;write_json(path,body);return {'path':str(path),'sha256':sha256_file(path)}
    student={'partitions':{pop:{'prediction':r['visual']} for pop,r in refs.items()}}
    sr=proof('task5_artifacts_ready.json',student)
    proof('task5_complete.json',{'protocol_sha256':identity,'complete':True,'independent_review':{'status':'GO'},
        'artifacts_ready_sha256':sr['sha256'],'partitions':student['partitions']})
    ready=proof('new_ready.json',{'protocol_sha256':identity,'outputs':motion,'models':models})
    review=proof('review.json',{'verdict':'GO','task8_revision':'ir_supervision_v2','unresolved_p1_p2':[],'ready':ready})
    complete=proof('new_complete.json',{'protocol_sha256':identity,'complete':True,'outputs':motion})
    body={'protocol_sha256':identity,'verdict':'GO','task8_revision':'ir_supervision_v2','complete':complete,
        'independent_review':review,'outputs':motion,'models':models}
    proof('task8_current.json',body)
    output,_=m.current_inputs(p,VerificationContext(p));assert output['final4']['skeleton']==m.ref(motion['final4']['skeleton'])
    for change in [{'task8_revision':'legacy'},{'verdict':'CONDITIONAL'},
                   {'outputs':{**motion,'final4':{**motion['final4'],'skeleton':refs['final4']['visual']}}}]:
        proof('task8_current.json',{**body,**change})
        with pytest.raises(ValueError):m.current_inputs(p,VerificationContext(p))


def test_fit_dev_population_overlap_is_rejected(tmp_path):
    m=module();a=prediction(tmp_path,[[1,0],[0,1],[1,0]],[1,1,1]);fp=tmp_path/'fit.csv'
    write_labels(fp,a.index,[0,1,0]);branches={k:a for k in m.BRANCHES}
    with pytest.raises(ValueError,match='overlap'):m.select_calibration(branches,fp,branches,fp,prior=np.full(40,.025))


def test_global_false_does_not_enable_unbound_diagnostic_model(no_vote_fixture):
    module()
    from dataclasses import replace
    from test_no_vote_v3_roles import registry
    from src.experiments.artifact_record import ArtifactRegistry
    r=registry(no_vote_fixture);p=replace(r.protocol,recipe={**r.protocol.recipe,'source_ops_bindings':{'formal_generation_ready':False}})
    r=ArtifactRegistry(p);file=r.root/'diag.bin';file.write_bytes(b'x')
    with pytest.raises(ValueError,match='not ready'):
        r.register(stage='A6-VSI',kind='supervised_model',phase='select',files=[file],
            fit_users=p.partitions['train12'].users,config={'role':'diagnostic_not_source_main'})
