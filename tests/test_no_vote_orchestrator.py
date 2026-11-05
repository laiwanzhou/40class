import importlib
import json
from pathlib import Path
import pytest

def module():
    assert importlib.util.find_spec('src.experiments.no_vote_orchestrator'),'orchestrator missing'
    return importlib.import_module('src.experiments.no_vote_orchestrator')

def test_dependency_order():
    m=module();names=[s.name for s in m.stages()]
    for a,b in [('heads_mechanisms','student'),('fullwindow','p12_global_knn'),('rf','motion_teacher'),
                ('p12_images','p12_global_knn'),('fusion','source_base'),('source_base','safe_bank_group_sequence_precedence'),
                ('safe_bank_group_sequence_precedence','target_adaptation'),('target_adaptation','freeze')]:
        assert names.index(a)<names.index(b)
    assert names[0]=='prepared_geometry'
    full=next(s for s in m.stages() if s.name=='fullwindow')
    assert full.markers==('fullwindow_features/refit14/full/artifact.json','fullwindow_features/final4/full/artifact.json')
    assert 'visual_review' in next(s for s in m.stages() if s.name=='student').dependencies
    assert 'task11_review' in next(s for s in m.stages() if s.name=='target_adaptation').dependencies

def test_isolated_smoke_reaches_sixteen_candidates_and_frozen(tmp_path):
    m=module();out=m.smoke(tmp_path/'smoke');body=json.loads(out.read_text())
    assert body['fixture'] and body['state']=='frozen' and len(body['candidates'])==16
    assert body['truth_reads']==body['raw_hash_reads']==body['ancestor_payload_reads']==0
    assert body['stage_order']==[s.name for s in m.stages()]
    assert body['filesystem_read_events']>0 and body['epoch_asset_checks']=='not_applicable_no_training'

def test_smoke_rejects_outside_payload_read(tmp_path,monkeypatch):
    m=module()
    monkeypatch.setattr(m.np,'savez',lambda *a,**kw:(tmp_path/'outside.json').read_text())
    with pytest.raises(ValueError,match='outside'):m.smoke(tmp_path/'isolated')

def test_failure_resume_and_marker_drift(tmp_path):
    m=module();calls=[];fail=[True]
    stages=(m.Stage('one',(),(),('one.json',)),m.Stage('two',('one',),(),('two.json',)))
    def runner(stage,log,context):
        calls.append(stage.name)
        if stage.name=='two' and fail[0]:raise RuntimeError('expected failure')
        (tmp_path/stage.markers[0]).write_text('{}')
    with pytest.raises(RuntimeError,match='failure'):m.execute(stages,tmp_path,'protocol',runner)
    assert json.loads((tmp_path/'orchestration/status.json').read_text())['status']=='failed'
    fail[0]=False;m.execute(stages,tmp_path,'protocol',runner)
    assert calls==['one','two','two']
    (tmp_path/'one.json').write_text('{"drift":true}')
    with pytest.raises(ValueError,match='drift'):m.execute(stages,tmp_path,'protocol',runner)

def test_revealed_run_refuses_dispatch_before_writes(tmp_path):
    m=module();(tmp_path/'run_state.json').write_text('{"state":"revealed"}')
    with pytest.raises(ValueError,match='revealed|closed'):m.execute(m.stages(),tmp_path,'p',lambda *x:pytest.fail('dispatch'))
    assert not (tmp_path/'orchestration').exists()

def test_producer_revision_drift_does_not_skip_completed_stage(tmp_path):
    m=module();stage=m.Stage('one',(),(('producer.py',),),('one.json',))
    def run(stage,log,context):(tmp_path/'one.json').write_text('{}')
    m.execute((stage,),tmp_path,'p',run,producer_fingerprints={'producer.py':'first'})
    with pytest.raises(ValueError,match='drift'):
        m.execute((stage,),tmp_path,'p',run,producer_fingerprints={'producer.py':'changed'})

def test_measured_history_handles_nested_and_unknown_timing():
    m=module()
    assert m.history_timing([{'training':{'seconds':2.5}},{'training':{'seconds':3}}])['training_seconds']==5.5
    assert m.history_timing([{'seconds':1.2}])['training_seconds']==1.2
    assert m.history_timing([{'epoch':1}])['training_seconds'] is None
