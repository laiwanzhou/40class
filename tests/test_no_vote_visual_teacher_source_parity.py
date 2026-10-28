"""Task4 source-math and narrowly scoped readiness contracts."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from src.experiments.artifact_record import VerificationContext
from test_no_vote_source_ops import v3_protocol


def module():
    assert importlib.util.find_spec('src.experiments.visual_teacher_v3'), 'Task4 source teacher is missing'
    return importlib.import_module('src.experiments.visual_teacher_v3')


def test_source_feature_head_selection(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_math(p,VerificationContext(p))
    from src.experiments.visual_teacher import feature_sets,candidate_grid
    rng=np.random.default_rng(4);x=rng.normal(size=(7,2,3,1024)).astype(np.float32)
    k=rng.normal(size=(7,2,3,400)).astype(np.float32)
    for name,values in ops['feature_sets'](x,k).items():np.testing.assert_array_equal(feature_sets(x,k)[name],values)
    base=dict(accuracy=.8,balanced_accuracy=.6,macro_f1=.5)
    a=dict(base,class_weight_power=.5,alpha=300.)
    b=dict(base,class_weight_power=0.,alpha=10000.)
    c=dict(base,class_weight_power=0.,alpha=300.)
    assert m.choose_source_candidate([a,b,c]) is c
    recipe={'families':list(ops['feature_sets'](x,k)),'class_weight_powers':[0.,.5,.75],'alphas':[300.,1000.,3000.,10000.]}
    assert len(candidate_grid(recipe))==72


def test_calibration_train_only(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);context=VerificationContext(p);ops=m.source_math(p,context)
    rng=np.random.default_rng(5);x=rng.normal(size=(80,24));y=np.tile(np.arange(40),2)
    model,temperature=m.fit_calibrated_head(x,y,300.,.75,protocol=p,context=context)
    scores=ops['aligned_scores'](model,x)
    assert temperature==pytest.approx(ops['fit_temperature'](scores,y),rel=1e-12)
    assert .09999<=temperature<=10.0001
    # API accepts only fit X/y, with no development/final label source.
    unrelated_validation=np.arange(388)%40;unrelated_final=np.arange(609)%40
    unrelated_validation[:]=39;unrelated_final[:]=39
    _,again=m.fit_calibrated_head(x,y,300.,.75,protocol=p,context=context)
    assert again==temperature


def test_ridge_class_alignment(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_math(p,VerificationContext(p))
    rng=np.random.default_rng(3);x=rng.normal(size=(9,5));y=np.repeat([0,5,39],3)
    model=ops['make_model'](300.);model.fit(x,y)
    scores=ops['aligned_scores'](model,x)
    raw=model.decision_function(x);floor=raw.min(1)-np.maximum(np.ptp(raw,axis=1),1.)
    np.testing.assert_array_equal(scores[:,[0,5,39]],raw)
    np.testing.assert_array_equal(scores[:,1],floor)


def test_visual_mechanism_ridge_source_parity(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);context=VerificationContext(p);ops=m.source_math(p,context)
    rng=np.random.default_rng(7);features=rng.normal(size=(12,2,3,1024)).astype(np.float32);labels=np.arange(12)%3
    model,mean=m.fit_mechanism_arrays(features,labels,protocol=p,context=context)
    norm=features/np.maximum(np.linalg.norm(features,axis=-1,keepdims=True),1e-8)
    np.testing.assert_array_equal(mean,norm.mean(0))
    expected=ops['make_model'](3000.)
    expected.fit(norm.reshape(12,-1),labels,ridge__sample_weight=ops['sample_weights'](labels,.75))
    for name,values in ops['perturbations'](norm,mean).items():
        np.testing.assert_allclose(ops['aligned_scores'](model,values.reshape(12,-1)),
            ops['aligned_scores'](expected,values.reshape(12,-1)),rtol=0,atol=0)
    assert len(ops['perturbations'](norm,mean))==10


def test_global_readiness_not_changed_by_visual_preflight(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);context=VerificationContext(p)
    before=p.identity()
    with pytest.raises(ValueError,match='binding|Task3|source'):
        m.prepare_visual_capability(p,context)
    assert p.identity()==before and p.recipe.get('source_ops_bindings',{}).get('formal_generation_ready',False) is False


@pytest.fixture
def source_population(no_vote_fixture,tmp_path):
    from dataclasses import replace
    import csv,json,hashlib,yaml
    from test_no_vote_source_ops import SOURCE_ROOT
    from src.experiments.teammate_source import SOURCE_FILE_COUNT,SOURCE_MANIFEST_SHA256
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_source_ops import compile_source_ops
    from src.experiments.no_vote_manifest import prepare_inputs,read_public_rows,row_index
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.no_vote_cache_import import prepare_from_existing,import_public_cache
    from src.experiments.no_vote_types import write_json,canonical_bytes
    config,body=no_vote_fixture
    body['source']={'root':str(SOURCE_ROOT),'manifest':str(SOURCE_ROOT.parent/'source_manifest.json'),
        'expected_files':SOURCE_FILE_COUNT,'expected_sha256':SOURCE_MANIFEST_SHA256}
    config.write_text(yaml.safe_dump(body),encoding='utf-8');old=load_protocol(config)
    source=tmp_path/'canonical.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id']);w.writeheader()
        for user in ['fit1','dev1','target1']:
            for k in range(40):w.writerow(dict(sample_id=f'{user}{k}',class_id=k,user_id=user))
    prepare_inputs(source,old,old.run_root/'protocol',tmp_path/'private')
    root=tmp_path/'new';root.mkdir();(root/'protocol').mkdir()
    (root/'protocol/weights_manifest.json').write_bytes((old.run_root/'protocol/weights_manifest.json').read_bytes())
    compiled=compile_source_ops(SOURCE_ROOT);manifest=root/'protocol/source.json';write_json(manifest,compiled)
    teacher=compiled['recipe']['visual_teacher']
    p=replace(old,run_id='source-test',run_root=root,
        public_manifests={k:root/'protocol'/v.name for k,v in old.public_manifests.items()},
        supervised_labels={k:root/'protocol'/v.name for k,v in old.supervised_labels.items()},
        recipe={**old.recipe,'source_contract_version':3,'visual_teacher':teacher,
            'source_ops_bindings':{'formal_generation_ready':False,'manifest_path':str(manifest),
                'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}})
    c=VerificationContext(p);prepare_from_existing(old,p,context=c)
    write_json(root/'protocol/task3_geometry_parity.json',{'protocol_sha256':p.identity(),
        'pixel_crop_quality_times_exact':True,'retry_reason_bits_exact':True,'detector_executed':False})
    registry=ArtifactRegistry(old);features={}
    # Data values are synthetic and do not come from a competition label file.
    labels={r['sample_id']:int(r['class_id']) for r in csv.DictReader(old.supervised_labels['refit14'].open())}
    for partition in ['refit14','final4']:
        idx=row_index(read_public_rows(old.public_manifests[partition]));values=np.zeros((len(idx.sample_ids),2,3,1024),np.float32)
        for i,sid in enumerate(idx.sample_ids):values[i,:,:,labels.get(sid,i%40)]=1
        valid=np.ones(len(idx.sample_ids),bool)
        if partition=='final4':valid[-1]=False
        path=old.run_root/(partition+'_features.npz');np.savez(path,sample_ids=idx.sample_ids,class_ids=np.arange(40),features=values,
            kinetics_logits=np.zeros((len(values),2,3,400),np.float32),valid=valid)
        ref=registry.register(stage='visual_features',kind='raw_cache',phase='raw',files=[path],rows=idx,
            config={'partition':partition,'backbone_frozen':True,'feature_signature':'fixture-source-backbone'})
        features[partition]=import_public_cache(ref,p,idx,c,source_protocol=old)
    return p,c,features


def test_source_heads_full_phase_flow_and_scoped_gate(source_population,monkeypatch):
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
    from src.experiments import visual_teacher as teacher
    from src.experiments.no_vote_types import canonical_hash
    m=module();p,c,features=source_population;m.prepare_visual_capability(p,c)
    r=ArtifactRegistry(p,context=c);fake=p.run_root/'disallowed.json';fake.write_text('{}')
    cap=m.capability(p,c)
    with pytest.raises(ValueError,match='not ready'):
        r.register(stage='A7',kind='supervised_model',phase='select',files=[fake],parents=[cap],fit_users=p.partitions['train12'].users)
    train=load_stage_inputs(p,'train12','select');dev=load_stage_inputs(p,'development2','select')
    selection=teacher.select_visual_head(train,dev,features['refit14'],protocol=p,context=c)
    assert selection.config['family']=='early_late' and len(selection.config['family_configs'])==6
    assert selection.budget['candidates']==72 and p.recipe['source_ops_bindings']['formal_generation_ready'] is False
    prior=teacher.fit_class_prior(train,'select',protocol=p,context=c)
    rows=row_index(read_public_rows(p.public_manifests['development2']))
    pred=teacher.predict_visual_teacher(selection.fit_artifact,features['refit14'],rows,prior,protocol=p,context=c)
    record=r.verify(selection.fit_artifact)
    assert record.config['calibration_partition']=='train12' and record.config['head_temperature']!=1
    refit=load_stage_inputs(p,'refit14','refit')
    model=teacher.fit_visual_head(refit,features['refit14'],selection,protocol=p,context=c)
    refit_prior=teacher.fit_class_prior(refit,'refit',protocol=p,context=c)
    target=row_index(read_public_rows(p.public_manifests['final4']))
    final=teacher.predict_visual_teacher(model,features['final4'],target,refit_prior,protocol=p,context=c)
    assert r.verify(model).config['calibration_partition']=='refit14'
    assert np.allclose(final.prediction.probabilities[-1],np.full(40,.025))
    with pytest.raises(ValueError,match='phase|population'):
        teacher.predict_visual_teacher(model,features['refit14'],rows,refit_prior,protocol=p,context=c)
    mech=m.fit_visual_mechanism_head(train,features['refit14'],'select',protocol=p,context=c)
    mechanisms=m.predict_visual_mechanisms(mech,features['refit14'],rows,prior,protocol=p,context=c)
    assert len(mechanisms)==10 and r.verify(mech).stage=='visual_mechanism'
    assert all(r.verify(x.artifact).config['head_temperature']==1. for x in mechanisms.values())


def test_fullwindow_matches_original_uniform_window(no_vote_fixture,tmp_path):
    import cv2
    from src.experiments.visual_fullwindow import prepare_fullwindow_clips
    from src.experiments.teammate_source import verify_teammate_source,load_teammate_symbol
    p=v3_protocol(no_vote_fixture)
    report=verify_teammate_source(p.source_root,p.source_root.parent/'source_manifest.json',verify_contents=False)
    prepare=load_teammate_symbol(report,'build_p46_videomae_cache','prepare_trial')
    crop=load_teammate_symbol(report,'build_p46_videomae_cache','square_crop')
    read_ir=load_teammate_symbol(report,'build_p30_shared_dir_roi_feature_cache','read_ir')
    ir=tmp_path/'ir';ir.mkdir();cache=tmp_path/'roi/trial_roi_cache';cache.mkdir(parents=True)
    ids=np.array([f'{i:08}' for i in range(5)])
    for i,sid in enumerate(ids):assert cv2.imwrite(str(ir/('IR_'+sid+'.png')),np.full((480,640),i*20,np.uint8))
    boxes=np.full((5,7,4),np.nan,np.float32);valid=np.zeros((5,7),bool)
    boxes[:,0]=[10,20,310,420];valid[:,0]=True
    boxes[0,5]=[60,100,180,240];valid[0,5]=True
    from src.experiments.pose_roi_adapter import REGIONS
    arrays=dict(frame_ids=ids,acquisition_ids=np.array(['']*5),region_names=np.array(REGIONS),
        roi_boxes_xyxy=boxes,roi_valid=valid,roi_quality=np.ones((5,7),np.float32))
    source_id='opaque/record/sample';original_cache=cache/(source_id+'.npz');original_cache.parent.mkdir(parents=True)
    np.savez(original_cache,**arrays)
    expected,metadata=prepare({'source_id':source_id,'ir_dir':str(ir)},cache.parent)
    actual,indices=prepare_fullwindow_clips({'ir_path':str(ir)},arrays,crop,read_ir)
    np.testing.assert_array_equal(indices,metadata['frame_indices'])
    assert indices[0]==0 and indices[-1]==4
    for a,b in zip(actual,expected):
        for frame_a,frame_b in zip(a,b):np.testing.assert_array_equal(frame_a,frame_b)


def test_fullwindow_negative_limit_rejected_before_loading():
    from src.experiments.visual_fullwindow import extract_fullwindow
    with pytest.raises(ValueError,match='limit'):
        extract_fullwindow(None,'refit14',max_trials=-1)


def test_bounded_teacher_targets_never_rescan_full_features(tmp_path,monkeypatch):
    from src.experiments.no_vote_types import TeacherTargets,Prediction,RowIndex,ArtifactRef
    rows=RowIndex(tuple(f'{i:064x}' for i in range(4)),('fit',)*4,tuple(range(40)))
    prediction=Prediction(rows,np.zeros((4,40)),np.full((4,40),.025),np.ones(4,bool),ArtifactRef(tmp_path/'record.json','0'*64))
    features=np.zeros((4,2,3,1024),np.float32)
    assert hasattr(TeacherTargets,'from_validated_features'), 'v3 bounded teacher target constructor missing'
    original=np.isfinite
    def guarded(values,*args,**kwargs):
        assert values is not features, 'full feature table was scanned again'
        return original(values,*args,**kwargs)
    monkeypatch.setattr(np,'isfinite',guarded)
    result=TeacherTargets.from_validated_features(prediction,features)
    assert result.features is features
